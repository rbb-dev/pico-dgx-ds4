"""dsgx10 media worker: images and clips read into pixels, in a small process with no model and no GPU.

Image parsers and a video demuxer stay out of the engine process (untrusted bytes). The worker reads an
image (Image.open(...).convert("RGB")), plans its grid and pads it as the checkpoint's reference does
(inference/image_processor.py: plan_image_grid, ImageOps.pad), and answers its rows, hash and grid with the
padded image as u8 [n_vit_h * 14][n_vit_w * 14][3]. The grid functions repeat the reference's, which
imports torch; this process never does. The engine takes the pixels through its 256-entry table
(bf16((v / 255 - 0.5) / 0.5)) into the ViT's patches: test_media_worker.py holds the bits equal to the
reference's own patches.

Protocol (TCP, 127.0.0.1 only): request  = u32 len, JSON {"images": [record, ...], "max_rows": N}
                                        or {"videos": [record, ...], "frame_budget": N}
A reply is a run of frames, each u32 len, JSON, u64 len, bytes. Images: one frame per image, in order,
{"ok": true, "image": {rows, hash, grid, ms, bytes}} with that image padded, u8 [n_vit_h * 14][n_vit_w * 14][3],
then the end frame {"ok": true, "images": n} with no bytes. An error at any image ends the reply with
{"ok": false, "error", "code"?} and no bytes; a reader drops the frames before it. Videos: one frame
{"ok": true, "videos": [...]} (timestamped frames, each an image record), no bytes.
A record is what encoding.py's image blocks give: {"url": data-URL | http(s) URL} or {"data": b64}.
"max_rows" (optional) lowers --max-rows for one request: the cap on the rows of all its images together.
Over it the reply ends with {"ok": false, "code": "too_many_rows", "error"} and work stops at that image.

Memory: each thing is held once. A request's body is read into one buffer; a clip goes to its file in pieces (the
request's string is dropped before the frame process starts) and the frame process opens the file; an image is
padded and sent before the next one is read, so the worker never holds two padded images. One lock covers a
request from the moment its length is read to the end of its reply: only one request's body is in memory.
"""

import argparse
import base64
import hashlib
import io
import json
import math
import os
import socket
import socketserver
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageOps

p = argparse.ArgumentParser()
p.add_argument("--model", default="/model")  # only inference/config.json is read from it
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--port", type=int, default=18950)
p.add_argument("--max-images", type=int, default=100)
# The rows (image tokens) of all the images of one request together. Memory follows rows, not the number
# of images: each row is 10 KiB here, again in the front-end and again in the engine. dsfront sends its
# own cap (--max-image-rows, the same default) with each request; the lower of the two applies.
p.add_argument("--max-rows", type=int, default=40000)
p.add_argument("--max-bytes", type=int, default=32 << 20)  # per fetched image
# The most bytes one request frame may carry. The memory cap holds idle + one request + one image's decode, so the
# worker bounds the one thing the sender sizes. A larger frame is read and thrown away in pieces and answered
# with an error frame; the connection lives.
p.add_argument("--max-request-bytes", type=int, default=72 << 20)
# The most pixels an image may have (width x height, read from its header before anything is decoded). PIL's own
# check, set to the same value, warns at this size and refuses only over twice it: `plan` refuses at this size.
p.add_argument("--max-image-pixels", type=int, default=40_000_000)
p.add_argument("--max-video-bytes", type=int, default=128 << 20)
p.add_argument("--video-timeout-s", type=float, default=60)
a = p.parse_args()
Image.MAX_IMAGE_PIXELS = a.max_image_pixels

MMAP_THRESHOLD = 1 << 20


def fix_mmap_threshold():
    """Every block of 1 MiB or more is its own mapping, returned to the system when freed, whichever thread freed it
    (glibc's M_MMAP_THRESHOLD, -3; setting it also turns off the rule that raises the threshold as large blocks are
    freed). The worker does not start without it: another C library, or a refused call, would let a finished
    request's memory stay behind."""
    import ctypes

    try:
        libc = ctypes.CDLL(None)
        libc.mallopt.argtypes = [ctypes.c_int, ctypes.c_int]
        libc.mallopt.restype = ctypes.c_int
        done = libc.mallopt(-3, MMAP_THRESHOLD)
    except (OSError, AttributeError):
        done = 0
    if done != 1:
        raise SystemExit("media worker: mallopt(M_MMAP_THRESHOLD, 1 MiB) was not accepted by the C library; refusing to start")


fix_mmap_threshold()

_lock = threading.Lock()  # one request at a time, from its body to the end of its reply
_args = None


def log(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


def vision_args():
    """The checkpoint's vision settings (inference/config.json), read once."""
    global _args
    if _args is None:
        with open(os.path.join(a.model, "inference", "config.json")) as f:
            _args = SimpleNamespace(**json.load(f))
    return _args


# ---- the reference's grid plan (inference/image_processor.py) --------------------------------------------

IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height: int, best_width: int, patch_size: int, downsample_ratio: int):
    """Token grid the aligner produces from a patch grid of this pixel size."""
    return math.ceil((best_height // patch_size) / downsample_ratio), math.ceil(
        (best_width // patch_size) / downsample_ratio
    )


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    """Largest aspect-preserving pixel size whose token grid still fits in max_n_token."""
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:  # very tall: collapse to a single column
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:  # very wide: collapse to a single row
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    return math.floor(height * beta / patch_size) * patch_size, math.floor(width * beta / patch_size) * patch_size


def safe_resize(height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token):
    """Shrink the pixel size until the image costs at most max_n_token LLM tokens."""
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token)
        n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


def plan_image_grid(width: int, height: int, args):
    """Resize plan for an image of the given original size; a pure function of its arguments."""
    p = args.vision_patch_size
    if args.vision_max_wh_ratio is not None and width > height * args.vision_max_wh_ratio:
        width = height * args.vision_max_wh_ratio
    if 0 < width * height < args.vision_min_pixels:
        ratio = (args.vision_min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    return safe_resize(height, width, best_height, best_width, p, args.vision_downsample_ratio, args.vision_max_n_token)


def load_image_bytes(record) -> bytes:
    """Image bytes from raw or base64 data, an Anthropic source with data, or a data: URL (the reference's
    load_image_bytes without its URL and path branches: `fetch` takes those, with its limits)."""
    data = record.get("data")
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return base64.b64decode(data)

    source = record.get("source")
    if isinstance(source, dict):
        if source.get("data") is not None:
            return base64.b64decode(source["data"])
        if source.get("url"):
            return load_image_bytes({"url": source["url"]})

    url = record.get("url")
    if isinstance(url, str) and url.startswith("data:"):
        header, _, payload = url.partition(",")
        if ";base64" not in header:
            raise ValueError(f"Unsupported data URL encoding: {header}")
        return base64.b64decode(payload)

    raise ValueError(f"Cannot load image from record: {list(record.keys())}")


class MediaError(ValueError):
    """A request's image or video cannot be taken; the message is ours and goes to the client as it is."""


def _public(host, kind="image"):
    """The address a fetch connects to for `host`: resolved here once, and public (no loopback, private, link-local or
    reserved: no SSRF into the cluster). Every address the name resolves to must be public. `kind` names the input in the
    message."""
    import ipaddress

    first = None
    for fam, _, _, _, sa in socket.getaddrinfo(host, None):
        ip = ipaddress.ip_address(sa[0])
        if not ip.is_global or ip.is_multicast:
            raise MediaError("%s URL host %s is not a public address" % (kind, host))
        first = first or sa[0]
    if first is None:
        raise MediaError("%s URL host %s does not resolve" % (kind, host))
    return first


def _pinned(conn_class, kind):
    """A connection factory for urllib: the name is resolved and checked once (`_public`) and the socket is made to that
    address, so the name cannot answer another address at the connect; the Host header and the TLS name stay the host's."""

    def make(host, **kw):
        conn = conn_class(host, **kw)

        def connect_to(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None):
            name, port = address
            return socket.create_connection((_public(name, kind) or name, port), timeout, source_address)

        conn._create_connection = connect_to
        return conn

    return make


_PIECE = 1 << 20  # what a clip is read, or decoded from its base64, at a time
_NOT_BASE64 = bytes(set(range(256)) - set(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="))


def _inline(record):
    """(holder, key) of the record's inline payload (its base64 text or raw bytes), in load_image_bytes's order,
    or None."""
    if isinstance(record.get("data"), (bytes, str)):
        return record, "data"
    source = record.get("source")
    if isinstance(source, dict) and source.get("data") is not None:
        return source, "data"
    url = record.get("url")
    if isinstance(url, str) and url.startswith("data:"):
        return record, "url"
    return None


def _inline_into(record, into, limit, kind):
    """A record's inline payload decoded in pieces into the open file `into`, the limit counted as it goes; the
    payload is then dropped from the record, so the request does not keep the text beside the file. Returns the
    byte count."""
    found = _inline(record)
    if found is None:
        raise ValueError(f"Cannot load image from record: {list(record.keys())}")
    holder, key = found
    value = holder[key]
    total = 0

    def put(chunk):
        nonlocal total
        total += len(chunk)
        if total > limit:
            raise MediaError("%s larger than %d bytes" % (kind, limit))
        into.write(chunk)

    if isinstance(value, bytes):
        put(value)
    else:
        start = 0
        if key == "url":
            comma = value.find(",")
            header = value if comma < 0 else value[:comma]
            if ";base64" not in header:
                raise ValueError(f"Unsupported data URL encoding: {header}")
            start = len(value) if comma < 0 else comma + 1
        carry = b""
        for at in range(start, len(value), _PIECE):
            piece = carry + value[at:at + _PIECE].encode("ascii").translate(None, _NOT_BASE64)
            cut = len(piece) - len(piece) % 4
            carry = piece[cut:]
            if cut:
                put(base64.b64decode(piece[:cut]))
        if carry:
            put(base64.b64decode(carry))
    del value
    del holder[key]
    return total


def fetch(record, max_bytes=None, kind="image", into=None):
    """Image (or, for kind "video", clip) bytes from a record (image_processor.load_image_bytes, with a
    size cap for URLs). The messages name the input by `kind`. With `into`, an open binary file, the bytes are
    written to it in pieces, never held whole, and the byte count is returned."""
    limit = a.max_bytes if max_bytes is None else max_bytes
    url = record.get("url")
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        import http.client
        from urllib.request import HTTPHandler, HTTPSHandler, Request, build_opener

        class Http(HTTPHandler):
            def http_open(self, req):
                return self.do_open(_pinned(http.client.HTTPConnection, kind), req)

        class Https(HTTPSHandler):
            def https_open(self, req):
                return self.do_open(_pinned(http.client.HTTPSConnection, kind), req, context=self._context)

        with build_opener(Http, Https).open(Request(url, headers={"User-Agent": "dsgx10-vision/1"}), timeout=20) as r:
            if into is None:
                b = r.read(limit + 1)
                if len(b) > limit:
                    raise MediaError("%s larger than %d bytes" % (kind, limit))
                return b
            total = 0
            while True:
                piece = r.read(_PIECE)
                if not piece:
                    return total
                total += len(piece)
                if total > limit:
                    raise MediaError("%s larger than %d bytes" % (kind, limit))
                into.write(piece)
    if isinstance(url, str) and url and not url.startswith("data:"):
        raise MediaError("%s url must be a data: URL or http(s)" % kind)  # no local paths over the API
    src = record.get("source")
    if isinstance(src, dict) and src.get("url") and not src.get("data"):
        return fetch({"url": src["url"]}, limit, kind, into)
    if into is not None:
        return _inline_into(record, into, limit, kind)
    raw = load_image_bytes(record)
    if len(raw) > limit:
        raise MediaError("%s larger than %d bytes" % (kind, limit))
    return raw


def fetch_video(record, into):
    """A clip's bytes, written to the open file `into` (the byte count is returned). Our own refusals (MediaError) go to the client as they are; anything a library
    raises while fetching or unpacking (urllib's "<urlopen error [Errno -2] ...>", base64's "Incorrect
    padding", a socket timeout) is logged and replaced by plain words."""
    from urllib.error import HTTPError

    try:
        return fetch(record, a.max_video_bytes, "video", into)
    except MediaError:
        raise
    except HTTPError as e:
        raise MediaError("could not fetch the video: the server answered HTTP %d" % e.code) from e
    except Exception as e:  # noqa: BLE001
        log("video fetch failed:", repr(e))
        url = record.get("url") or (record.get("source") or {}).get("url") if isinstance(record, dict) else None
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            raise MediaError("could not fetch the video from its URL") from e
        raise MediaError("the video data is not valid base64") from e


def decode_videos(req):
    videos = req.get("videos")
    budget = req.get("frame_budget", a.max_images)
    if not isinstance(budget, int) or isinstance(budget, bool) or not 1 <= budget <= a.max_images:
        raise MediaError("invalid video frame budget")
    if not isinstance(videos, list) or not 1 <= len(videos) <= budget:
        raise MediaError("not enough frame budget for the videos")
    results = []
    for record in videos:
        count = record.get("max_frames", 32)
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= a.max_images:
            raise MediaError("max_frames must be an integer between 1 and %d" % a.max_images)
        count = min(count, budget // len(videos))
        with tempfile.TemporaryDirectory(prefix="vision-video-") as tmp:
            clip = Path(tmp) / "clip"
            with open(clip, "wb") as f:
                size = fetch_video(record, f)
            try:
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).with_name("video_frames.py")), str(clip), "--max-frames", str(count)],
                    capture_output=True, timeout=a.video_timeout_s, check=False,
                )
            except subprocess.TimeoutExpired as e:
                raise MediaError("video decoding timed out") from e
        # The decoder answers in plain words (video_frames.VideoError). Whatever else comes out of the
        # subprocess (a crash, no JSON, a library's error text on stderr) is logged here and the
        # client is told the clip is not a readable video.
        if result.stderr.strip():
            log("video decoder stderr:", result.stderr[-400:].decode(errors="replace").strip())
        try:
            decoded = json.loads(result.stdout)
        except (ValueError, UnicodeError):
            decoded = None
        if not isinstance(decoded, dict) or result.returncode or not decoded.get("ok"):
            said = decoded.get("error") if isinstance(decoded, dict) else None
            if not isinstance(said, str) or not said or result.returncode not in (0, 1):
                log("video decoder failed: exit %s, %d B of output" % (result.returncode, len(result.stdout)))
                said = "not a readable video"
            raise MediaError(said)
        if not isinstance(decoded.get("duration"), (int, float)) or not isinstance(decoded.get("frames"), list):
            raise MediaError("not a readable video")
        results.append({"duration": decoded["duration"], "frames": decoded["frames"]})
        log(f"video {size} B, {decoded['duration']:.3f} s -> {len(decoded['frames'])} timestamped frames")
    return results


def plan(raw: bytes, args=None):
    """One image read and its grid planned (the reference's load_image up to the padding): the RGB image,
    the rows its span will take and the grid (n_vit_h, n_vit_w, n_llm_h, n_llm_w). CPU work."""
    args = args or vision_args()
    with Image.open(io.BytesIO(raw)) as source:
        pixels = source.width * source.height  # the header's size: nothing is decoded yet
        if pixels > a.max_image_pixels:
            raise Image.DecompressionBombError(
                f"Image size ({pixels} pixels) exceeds limit of {a.max_image_pixels} pixels, could be decompression bomb DOS attack."
            )
        image = source.convert("RGB")
    n_llm_h, n_llm_w, best_height, best_width = plan_image_grid(image.width, image.height, args)
    p = args.vision_patch_size
    return image, num_image_tokens(n_llm_h, n_llm_w), (best_height // p, best_width // p, n_llm_h, n_llm_w)


def pad(image, grid, args=None):
    """The image at its planned pixel size as u8 [n_vit_h * p][n_vit_w * p][3] (the reference's resize or
    ImageOps.pad with grey 127)."""
    import numpy as np

    args = args or vision_args()
    p = args.vision_patch_size
    height, width = grid[0] * p, grid[1] * p
    if args.vision_max_wh_ratio is not None and image.width >= args.vision_max_wh_ratio * image.height:
        image = image.resize((width, height))
    else:
        image = ImageOps.pad(image, (width, height), color=(127, 127, 127))
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


def measure(raw: bytes, args=None):
    """(padded u8 pixels, rows, grid) of one image."""
    image, rows, grid = plan(raw, args)
    return pad(image, grid, args), rows, grid


class TooManyRows(ValueError):
    """The images of one request take more rows than the cap (the reply's code: too_many_rows)."""


def row_limit(req):
    """The cap on a request's rows in total: --max-rows, or the request's own max_rows when lower."""
    limit = a.max_rows
    asked = req.get("max_rows")
    if isinstance(asked, int) and not isinstance(asked, bool) and asked > 0:
        limit = min(limit, asked)
    return limit


def handle(req):
    """The images of a request, one at a time: yields (record, padded u8 pixels) for each, in order. An image is
    fetched, planned and padded only when the one before it has been taken (the caller sends it and lets go), so
    no two padded images are held. Raises at the first image that cannot be taken; the images before it were
    already yielded."""
    recs = req.get("images") or []
    if not recs or len(recs) > a.max_images:
        raise ValueError("between 1 and %d images per request" % a.max_images)
    limit, total = row_limit(req), 0
    for k, r in enumerate(recs, 1):
        t0 = time.time()
        raw = fetch(r)
        size = len(raw)
        digest = int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")
        image, rows, grid = plan(raw)
        del raw
        total += rows
        if total > limit:  # stop here, before this image is padded: it and the rest are never taken
            raise TooManyRows("at most %d image tokens per request; the first %d of %d images already take %d"
                              % (limit, k, len(recs), total))
        pixels = pad(image, grid)
        del image
        ms = 1e3 * (time.time() - t0)
        log(f"image {size} B -> {grid[2]}x{grid[3]} tokens ({rows} rows) in {ms:.0f} ms")
        yield {"rows": rows, "hash": digest, "grid": grid, "ms": round(ms, 1), "bytes": size}, pixels
        del pixels


def recv_exact(s, n):
    """n bytes from the socket, read straight into the one buffer that is returned."""
    b = bytearray(n)
    view = memoryview(b)
    got = 0
    while got < n:
        k = s.recv_into(view[got:])
        if not k:
            raise ConnectionError("closed")
        got += k
    return b


class ClientGone(Exception):
    """The peer stopped reading its reply."""


def send_frame(s, hdr, data=b""):
    """One reply frame: u32 len, JSON, u64 len, bytes (the bytes are sent as they are, not joined to the header)."""
    j = json.dumps(hdr).encode()
    try:
        s.sendall(struct.pack("<I", len(j)) + j + struct.pack("<Q", len(data)))
        if len(data):
            s.sendall(data)
    except OSError as e:
        raise ClientGone() from e


def answer(s, n):
    """Read one request body of n bytes and send its reply. False when the peer closed before the body was whole."""
    try:
        body = recv_exact(s, n)
    except ConnectionError:
        return False
    try:
        req = json.loads(body)
        body = None  # the request is its parsed form from here on
        if "videos" in req:
            try:
                videos = decode_videos(req)
            except MediaError:
                raise
            except Exception as e:  # noqa: BLE001  (not one of our refusals: its text stays in the log)
                log("video error:", repr(e))
                raise MediaError("the video could not be processed") from e
            send_frame(s, {"ok": True, "videos": videos})
        else:
            count = 0
            for item in handle(req):
                count += 1
                send_frame(s, {"ok": True, "image": item[0]}, memoryview(item[1]).cast("B"))
                item = None
            send_frame(s, {"ok": True, "images": count})
    except ClientGone:
        raise
    except Exception as e:  # noqa: BLE001  (the error goes back to the client as a 400)
        log("error:", repr(e))
        hdr = {"ok": False, "error": str(e)[:500]}
        if isinstance(e, TooManyRows):
            hdr["code"] = "too_many_rows"
        send_frame(s, hdr)
    return True


def discard(s, n):
    """Read and throw away n bytes in 1 MiB pieces; False when the peer closed first."""
    piece = bytearray(min(n, _PIECE))
    view = memoryview(piece)
    while n > 0:
        k = s.recv_into(view[:min(n, len(piece))])
        if not k:
            return False
        n -= k
    return True


class H(socketserver.BaseRequestHandler):
    def handle(self):
        s = self.request
        while True:
            try:
                (n,) = struct.unpack("<I", recv_exact(s, 4))
            except ConnectionError:
                return
            try:
                if n > a.max_request_bytes:  # no lock and nothing held: the bytes are thrown away as they arrive
                    msg = "the request is %d bytes; the media worker takes at most %d" % (n, a.max_request_bytes)
                    log("error:", msg)
                    if not discard(s, n):
                        return
                    send_frame(s, {"ok": False, "code": "request_too_large", "error": msg})
                    continue
                with _lock:  # taken before the body is read: the bytes wait in the sender
                    if not answer(s, n):
                        return
            except ClientGone:
                return


class S(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    srv = S((a.host, a.port), H)
    log(f"listening on {a.host}:{a.port} (mmap threshold {MMAP_THRESHOLD >> 20} MiB)")
    srv.serve_forever()
