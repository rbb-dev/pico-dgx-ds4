"""The client half of media_highwater.sh: makes one test input and sends it to a running media worker.

    python3 media_worker_highwater.py image 40000000 --port 18951     one image of about that many pixels (RGBA)
    python3 media_worker_highwater.py doc 25 --port 18951             the 40 MP image by URL (this process serves it), then 25 inline 4 MP pages
    python3 media_worker_highwater.py bound 75497472 --port 18951     a frame one byte over that bound, then a small image on the same connection
    python3 media_worker_highwater.py clip 47 --port 18951            a clip of that many MiB, inline in the request
    python3 media_worker_highwater.py clipurl 128 --port 18951        a clip of that many MiB, by a URL this process serves
    python3 media_worker_highwater.py mixed 3 --port 18951            3 image requests (40 MP) and the 128 MiB URL clip, at once
                                                                      (--rounds N: the four requests N times over, one after the other round)

It runs in its own container (the media image with this file mounted), so its memory is not the worker's.
Prints one JSON line: what was sent and whether the worker took it or refused it, and with what words. A reply
is a run of frames (u32 len, JSON, u64 len, bytes) up to the end frame or an error frame; the frames are counted
and dropped as they arrive. The image is an RGBA PNG of about the pixel count asked, smooth with noise in the low
bits (4 B a pixel once decoded, a small file under the worker's per-image limit). The clip is mpeg4 noise frames
at a bitrate that reaches the size asked, in MiB, under the worker's per-clip limit.
"""
import argparse
import base64
import io
import json
import http.server
import os
import socket
import struct
import threading
import time

import numpy as np
from PIL import Image


def big_image(pixels=40_000_000, limit=32 << 20):
    """An RGBA PNG of `pixels` that stays under the worker's per-image byte limit: smooth gradients with a
    little noise in the low bits, so it decodes to 4 B a pixel but the file is small."""
    side = int(pixels ** 0.5)
    rng = np.random.default_rng(1)
    ramp = np.linspace(0, 252, side).astype(np.uint8)
    px = np.empty((side, side, 4), dtype=np.uint8)
    px[..., 0], px[..., 1], px[..., 2], px[..., 3] = ramp[None, :], ramp[:, None], ramp[::-1][None, :], 255
    px[..., :3] |= rng.integers(0, 2, (side, side, 3), dtype=np.uint8)
    out = io.BytesIO()
    Image.fromarray(px, "RGBA").save(out, "PNG", compress_level=6)
    if out.tell() > limit:
        raise SystemExit("the test image is %d B, over the worker's %d B limit" % (out.tell(), limit))
    return out.getvalue()


def big_clip(target=128 << 20):
    import av
    from fractions import Fraction

    out = io.BytesIO()
    with av.open(out, "w", format="mp4") as c:
        s = c.add_stream("mpeg4", rate=24)
        s.width, s.height, s.pix_fmt, s.bit_rate = 1280, 720, "yuv420p", 80_000_000
        rng = np.random.default_rng(2)
        i = 0
        while out.tell() < target - min(4 << 20, target // 4):
            f = av.VideoFrame.from_ndarray(rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8), format="rgb24")
            f.pts, f.time_base = i, Fraction(1, 24)
            for pk in s.encode(f):
                c.mux(pk)
            i += 1
        for pk in s.encode():
            c.mux(pk)
    return out.getvalue()


def exact(s, n):
    b = bytearray(n)
    view, got = memoryview(b), 0
    while got < n:
        k = s.recv_into(view[got:])
        if not k:
            raise ConnectionError("closed")
        got += k
    return b


def ask(port, req):
    """Send one request; read its reply frames to the end. Returns (last header, image frames, bytes of images, rows)."""
    with socket.create_connection(("127.0.0.1", port), timeout=600) as s:
        body = json.dumps(req).encode()
        s.sendall(struct.pack("<I", len(body)) + body)
        del body
        frames = nbytes = rows = 0
        while True:
            (n,) = struct.unpack("<I", exact(s, 4))
            hdr = json.loads(exact(s, n))
            (m,) = struct.unpack("<Q", exact(s, 8))
            exact(s, m)
            nbytes += m
            if "image" in hdr:
                frames, rows = frames + 1, rows + hdr["image"]["rows"]
            if not hdr.get("ok") or "image" not in hdr:
                return hdr, frames, nbytes, rows


def ask_on(s, req):
    """One request on an open connection; its reply frames to the end, as ask()."""
    body = json.dumps(req).encode()
    s.sendall(struct.pack("<I", len(body)) + body)
    frames = nbytes = rows = 0
    while True:
        (n,) = struct.unpack("<I", exact(s, 4))
        hdr = json.loads(exact(s, n))
        (m,) = struct.unpack("<Q", exact(s, 8))
        exact(s, m)
        nbytes += m
        if "image" in hdr:
            frames, rows = frames + 1, rows + hdr["image"]["rows"]
        if not hdr.get("ok") or "image" not in hdr:
            return hdr, frames, nbytes, rows


def over_the_bound(port, bound, sync=None):
    """A frame of bound + 1 bytes (sent in pieces), then a small image on the same connection."""
    small = base64.b64encode(big_image(1_000_000)).decode()
    with socket.create_connection(("127.0.0.1", port), timeout=600) as s:
        n = bound + 1
        s.sendall(struct.pack("<I", n))
        piece = b" " * (1 << 20)
        left = n
        while left:
            s.sendall(piece[:min(left, len(piece))])
            left -= min(left, len(piece))
        (k,) = struct.unpack("<I", exact(s, 4))
        refusal = json.loads(exact(s, k))
        (m,) = struct.unpack("<Q", exact(s, 8))
        exact(s, m)
        if sync:  # the caller reads the worker's memory.peak here, before the image below
            open(os.path.join(sync, "refused"), "w").close()
            for _ in range(600):
                if os.path.exists(os.path.join(sync, "go")):
                    break
                time.sleep(0.1)
        after = ask_on(s, {"images": [{"data": small}]})
    said = refusal.get("code") == "request_too_large" and str(n) in str(refusal.get("error"))
    ok = said and after[0].get("ok") and after[1] == 1
    return {"result": "ok" if ok else "refused: " + json.dumps([refusal, after[0]]), "frames": after[1], "rows": after[3], "reply_bytes": after[2]}


def outcome(hdr, frames, nbytes, rows):
    state = "ok" if hdr.get("ok") else "refused: " + str(hdr.get("error"))
    return {"result": state, "frames": frames, "rows": rows, "reply_bytes": nbytes}


def serve_clip(blob):
    """The clip on 127.0.0.1 (a port the OS picks), for the worker to fetch by URL; returns (server, url)."""
    class Clip(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Clip)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d/clip.mp4" % srv.server_address[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("image", "doc", "bound", "clip", "clipurl", "mixed"))
    ap.add_argument("size", type=int, help="pixels (image), pages (doc), bytes (bound), image requests (mixed) or MiB (clip, clipurl)")
    ap.add_argument("--port", type=int, default=18951)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--sync", default=None, help="bound: a folder; writes refused there after the error frame and waits for go")
    a = ap.parse_args()
    t0 = time.time()
    sent = 0
    try:
        if a.kind == "image":
            blob = big_image(a.size)
            sent = len(blob)
            res = outcome(*ask(a.port, {"images": [{"data": base64.b64encode(blob).decode()}]}))
        elif a.kind == "doc":
            big = big_image(40_000_000)
            srv, url = serve_clip(big)
            page = base64.b64encode(big_image(4_000_000)).decode()
            records = [{"url": url}] + [{"data": page}] * a.size
            sent = len(big) + len(page) * 3 // 4 * a.size
            res = outcome(*ask(a.port, {"images": records}))
        elif a.kind == "bound":
            res = over_the_bound(a.port, a.size, a.sync)
        elif a.kind == "clip":
            blob = big_clip(a.size << 20)
            sent = len(blob)
            req = {"videos": [{"data": base64.b64encode(blob).decode(), "max_frames": 32}], "frame_budget": 32}
            del blob
            res = outcome(*ask(a.port, req))
        elif a.kind == "clipurl":
            blob = big_clip(a.size << 20)
            sent = len(blob)
            srv, url = serve_clip(blob)
            res = outcome(*ask(a.port, {"videos": [{"url": url, "max_frames": 32}], "frame_budget": 32}))
        else:
            blob = big_clip(128 << 20)
            srv, url = serve_clip(blob)
            image_req = {"images": [{"data": base64.b64encode(big_image(40_000_000)).decode()}]}
            clip_req = {"videos": [{"url": url, "max_frames": 32}], "frame_budget": 32}
            sent = len(blob)
            answers = []

            def go(req):
                answers.append(outcome(*ask(a.port, req)))

            for _ in range(a.rounds):
                threads = [threading.Thread(target=go, args=(r,)) for r in [image_req] * a.size + [clip_req]]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
            states = sorted(x["result"] for x in answers)
            res = {"result": "ok" if states == ["ok"] * len(answers) else "refused: " + "; ".join(states), "requests": len(answers)}
    except Exception as e:  # noqa: BLE001
        res = {"result": "failed: %r" % e}
    print(json.dumps({"kind": a.kind, "size": a.size, "sent_bytes": sent, "seconds": round(time.time() - t0, 1), **res}), flush=True)


if __name__ == "__main__":
    main()
