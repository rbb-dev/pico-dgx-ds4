"""CPU tests of the media worker's request limits; no torch, no model, no GPU, no network (the encoder's own
variant, VISION_MODULE=vision_encoder, needs torch).

    python3 -m unittest discover -s tools/vision -p test_encoder_limits.py -v

media_worker.py is imported against a stand-in config directory, with stand-ins for the image reading (`plan`)
and the padding (`pad`), so only the request handling runs: how many rows a request may take in total, where
work stops, and what the reply says. Set VISION_MODULE=vision_encoder to run the same file against the encoder.
"""
import base64
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


MODULE = os.environ.get("VISION_MODULE", "media_worker")


def load_media_worker(max_rows=100):
    """The worker imported against a stand-in model directory (a config only), with --max-rows MAX_ROWS.
    A pixel is a byte: the stand-in "image" b"30" has 30 rows and pads to a [1, 30, 3] array."""
    model = tempfile.mkdtemp(prefix="vision-test-model-")
    os.makedirs(os.path.join(model, "inference"))
    with open(os.path.join(model, "inference", "config.json"), "w") as f:
        json.dump({"vision_n_layers": 1}, f)
    sys.modules.pop(MODULE, None)
    argv, sys.argv = sys.argv, [MODULE + ".py", "--model", model, "--max-rows", str(max_rows)]
    sys.path.insert(0, HERE)
    try:
        enc = __import__(MODULE)
    finally:
        sys.argv = argv
        sys.path.remove(HERE)
    enc.planned = []
    enc.padded = []

    def plan(raw):
        enc.planned.append(raw)
        return raw, int(raw), (1, 1, 1, 1)

    def pad(image, grid):
        enc.padded.append(image)
        return np.zeros((1, int(image), 3), dtype=np.uint8)

    enc.plan = plan
    enc.pad = pad
    enc.row_bytes = 3
    return enc


def load_vision_encoder(max_rows=100):
    """vision_encoder imported against a stand-in checkpoint directory, with --max-rows MAX_ROWS."""
    import torch

    model = tempfile.mkdtemp(prefix="vision-test-model-")
    os.makedirs(os.path.join(model, "inference"))
    with open(os.path.join(model, "inference", "config.json"), "w") as f:
        json.dump({"vision_n_layers": 1}, f)
    ip = types.ModuleType("image_processor")

    def load_image_bytes(record):
        data = record.get("data") or record.get("url", "").split(",", 1)[-1]
        return base64.b64decode(data)

    ip.load_image_bytes = load_image_bytes
    vision = types.ModuleType("vision")
    vision.Aligner = vision.ViT = object
    vision.F = None
    saved = {k: sys.modules.get(k) for k in ("image_processor", "vision", "vision_encoder")}
    sys.modules.update({"image_processor": ip, "vision": vision})
    sys.modules.pop("vision_encoder", None)
    argv, sys.argv = sys.argv, ["vision_encoder.py", "--model", model, "--max-rows", str(max_rows)]
    sys.path.insert(0, HERE)
    try:
        import vision_encoder as enc
    finally:
        sys.argv = argv
        sys.path.remove(HERE)
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    enc.calls = []

    def measure(raw):  # the "image" is its row count in ASCII: b"30" is an image of 30 rows
        enc.measured.append(raw)
        return raw, torch.zeros(int(raw), dtype=torch.int64), (1, 1, 1, 1)

    def encode(patches, types, grid):  # the ViT's stand-in: counts what is really encoded
        enc.calls.append(patches)
        return torch.zeros((types.numel(), 8), dtype=torch.bfloat16), grid

    enc.measured = []
    enc.measure = measure
    enc.encode = encode
    enc.torch.cuda.synchronize = lambda: None
    enc.padded, enc.planned = enc.calls, enc.measured  # the names the tests below use
    enc.row_bytes = 8 * 2
    return enc


def load_encoder(max_rows=100):
    return (load_vision_encoder if MODULE == "vision_encoder" else load_media_worker)(max_rows)


def image(rows):
    return {"url": "data:image/png;base64," + base64.b64encode(str(rows).encode()).decode()}


def drain(enc, req):
    """(records, padded bytes) of every image `handle` takes, one at a time as the worker sends them."""
    taken = enc.handle(req)
    if isinstance(taken, tuple):  # the encoder variant answers whole
        return taken
    records, data = [], []
    for record, pixels in taken:
        records.append(record)
        data.append(pixels.tobytes())
        del pixels
    return records, b"".join(data)


def read_reply(enc, s):
    """The reply to one request: (header, bytes) frames up to the end frame, an error frame, or a whole answer."""
    frames = []
    while True:
        (n,) = struct.unpack("<I", enc.recv_exact(s, 4))
        hdr = json.loads(enc.recv_exact(s, n))
        (m,) = struct.unpack("<Q", enc.recv_exact(s, 8))
        frames.append((hdr, bytes(enc.recv_exact(s, m))))
        if not hdr.get("ok") or "image" not in hdr:
            return frames


def ask_over_the_wire(enc, address, req):
    """(image records, their bytes, the closing header) of one request to a running worker."""
    with socket.create_connection(address, timeout=10) as s:
        body = json.dumps(req).encode()
        s.sendall(struct.pack("<I", len(body)) + body)
        frames = read_reply(enc, s)
    last = frames[-1][0]
    if "images" in last and isinstance(last["images"], list):  # the encoder variant
        return last["images"], frames[-1][1], last
    return [h["image"] for h, _ in frames[:-1]], b"".join(d for _, d in frames[:-1]), last


class RowLimitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.enc = load_encoder(max_rows=100)

    def setUp(self):
        self.enc.padded.clear()
        self.enc.planned.clear()

    def test_default_cap_matches_the_front_end(self):
        self.assertEqual(self.enc.p.parse_args([]).max_rows, 40000)  # dsfront --max-image-rows

    def test_rows_up_to_the_cap_pass(self):
        out, data = drain(self.enc, {"images": [image(30), image(30), image(40)]})
        self.assertEqual([i["rows"] for i in out], [30, 30, 40])
        self.assertEqual(len(data), 100 * self.enc.row_bytes)

    def test_rows_over_the_cap_stop_at_the_image_that_crosses_it(self):
        with self.assertRaises(self.enc.TooManyRows) as e:
            drain(self.enc, {"images": [image(30)] * 3 + [image(11)] + [image(30)] * 6})
        self.assertEqual(str(e.exception), "at most 100 image tokens per request; the first 4 of 10 images already take 101")
        self.assertEqual(len(self.enc.planned), 4, "the six images after it are not even read")
        self.assertEqual(self.enc.padded, [b"30"] * 3, "the image that crosses the cap is refused before it is padded")

    def test_41_full_size_images_cost_40_paddings(self):
        enc = load_encoder(max_rows=40000)
        with self.assertRaises(enc.TooManyRows) as e:
            drain(enc, {"images": [image(994)] * 41})
        self.assertEqual(str(e.exception), "at most 40000 image tokens per request; the first 41 of 41 images already take 40754")
        self.assertEqual(len(enc.padded), 40)
        enc.padded.clear()
        out, _ = drain(enc, {"images": [image(994)] * 40})
        self.assertEqual((len(out), len(enc.padded)), (40, 40))

    def test_a_request_can_lower_the_cap_but_not_raise_it(self):
        self.assertEqual(self.enc.row_limit({"max_rows": 50}), 50)
        with self.assertRaisesRegex(self.enc.TooManyRows, "at most 50 image tokens"):
            drain(self.enc, {"images": [image(30), image(30)], "max_rows": 50})
        self.assertEqual(self.enc.row_limit({"max_rows": 100000}), 100)
        for ignored in (None, True, "50", 0, -1, 2.5, [50]):
            self.assertEqual(self.enc.row_limit({"max_rows": ignored}), 100, ignored)

    def test_the_reply_says_too_many_rows(self):
        srv = self.enc.S(("127.0.0.1", 0), self.enc.H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            def ask(req):
                return ask_over_the_wire(self.enc, srv.server_address, req)

            recs, data, hdr = ask({"images": [image(60), image(60)], "max_rows": 40000})
            self.assertEqual((hdr["ok"], hdr["code"]), (False, "too_many_rows"))
            self.assertTrue(hdr["error"].startswith("at most 100 image tokens per request"))
            recs, data, hdr = ask({"images": [image(60), image(40)], "max_rows": 40000})
            self.assertTrue(hdr["ok"])
            self.assertEqual(len(data), 100 * self.enc.row_bytes)
            recs, data, hdr = ask({"images": []})
            self.assertEqual((hdr["ok"], hdr.get("code")), (False, None), "other errors carry no code")
        finally:
            srv.shutdown()
            srv.server_close()


class FakeSocket:
    """A socket that hands out `data` a few bytes at a time, through either read call, and keeps the buffers it
    was asked to fill."""

    def __init__(self, data, step=7):
        self.data, self.step, self.filled = data, step, []

    def recv_into(self, buf):
        self.filled.append(buf.obj if isinstance(buf, memoryview) else buf)
        k = min(len(buf), self.step, len(self.data))
        buf[:k] = self.data[:k]
        self.data = self.data[k:]
        return k

    def recv(self, n):
        k = min(n, self.step, len(self.data))
        out, self.data = self.data[:k], self.data[k:]
        return out


@unittest.skipUnless(MODULE == "media_worker", "the media worker's memory")
class HeldOnceTests(unittest.TestCase):
    """Each thing the worker holds, it holds once."""

    @classmethod
    def setUpClass(cls):
        cls.enc = load_encoder(max_rows=100)

    def serve(self):
        srv = self.enc.S(("127.0.0.1", 0), self.enc.H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (srv.shutdown(), srv.server_close()))
        return srv

    def test_recv_exact_returns_the_buffer_it_filled(self):
        sock = FakeSocket(b"0123456789abcdef")
        got = self.enc.recv_exact(sock, 16)
        self.assertTrue(sock.filled, "the bytes are read straight into the buffer")
        self.assertTrue(all(b is sock.filled[0] for b in sock.filled))
        self.assertIs(got, sock.filled[0], "no copy of the filled buffer is made")
        self.assertEqual(bytes(got), b"0123456789abcdef")
        with self.assertRaises(ConnectionError):
            self.enc.recv_exact(FakeSocket(b"abc"), 4)

    def test_images_are_answered_one_frame_each_then_an_end_frame(self):
        srv = self.serve()
        with socket.create_connection(srv.server_address, timeout=10) as s:
            body = json.dumps({"images": [image(30), image(40)]}).encode()
            s.sendall(struct.pack("<I", len(body)) + body)
            frames = read_reply(self.enc, s)
        self.assertEqual(len(frames), 3)
        for (hdr, data), rows in zip(frames, (30, 40)):
            self.assertEqual((hdr["ok"], hdr["image"]["rows"], len(data)), (True, rows, rows * self.enc.row_bytes))
            self.assertNotIn("images", hdr)
        self.assertEqual(frames[2], ({"ok": True, "images": 2}, b""))

    def test_an_error_frame_ends_the_reply_after_the_frames_already_sent(self):
        srv = self.serve()
        with socket.create_connection(srv.server_address, timeout=10) as s:
            body = json.dumps({"images": [image(60), image(60), image(10)]}).encode()
            s.sendall(struct.pack("<I", len(body)) + body)
            frames = read_reply(self.enc, s)
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0][0]["image"]["rows"], 60)
        hdr, data = frames[1]
        self.assertEqual((hdr["ok"], hdr["code"], data), (False, "too_many_rows", b""))

    def test_no_two_padded_images_are_held_at_once(self):
        import weakref

        live, most = [], []
        saved = self.enc.pad

        def pad(image, grid):
            most.append(sum(1 for r in live if r() is not None))
            arr = np.zeros((1, int(image), 3), dtype=np.uint8)
            live.append(weakref.ref(arr))
            return arr

        self.enc.pad = pad
        try:
            recs, data, last = ask_over_the_wire(self.enc, self.serve().server_address, {"images": [image(10)] * 6})
        finally:
            self.enc.pad = saved
        self.assertEqual((len(recs), last["images"]), (6, 6))
        self.assertEqual(max(most), 0, "an image is let go before the next is padded")

    def test_one_lock_is_taken_after_the_length_and_before_the_body(self):
        enc = self.enc
        srv = self.serve()
        inside, release, reads = threading.Event(), threading.Event(), []
        real_plan, real_recv = enc.plan, enc.recv_exact

        def plan(raw):
            inside.set()
            self.assertTrue(release.wait(20))
            return real_plan(raw)

        def recv_exact(s, n):
            if threading.current_thread() is not threading.main_thread():
                reads.append(n)
            return real_recv(s, n)

        def send(s, req):
            body = json.dumps(req).encode()
            s.sendall(struct.pack("<I", len(body)) + body)
            return len(body)

        enc.plan, enc.recv_exact = plan, recv_exact
        try:
            with socket.create_connection(srv.server_address, timeout=30) as first, \
                    socket.create_connection(srv.server_address, timeout=30) as second:
                send(first, {"images": [image(30)]})
                self.assertTrue(inside.wait(20))
                second_body = send(second, {"images": [image(11)] * 7})
                time.sleep(0.5)
                self.assertNotIn(second_body, reads, "a second connection's body is not read while the first request runs")
                self.assertIn(4, reads)
                release.set()
                self.assertEqual(read_reply(enc, first)[-1][0], {"ok": True, "images": 1})
                self.assertEqual(read_reply(enc, second)[-1][0], {"ok": True, "images": 7})
                self.assertIn(second_body, reads)
        finally:
            release.set()
            enc.plan, enc.recv_exact = real_plan, real_recv


@unittest.skipUnless(MODULE == "media_worker", "the media worker's memory")
class RequestBoundAndAllocatorTests(unittest.TestCase):
    """The worker bounds the request it will read, fixes glibc's mmap threshold, and has no switch for the
    public-address rule."""

    @classmethod
    def setUpClass(cls):
        cls.enc = load_encoder(max_rows=100)

    def serve(self):
        srv = self.enc.S(("127.0.0.1", 0), self.enc.H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (srv.shutdown(), srv.server_close()))
        return srv

    def test_the_default_bound_is_72_mib(self):
        self.assertEqual(self.enc.p.parse_args([]).max_request_bytes, 72 << 20)

    def test_a_larger_frame_is_thrown_away_and_answered_with_an_error_frame(self):
        enc = self.enc
        saved, enc.a.max_request_bytes = enc.a.max_request_bytes, 4000
        srv = self.serve()
        try:
            with socket.create_connection(srv.server_address, timeout=10) as s:
                big = json.dumps({"images": [image(1)], "pad": "x" * 4000}).encode()
                self.assertGreater(len(big), 4000)
                held = enc._lock.acquire(timeout=5)  # the bound is checked without the lock
                try:
                    s.sendall(struct.pack("<I", len(big)) + big)
                    frames = read_reply(enc, s)
                finally:
                    if held:
                        enc._lock.release()
                hdr, data = frames[0]
                self.assertEqual((hdr["ok"], hdr["code"], data), (False, "request_too_large", b""))
                self.assertEqual(hdr["error"], "the request is %d bytes; the media worker takes at most 4000" % len(big))
                body = json.dumps({"images": [image(5)]}).encode()  # the same connection goes on
                s.sendall(struct.pack("<I", len(body)) + body)
                self.assertEqual(read_reply(enc, s)[-1][0], {"ok": True, "images": 1})
                exact = json.dumps({"images": [image(5)]}).encode()
                exact += b" " * (4000 - len(exact))  # a frame of exactly the bound is read
                s.sendall(struct.pack("<I", len(exact)) + exact)
                self.assertEqual(read_reply(enc, s)[-1][0], {"ok": True, "images": 1})
        finally:
            enc.a.max_request_bytes = saved

    def test_the_old_256_mib_check_is_gone(self):
        enc = self.enc
        saved, enc.a.max_request_bytes = enc.a.max_request_bytes, 1000
        srv = self.serve()
        try:
            with socket.create_connection(srv.server_address, timeout=10) as s:
                s.sendall(struct.pack("<I", 1001) + b"x" * 1001)
                self.assertEqual(read_reply(enc, s)[0][0]["code"], "request_too_large")
        finally:
            enc.a.max_request_bytes = saved

    def test_the_threshold_is_set_when_the_worker_is_imported(self):
        code = ("import sys; sys.argv=['media_worker.py','--model',%r]; sys.path.insert(0,%r); import media_worker as m, ctypes; "
                "l=ctypes.CDLL(None); l.mallopt.argtypes=[ctypes.c_int,ctypes.c_int]; "
                "print(m.MMAP_THRESHOLD, l.mallopt(-3, 1<<20))") % (self.model_dir(), HERE)
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.stdout.split(), [str(1 << 20), "1"], r.stderr[-300:])

    def model_dir(self):
        d = tempfile.mkdtemp(prefix="vision-test-model-")
        os.makedirs(os.path.join(d, "inference"))
        with open(os.path.join(d, "inference", "config.json"), "w") as f:
            json.dump({"vision_n_layers": 1}, f)
        return d

    def test_freed_blocks_go_back_whichever_thread_freed_them(self):
        code = ("import sys, threading; sys.argv=['media_worker.py','--model',%r]; sys.path.insert(0,%r); import media_worker\n"
                "def rss():\n    return int(open('/proc/self/statm').read().split()[1]) * 4096 >> 20\n"
                "def first():\n    b = bytearray(27 << 20); b[::4096] = b'x' * len(b[::4096]); del b\n"
                "def second():\n    xs = [bytearray(16 << 20) for _ in range(8)]\n"
                "    for x in xs: x[::4096] = b'x' * len(x[::4096])\n    del xs\n"
                "t = threading.Thread(target=first); t.start(); t.join(); base = rss()\n"
                "t = threading.Thread(target=second); t.start(); t.join()\n"
                "print(rss() - base)") % (self.model_dir(), HERE)
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
        self.assertLess(int(r.stdout.strip() or 999), 16, r.stderr[-300:])

    def test_a_refused_mallopt_stops_the_start(self):
        import ctypes

        class Libc:
            def __init__(self):
                self.mallopt = lambda *args: 0

        real = ctypes.CDLL
        ctypes.CDLL = lambda *args, **kw: Libc()
        try:
            with self.assertRaises(SystemExit) as e:
                self.enc.fix_mmap_threshold()
        finally:
            ctypes.CDLL = real
        self.assertIn("mallopt", str(e.exception))

    def test_the_public_address_rule_has_no_switch(self):
        saved = os.environ.get("VISION_ALLOW_PRIVATE")
        os.environ["VISION_ALLOW_PRIVATE"] = "1"
        try:
            with self.assertRaisesRegex(ValueError, "not a public address"):
                self.enc._public("127.0.0.1")
        finally:
            if saved is None:
                del os.environ["VISION_ALLOW_PRIVATE"]
            else:
                os.environ["VISION_ALLOW_PRIVATE"] = saved


if __name__ == "__main__":
    unittest.main()
