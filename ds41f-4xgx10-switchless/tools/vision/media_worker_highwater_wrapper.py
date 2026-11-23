"""Starts the media worker with only its public-address rule replaced (a worker that fetches from 127.0.0.1),
for media_highwater.sh's URL-clip rows. The worker's own code has no switch for it: this file imports the
worker as the image has it (from /app), replaces `_public`, and serves exactly as the worker's main does.

    python3 media_worker_highwater_wrapper.py --host 127.0.0.1 --port 18951 --max-image-pixels 40000000

The arguments are the worker's own. It is mounted into the media container, never built into the image.
"""
import sys

sys.path.insert(0, "/app")
import media_worker as mw  # noqa: E402  (parses sys.argv as the worker does)

mw._public = lambda host, kind="image": None

if __name__ == "__main__":
    srv = mw.S((mw.a.host, mw.a.port), mw.H)
    mw.log(f"listening on {mw.a.host}:{mw.a.port} (mmap threshold {getattr(mw, 'MMAP_THRESHOLD', 0) >> 20} MiB, public-address rule replaced)")
    srv.serve_forever()
