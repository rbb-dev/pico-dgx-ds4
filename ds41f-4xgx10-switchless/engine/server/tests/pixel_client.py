#!/usr/bin/env python3
"""A SUBMIT that carries its images as pixels (kFlagPixels), on the tests' one Connection (serve_test.Engine), then a text request on the
same connection. Until the encoder is in the engine every well-formed pixel request is answered DONE kError "this engine has no image
encoder" and the connection lives; a refused frame (a grid that does not match its span, rows over 1,024) is answered with its reason;
a malformed frame (both flags, a wrong byte count) closes the connection.

  python3 engine/server/tests/pixel_client.py --socket S --tokens T.npy --image noise_544.u8:39x39 [--image F:HxW ...]
        [--version 5] [--prefix] [--bad rows|bytes|both]

An image file is the media worker's padded u8 image, exactly 588 * H * W bytes (H x W patches), as the reference run's tests/images make them
(tools/vision/make_test_images.py: noise_544 39 x 39, noise_768 55 x 55, checker_770x1022 73 x 55, strip_42x42882 3 x 3,063,
tall_42x42882 1,533 x 3, each planned by the reference's image_processor.plan_image_grid; write the planned image's bytes to FILE.u8).

Settings: an engine of any build with this frame reader, started as production or with --keep-gib 0; tokens file T.npy of at least
2,000 ids; no encoder needed. The run prints each answer. Pass line: every pixel request answered with the reason above (kind 'error') or
the refusal text; the text request after it runs ('stop' or 'length', tokens present) on the same socket; with --bad bytes or both the
engine closes the socket (EOFError on the next read) and a new connection serves the text request.
"""
import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_test import Engine, SUBMIT, F_KEEP, F_PREFIX, load_npy_i32  # noqa: E402

F_IMAGES, F_PIXELS = 8, 32
IMAGE_ID = 129264


def rows_of(h, w):
    return -(-h // 3) * (-(-w // 3) + 1) + 2


def pad8(n):
    return (n + 7) & ~7


def pixel_submit(e, session, text_ids, images, prefix=None, bad=None):
    """images: [(h, w, bytes)]. The prompt is the text ids with each image's span of IMAGE_ID ids after them."""
    rid = e.next_id
    e.next_id += 1
    flags = F_KEEP | F_PIXELS | (F_IMAGES if bad == 'both' else 0) | (F_PREFIX if prefix else 0)
    prompt = list(text_ids)
    spans = b''
    for h, w, _ in images:
        rows = rows_of(h, w) + (1 if bad == 'rows' else 0)
        spans += struct.pack('<IIQ', len(prompt), rows, 0x1234)
        prompt += [IMAGE_ID] * rows
    hdr = struct.pack('<QQQQffIIIIIIII', rid, session, 0, 0, 0.0, 1.0, 0, 8, flags, 0, 0, 1, len(prompt), 0)
    px = struct.pack('<QII', prefix[0], prefix[1], 0) if prefix else b''
    grids = b''.join(struct.pack('<HH', h, w) for h, w, _ in images)
    grids += b'\0' * (pad8(len(grids)) - len(grids))
    data = b''
    for h, w, b in images:
        assert len(b) == 588 * h * w, (len(b), h, w)
        data += b + b'\0' * (pad8(len(b)) - len(b))
    if bad == 'bytes':
        data = data[:-1]
    e.send(SUBMIT, hdr + px + struct.pack('<I', 1) + struct.pack('<%dI' % len(prompt), *prompt) + struct.pack('<I', len(images)) + spans + grids + data)
    return rid, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--image', action='append', required=True, help='FILE:HxW')
    ap.add_argument('--version', type=int, default=5)
    ap.add_argument('--prefix', action='store_true', help='send a SubmitPrefix block (epoch 0, 0 ids: a refused build is not reached)')
    ap.add_argument('--bad', choices=['rows', 'bytes', 'both'])
    a = ap.parse_args()
    T = load_npy_i32(a.tokens)
    images = []
    for spec in a.image:
        f, hw = spec.rsplit(':', 1)
        h, w = (int(x) for x in hw.split('x'))
        images.append((h, w, open(f, 'rb').read()))
    e = Engine(a.socket, a.version)
    try:
        r = e.collect(pixel_submit(e, 1, T[:200], images, prefix=(0, 0) if a.prefix else None, bad=a.bad))
        print('pixel request: reason %s, error %r' % (r['reason'], r['error']))
    except EOFError:
        print('pixel request: the engine closed the connection (a malformed frame)')
        e = Engine(a.socket, a.version)
    t = e.run(2, T[:500], max_new=16)
    print('text request: reason %s, %d tokens' % (t['reason'], len(t['ids'])))
    ok = t['reason'] in ('stop', 'length') and t['ids']
    e.release(2)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
