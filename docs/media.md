# Images and clips: the media worker and the engine's encoder

A chat request can carry images (`image_url` and `image` parts) and clips (`video_url`). This build reads them in a small
**media worker** that has no model and no GPU, sends their pixels to the engine in the request, and encodes them inside the
engine with the model's vision tower split over the four Sparks. [vision.md](vision.md) describes the whole path, the limits and the
measurements, and the alternative path (`--vision container`), where a separate encoder container (PyTorch) answers the rows. The user's view is in the [README](../README.md#images-and-clips).

```
client --image_url / base64 / video_url--> dsfront --> media worker (127.0.0.1)
                                                        <-- one frame per image: rows, hash, grid and the padded pixels --
dsfront --one request frame: prompt ids, spans, grids, pixels (flag 32)--> rank 0 --> ranks 1-3
engine: encode (tower split over the ring) --> the image's rows written into the request's store chunks
prefill: rows injected at their span; span rows route with the image bias; Engram is dead on spans; decode is untouched
```

## The media worker

`tools/vision/media_worker.py`: Python with Pillow, numpy and PyAV, no PyTorch, no GPU. It reads untrusted bytes, so image
parsers and the video demuxer stay out of the engine's process. It reads an image (`Image.open(...).convert("RGB")`), plans its
grid and pads it as the checkpoint's own `image_processor.py` does (the grid functions are ported from it), and answers the
image's rows, hash and grid with the padded image as bytes. The engine turns the bytes into the tower's patches through its
256-entry table, and `test_media_worker.py` holds those patches bit-equal to the reference's.

The container (`tools/vision/Dockerfile.media-worker`) is `python:3.12-slim` pinned by digest with Pillow, numpy and PyAV
pinned by version and hash, no runtime dependencies, running as an unprivileged user; its default command is
`--host 127.0.0.1 --port 18951 --max-images 100`. The script itself listens on port 18950 when given no `--port`, and the
container encoder ([vision.md](vision.md)) also uses 18950: run one at a time. The front-end's `--media-addr` example and the
container's default command use 18951. Its memory cap is the engine's `--media-mib` (512).

Build and run it as [setup.md](setup.md#8-the-media-worker) describes: `docker build -f Dockerfile.media-worker -t pico-media:<tag> .` in
`tools/vision`, then the `pico-media` unit (`deploy/pico-media.service`), which runs it with host networking, no GPU, a read-only root,
512 MiB with no swap, and the checkpoint's `inference/config.json` as its only mount. `service.sh` does not start the media worker.

Protocol (TCP on 127.0.0.1): a request is `u32 len`, then JSON `{"images": [record, ...], "max_rows": N}` or `{"videos":
[record, ...], "frame_budget": N}`; a record is `{"url": data-URL or http(s) URL}` or `{"data": base64}`. The response is a run
of frames, each `u32 len, JSON, u64 len, bytes`: for images one frame per image in order (`{"ok": true, "image": {rows, hash,
grid, ms, bytes}}` and the padded `u8 [n_vit_h x 14][n_vit_w x 14][3]`), then an end frame `{"ok": true, "images": n}`; an
error frame (`{"ok": false, "error", "code"}`) ends the response at any image and the images before it are dropped, so neither
side holds a request's images twice. The worker holds each thing once: a request's body is read into one buffer, a clip goes
to a file and the frame process opens it, an image is padded and sent before the next is read, and one lock covers a request
from its length to the end of its response.

| Option | Default | |
|---|---|---|
| `--host`, `--port` | 127.0.0.1, 18950 | where it listens |
| `--model DIR` | /model | only `inference/config.json` is read |
| `--max-images N` | 100 | images (and frames) a request |
| `--max-rows N` | 40,000 | rows (image tokens) of all a request's images; `dsfront` sends its own cap and the lower applies |
| `--max-bytes N` | 32 MiB | one fetched image |
| `--max-request-bytes N` | 72 MiB | one request frame; a larger one is read and thrown away in pieces and answered with an error |
| `--max-image-pixels N` | 40,000,000 | width x height, read from the header before anything is decoded |
| `--max-video-bytes N` | 128 MiB | one clip |
| `--video-timeout-s S` | 60 | the frame process's time |

Public http(s) addresses only (no fetch into the local network); nested media protocols are disabled. Clips are sampled to
up to 32 frames (`max_frames` 1 to 100 in the part), uniformly from the start through the end, each frame at most 768x768
with its timestamp and the clip's duration, in a separate process. Frames and stills share the limit of 100 per request and the
row cap. Audio is ignored.

## The front-end

`dsfront serve --media-addr HOST:PORT` sends each request's pixels in its frame (`--vision engine` is the default). Without
`--media-addr` and without `--vision`, image requests are refused; `--vision engine` given without `--media-addr` is an error. A request
that cannot run (an image over the pixel limit, a part over 400 MiB, rows over the cap) is answered 400, 413 or 429 and never
reaches the engine. `--vision container --vision-addr HOST:PORT` is the alternative path. The encoder's kind is part of every image's
hash, so a session made under one encoder is never continued under the other.

| Option | Default | |
|---|---|---|
| `--max-image-rows N` | 40,000 | rows of all one request's images |
| `--max-image-rows-in-flight N` | 80,000 | rows across requests in flight; a request that does not fit is answered 429 with `Retry-After` |
| `--front-image-mib N` | 400 | the pixel bytes the image requests in flight count for against the session table's budget (`--front-table-mib`); one request over it is refused with 413 |
| `--max-body-mb N` | 64 | the JSON body |

## The request frame (flag 32, pixels)

After the prompt ids: `u32 n` (1 to 100); `n` spans (`start`, `rows`, `u64 hash`); `n` grids `u16 n_h, u16 n_w` (patches; the
array padded to 8 bytes); then each image's `588 x n_h x n_w` bytes of u8, each padded to 8 bytes. A span's rows are
`ceil(n_h / 3) x (ceil(n_w / 3) + 1) + 2`, at most 1,024; a grid is at most 9,216 patches. A frame is either pixels or rows
(`flag 8`, the container encoder's), never both; any other byte count is malformed and closes the connection. A well-formed frame the engine
will not run is answered with an error and the connection lives (a grid of 0 patches, more than 9,216 patches, rows that do not
match their grid). The layout and its checks are in `engine/server/pixels.h`.

## The engine's encoder

- **Weights.** `--vision-weights FILE` names each rank's own quarter of the tower (a quarter of every large matrix, 232.80 MiB a
  rank), cut by `dsw_convert --vision`; the ranks check that the four files are one set. They load before the store is sized
  and are counted among the lines outside the store.
- **The tower.** 32 blocks run split four ways over the ring, 131 in-place gathers an image: a bf16 tensor-core GEMM with fp32
  accumulation and one rounding, dense attention with an fp32 online softmax, 2-D rotary, rmsnorm, swiglu and gelu
  (`engine/vision`). Each rank writes the image's span rows straight into the request's chunks, so no row crosses the fabric.
  Rank 0 copies the image's pixels through a pinned staging buffer (6 MiB) and a gather brings the image to every rank.
- **The work area** is a tenant of the store ([store.md](store.md)): device memory, two groups (356 MiB), taken once at start and
  held while the engine runs. The start prints "vision work area: ... the engine encodes images" or says the engine has no image
  encoder until the next start. A take that fails on any rank fails on all four.
- **One image at a time.** From an image's first block to its last the work area is that image's. No other image starts, and an
  image of another request that is between blocks makes a pixel request wait for its last block, never for another request's
  whole Prompt Processing pass; a text request is never held.
- **Paced beside the decoders.** An encode unit is the longest run of whole images whose time fits the interleave gap
  (`--interleave-gap-ms`, 600) less one decode step while sessions decode, at least one image; an image longer than that goes in
  ranges of blocks. With nobody decoding every image is encoded back to back. The image rows' processing is paced the same way
  ([interleave.md](interleave.md)).
- **Which images, and when.** An image is encoded before the first row piece that reaches its span, in position order. Images
  whose spans end before the first position being processed are not encoded, and a resume whose restart moved back encodes what it now
  needs. A request's rows stay in the store until it ends.
- **The vote.** At an image's last block each rank sums its span's bytes and the four sums are compared in one host all-gather; a
  difference ends the request with an error and frees its rows.

## Measured

On four Dell Pro Max GB10, through the front-end's API, eight users decoding and one image request among them, medians of five
runs; "32 frames" is 32 frames of 768 x 768 sent in one request (12,224 rows). The builds and the ranges are in
[vision.md](vision.md#measured).

| | the encoder container, the rows processed in one run | the engine's encoder, the rows processed in pieces |
|---|---|---|
| one 768 x 768 image: time to first token | 0.737 s | 0.443 s |
| one image: the users' longest pause | 0.33 s | 0.51 s (inside the 600 ms gap) |
| 32 frames: time to first token | 12.56 s | 10.20 s |
| 32 frames: the users' longest pause | 2.01 s | 0.58 s |

With the container still encoding, processing the image rows in pieces (above) brought the users' longest pause during the 32 frames
from 2.01 s to 0.59 s and made the frames' first token later, 14.98 s against 12.56 s; encoding in the engine took that back
and more.

Timed by the engine itself, from a request's arrival at rank 0's socket (no front-end, no media worker), on the build that brought
the encoder in: no encode unit held the decoding users longer than the 600 ms gap (557 to 567 ms by the engine's clock, five
runs); the decoding users' client saw them silent for at most 672 to 696 ms in the same runs (the end of a decode run adds about
0.1 s). Every image the four
ranks encoded was byte-equal to the unsplit run (ranges, two flights, cancels, a resume past the image). On one Spark with the
engine stopped, a rank's share of the 32-block tower took 12.8 ms for a small image to 141.8 ms for a large one (1,521 to 9,189
patches, `bench_vit`).

On an earlier build that encoded with the container, the second turn of an image conversation took nearly all of its prompt from
cache and started in 0.25 s (the first turn, cold: 1.40 s; through the front-end).

## Tests

`engine/vision/tests` (`vit_test`: split against unsplit in bytes on four grids, with ten deliberately wrong builds that must
differ; `vit_ring_test`, `vit_gather_test`, `bench_vit`), `engine/server/tests` (`vit_engine_test.py`: bytes, ranges, two
flights, cancel, vote, resume; `vit_sched_test.cpp`, `pixels_test.cpp`, `pixel_client.py`, `image_store_test.py`),
`tools/vision` (`test_media_worker.py`, `test_video_frames.py`, `test_video_errors.py`, `test_encoder_limits.py`; against a
running endpoint `smoke_100.py`, `smoke_video.py`, `imgtest.py`).
