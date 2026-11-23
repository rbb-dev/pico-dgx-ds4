# Images and video

A chat request can carry images (`image_url` and `image` parts) and video clips (`video_url` parts). In this build the model's
own vision encoder runs **inside the engine**: its weights are cut four ways and held in the device memory of all four Sparks,
like the rest of the model, and the four GPUs encode every image together over the ring. A small **media worker**, with no model
and no GPU, opens the files and hands the front-end their pixels. This page explains what pico-dgx-ds4 does with an image, what
this build changed and why, where each part runs, the path of one image and of one clip from the client to the first token, the
options, what was measured and how it is tested.

[media.md](media.md) describes the media worker in depth, the README's [Images and clips](../README.md#images-and-clips) is the
user's view, and [setup.md](setup.md) (section 5.2, and [section 8](setup.md#8-the-media-worker)) sets it up. pico-dgx-ds4's own account
of its side encoder is [ds41f-4xgx10-switchless/docs/vision.md](../ds41f-4xgx10-switchless/docs/vision.md). Paths below are relative to
`ds41f-4xgx10-switchless/` unless they start with `deploy/` or `docs/`.

Contents: [The model's vision part](#the-models-vision-part), [pico-dgx-ds4's encoder](#pico-dgx-ds4-an-encoder-beside-the-engine),
[What this build changed, and why](#what-this-build-changed-and-why), [Where each part runs](#where-each-part-runs),
[The path of one image](#the-path-of-one-image), [The path of a clip](#the-path-of-a-clip),
[The engine's encoder](#the-engines-encoder), [Memory: what left rank 0](#memory-what-left-rank-0), [Options](#options),
[Limits and refusals](#limits-and-refusals), [The pixels frame](#the-pixels-frame), [Measured](#measured), [Tests](#tests),
[The container encoder](#the-container-encoder).

```
client --image_url / base64 / video_url--> dsfront (rank 0's Spark)
dsfront --the image and clip records--> media worker (rank 0's Spark, 127.0.0.1:18951, a container: no model, no GPU)
        <--per image: the padded pixels, its rows, hash and grid--
dsfront --one request frame: prompt ids, spans, grids, pixels (flag 32)--> engine rank 0
engine, all four ranks: the tower split four ways, gathers over the ring --> the image's rows in each rank's store chunks
prefill: the rows at their span, the image routing bias, no Engram on span rows --> the first token; decoding is untouched
```

## The model's vision part

DeepSeek-V4.1-Flash-1017 ships, beside the text model:

- a **vision tower** (`vision.*`): a patch embedding of 14 x 14-pixel patches, 32 transformer blocks of width 1,024 (16 attention
  heads of 64, a SwiGLU MLP 2,816 wide), 2-D rotary positions and a final norm;
- an **aligner** (`aligner.*`): it joins each 3 x 3 group of neighbouring patches into one row of the language model's width,
  5,120, through two matrices with a GELU between them;
- three learned rows, `image_start`, `image_newline` and `image_end`;
- a routing bias for image rows in every MoE layer (`ffn.gate.bias_vl`).

The encoder is 266 tensors and about 485 million parameters, 0.97 GB in bf16 (counted from the shapes in
`engine/vision/vit_weights.h`). An image padded to `n_h x n_w` patches becomes a **span** of
`ceil(n_h / 3) x (ceil(n_w / 3) + 1) + 2` rows in the prompt: the start row, each row of aligner tokens followed by a newline
row, the end row. A 768 x 768 image is 55 x 55 patches and 382 rows; an image is at most 1,024 rows and 9,216 patches. Every
position of a span carries the same token id (the image placeholder); the prefill puts the span's rows in place of that token's
embedding.

## pico-dgx-ds4: an encoder beside the engine

In pico-dgx-ds4 (by Pico), images reach the model through a side encoder; this build keeps the way the engine puts the rows
into the prompt:

- **The encoder** is a separate PyTorch process in a container on rank 0's Spark (`tools/vision/vision_encoder.py`). It imports
  the checkpoint's own `inference/vision.py` and `inference/image_processor.py`, so the preprocessing (at least 544 x 544 pixels,
  grey padding, patch 14, at most 1,024 rows an image) and the span layout are the publisher's. It runs the whole tower on rank
  0's GPU and answers every row of each image's span, bf16 x 5,120: 10 KiB a row.
- **The front-end** sends each request's images to it, writes the placeholder once per row of each span, and sends the rows with
  the request (flag 8).
- **The engine**: rank 0 broadcasts the spans and the rows to ranks 1 to 3; the prefill replaces the span positions' embeddings
  with the rows, runs the MoE router again with `bias_vl` on span rows only, and zeroes Engram's contribution on them. Decoding
  has no image branch: the same code, kernels and graphs as text.

pico-dgx-ds4 accepts up to 100 images a request and video input through timestamped frames (from pull requests by MiaAI-Lab)
and caps a request's image rows (`--max-image-rows`). By pico-dgx-ds4's own measurement the encoder holds 0.95 GiB of GPU memory
and about 0.4 GiB of host memory on rank 0's Spark.

## What this build changed, and why

**Why.** On this cluster the side encoder cost four things:

1. **Rank 0 carried it alone, and every Spark paid.** The tower's weights, PyTorch and its working memory lived on rank 0's Spark.
   The store is the same size on every Spark and is set by the Spark with the least memory to offer, which is rank 0 (it also holds
   the front-end), so what rank 0 alone held came off the store of all four. The engine set 1,260 MiB aside for the container on
   rank 0 (`--encoder-mib`), and PyTorch's cache grew with image traffic and kept what it took: measured here, after a series of
   image tests the container held 974 MiB, and rank 0 had 1.2 GiB less free memory at the next start until the container was
   restarted, so that start's store was smaller on every Spark.
2. **Rows are heavy.** A row is 10 KiB, and it travelled from the encoder to the front-end, to rank 0 and over the fabric to ranks 1
   to 3. 32 frames of 768 x 768 are 12,224 rows: the container's answer for them was 119 MiB (measured), where their pixels are
   54 MiB (computed).
3. **One GPU encoded while three waited, and the other users stopped.** The tower ran on rank 0's GPU alone, and an image
   request's rows were read in one run: measured here with eight users decoding, a request of 32 video frames held up their
   output for 1.9 to 2.2 s, where their longest pause without it was 0.31 s.
4. **PyTorch in the serving path.** The encoder's container image as run here (NVIDIA's PyTorch base with PyAV) is 24.4 GB on
   disk, and the process that parsed untrusted image files had the GPU.

**What changed.**

1. **The tower's weights cut four ways.** `dsw_convert --vision` writes each rank's quarter: every large matrix is cut by its
   output rows. Rank r holds heads 4r to 4r+3 of q, k and v, and a quarter of the outputs of `wo`, of the MLP's gate and up halves,
   of its down matrix and of both aligner matrices; the patch embedding, the final norm and the three learned rows are whole on
   every rank. `--vision-weights` loads a rank's file into its device memory at the start (232.80 MiB on the device), before the
   store is sized (`engine/weights/plan.cpp`, `engine/vision/vit_file.h`).
2. **The encoder runs in the engine, on all four GPUs.** The engine has its own kernels for it (`engine/vision`): a bf16
   tensor-core matrix product with fp32 accumulation and one rounding, dense attention with an fp32 online softmax, 2-D rotary,
   RMSNorm, SwiGLU and GELU. Because every matrix is cut by its outputs, every output value is computed whole on one rank: the
   ranks exchange finished bf16 blocks (four all-gathers a block, 131 for a whole image) and never partial sums, so the four-rank
   run gives the same bytes as the same kernels run unsplit on one GPU (`engine/vision/vit_rank.h`, `engine/model/vit_cmd.cu`).
3. **Pixels in the request.** The front-end sends each image's padded pixels in the request (flag 32): 588 bytes a patch, 5,292
   bytes for each 10,240-byte row they replace. The engine turns them into the tower's input on the GPU
   (`engine/server/pixels.h`).
4. **The media worker.** A small Python process (Pillow, numpy and PyAV; no PyTorch, no model, no GPU; a 331 MB container image)
   reads the images and decodes the clips, so image parsers and the video demuxer stay out of the engine's process
   (`tools/vision/media_worker.py`, [media.md](media.md)).
5. **Image rows in the store.** A request's rows live in the store's chunks on every rank, 204 rows a chunk: reserved when the
   request is admitted, taken at its first pass, read by every later pass (a resume sends and encodes nothing again), given back
   however it ends. Rank 0 keeps no copy, and since each rank writes the rows itself, no row crosses the fabric after the encode
   (`engine/model/img_rows.h`, `engine/server/img_hold.h`).
6. **Encoding paced beside the decoding users.** Images are encoded in units that fit the interleave gap, and an image request's
   rows are processed in pieces, so the users decoding beside it keep getting tokens (`engine/server/vit_sched.h`,
   `engine/server/interleave.h`, [interleave.md](interleave.md)).
7. **The container out of the serving path.** With `--vision-weights` loaded and `--vision engine` (the front-end's default), no
   encoder container runs, and rank 0's line for an image process outside the store is the media worker's 512 MiB cap
   (`--media-mib`) instead of the container's 1,260 MiB. pico-dgx-ds4's container encoder stays available as an alternative
   ([The container encoder](#the-container-encoder)).

## Where each part runs

| Part | Where | What it does and holds |
|---|---|---|
| front-end (`dsfront`) | rank 0's Spark, CPU | the API, the chat encoding and the image limits; the session table, with each image's place, rows and hash; a request's pixels while it is in flight, counted against `--front-image-mib` |
| media worker (`pico-media`) | rank 0's Spark, a container on 127.0.0.1, CPU only | fetches, decodes and pads images, samples clips into frames; one request at a time; capped at 512 MiB, 12 to 36 MiB idle (measured) |
| engine, rank 0 | rank 0's Spark | holds a request's pixels in its frame until the request ends; decides which image is encoded when; copies the pixels to the GPU through a 6 MiB pinned buffer |
| engine, every rank | all four Sparks, GPU | its quarter of the tower (232.80 MiB of device memory); the work area (two groups of the store, 356 MiB); its share of every encode; the request's image rows in its own store chunks; the prefill that reads them |

## The path of one image

A client sends a chat request with one 768 x 768 image (an `image_url` part with a data URL or a public http(s) address):

1. **The front-end takes the request** (`frontend/src/server.rs`, `start_images`). It checks the body (64 MiB), counts the images
   (at most 100 a request, earlier turns and clips' frames included), and reserves the most rows they can take against
   `--max-image-rows-in-flight`; a request that does not fit is answered 429 with `Retry-After` and is not queued. The chat
   encoding writes one placeholder where the image goes.
2. **The media worker makes the pixels.** The front-end sends the image's record to the worker with the request's row cap. The
   worker fetches it (public addresses only, 32 MiB at most), hashes the file (the first 8 bytes of its SHA-256), reads the size
   from the header (40 megapixels at most), decodes it to RGB, plans the grid as the checkpoint's `image_processor.py` does, pads
   it with grey and answers the rows, the hash, the grid and the padded image as bytes: 55 x 55 patches, 382 rows, 1.78 MB.
3. **The front-end builds the prompt and picks the session.** It writes the placeholder 382 times, puts the encoder's kind into the
   image's hash, and asks its session table for a conversation to continue. A session is a candidate only when every image inside
   the text it shares with the prompt has the same place, rows and hash on both sides (`frontend/src/session.rs`, `images_agree`):
   the engine matches token ids only, and every image position has the same id. A changed image rules the session out.
4. **One frame to the engine.** The request goes to rank 0 as one frame: the prompt ids, each image's span (start, rows, hash), its
   grid and its pixels ([The pixels frame](#the-pixels-frame)).
5. **Rank 0 checks and admits it.** Before it keeps anything, rank 0 checks the frame: the rows match the grid, at most 1,024 rows
   and 9,216 patches an image, the byte count exact. It keeps the pixels in the frame's buffer until the request ends and admits
   the request counting the store chunks its rows need (two for 382 rows) beside what its conversation needs.
6. **Every rank takes the rows' chunks** from the store at the request's first pass.
7. **The four ranks encode the image**, before the first piece of the prompt that reaches its span. Rank 0 copies the pixels
   through its pinned buffer into its block of the work area's pixel slot, and one all-gather gives every rank the image. Each
   rank turns the bytes into the tower's input on its GPU (a 256-entry table, `(v / 255 - 0.5) / 0.5` in bf16), embeds the
   patches, runs its quarter of the 32 blocks with four all-gathers a block, then its quarter of the aligner; the last gather
   leaves the whole span on every rank, and each rank writes it into its own chunks. The four ranks compare a sum of the span's
   bytes; a difference ends the request with an error and frees its rows. In the engine a 768 x 768 image took 77.8 ms
   (measured).
8. **The prompt is processed.** At the span's positions the rows replace the placeholder's embedding, the MoE router adds
   `bias_vl` on those rows, and Engram adds nothing on them. With users decoding, the encode and the prompt's pieces take turns with
   their steps ([The engine's encoder](#the-engines-encoder)).
9. **The first token** is sampled at the end of the prompt, as for text, and decoding runs as for text. When the request ends,
   however it ends, every rank gives its rows' chunks back.

**The next turn.** A chat client sends the whole conversation again, the image included. The front-end continues the kept
session, the engine starts after the image's span, and the image is not encoded again: measured, a later turn whose starting point
lay past its image encoded nothing. Its rows' chunks are still reserved for that turn (two for this image), so that if its session
is evicted and the prompt has to be processed again from the start, the image is encoded then, once, without asking a full store for
room.

## The path of a clip

1. A `video_url` part carries a public http(s) address or a base64 data URL, and optionally `max_frames` (1 to 100, 32 by default).
2. The front-end sends the clip's record to the media worker with the frame budget: 100 less the request's other images,
   shared between its clips (`frontend/src/video.rs`).
3. The worker writes the clip to a file on its 192 MiB in-memory disk (128 MiB at most a clip), and a child process with a 60 s
   limit opens it with PyAV, seeks to up to 32 evenly spaced times from the start through the end, decodes the frame at each,
   shrinks it to fit 768 x 768 and stores it as a JPEG. It answers the clip's duration and each frame with its time
   (`tools/vision/video_frames.py`). Audio is ignored.
4. The front-end puts text and images in the clip's place: `[Video 1: duration D seconds]`, then for each frame
   `[Video 1, frame j at T seconds]` and the frame as an image part, then `[End of video 1]`.
5. From here the frames are images. They go to the media worker again as images (pixels, hash, grid), count against the 100
   images and the row cap, and follow the image path above: 32 frames of 768 x 768 are 12,224 rows and 54 MiB of pixels.
6. In the engine the frames are encoded in prompt order, each before the first piece of the prompt that reaches it. Beside
   decoding users the encode goes in units of whole frames that fit the gap, and the rows in pieces, with the users' steps between
   them; with nobody decoding it runs back to back.

## The engine's encoder

- **Weights.** `--vision-weights FILE` names each rank's quarter of the tower (`{rank}` in the path is the rank), cut by
  `dsw_convert --vision` and checked against the checkpoint byte for byte by `dsw_verify --vision`. At the start each rank checks
  that its file holds the 266 tensors of its quarter, and the four ranks compare a digest of their files before any byte is read:
  a file of another rank, another cut or another checkpoint ends the start on every rank. The weights load before the store is
  sized, so the memory left to the host accounts for them; each rank prints the tensors, the bytes on the device (232.80 MiB) and
  what the load cost. Without the option the engine runs as a text engine and answers a pixel request "this engine has no image
  encoder".
- **The work area** is a tenant of the store ([store.md](store.md)): device memory, two groups (356 MiB) on every rank, holding the
  largest image's buffers, the four-block pixel slot and the scratch of the sums. The start hands it over once, before any request
  and before kept conversations are read back, and it is held while the engine runs: it is never given back and never evicted.
  It is taken whole or not at all, and a take that fails on any rank fails on all four: the engine then serves text and answers a
  pixel request "the image encoder has no work area". Each rank's start line says which: "vision work area: ... the engine encodes
  images" or "the engine has no image encoder until the next start".
- **The tower on the ring.** One encode command runs a range of blocks of one image on every rank (`engine/model/vit_cmd.cu`):
  rank 0's pixels through its 6 MiB pinned buffer into the pixel slot, the pixel gather, then each rank's part of every phase and
  one in-place all-gather after it (the ring's `allgather_multi`, [ring.md](ring.md)): 1 for the pixels, 4 a block and 2 for the
  aligner, 131 for a whole image. At the last block each rank writes the span into its store chunks in runs that do not cross a
  chunk's edge.
- **One image at a time.** From an image's first block to its last, the work area is that image's: no image of any other request
  starts. A request that ends between two ranges of blocks (done, cancelled, its client gone, an error) clears the mark, and the
  next image starts at block 0. While an image of another request is between its blocks a pixel request waits in the queue for
  that image's last block, never for the other request's whole prompt; a text request is never held.
- **Which images, and when.** An image is encoded before the first piece of the prompt that reaches its span, in prompt order.
  Images whose spans end before the position the read starts from are not encoded; a resume that starts further back encodes
  the ones it then needs. A prefill that reached an unwritten span would stop the engine (an assertion of this rule).
- **The sum check.** At an image's last block each rank sums its span's bytes, and the four sums are compared in one host
  all-gather; a difference ends the request with an error and frees its rows. It proves the four ranks hold the same rows.
- **Paced beside the decoding users** (`engine/server/vit_sched.h`, `engine/server/interleave.h`). With nobody decoding, a
  request's images are encoded back to back and its prompt is processed as any prompt is. With users decoding, an image request is
  read in one run only when its encode, its rows and the end of a decode run together fit the gap (`--interleave-gap-ms`, 600);
  else it goes in units. An encode unit is the longest run of whole images whose time fits the gap less the measured end of a
  decode run, at least one image; an image longer than that goes in ranges of blocks. An image's time is modelled as
  `k x blocks / 32 x (4.24 + 0.021823 n + 8.192e-7 n^2)` ms for n patches, fitted to five measured sizes, with `k` following
  the measured units. The end of a decode run is measured as well: a paused run ends with a step whose tokens are not sent, and the
  next run's first tokens come one step after it starts, so the users wait for a unit plus about two steps; the engine keeps a
  moving average of that extra (at least one decode step) and subtracts it from the gap. Rank 0 prints one line for each request
  processed in pieces: its encode units and row pieces, the longest of each and the longest pause across one.

Test builds (`-DDSGB10_TEST_FAULTS=ON`) read five settings on this path, which the served build does not:
`DSGB10_VIT_FAIL=rank:R,at:alloc|vote` refuses rank R's work area or makes its sum differ; `DSGB10_VIT_DUMP=DIR` has every rank
write an encoded image's span rows to `DIR/rank<R>-key<K>-span<S>.bin`; `DSGB10_VIT_POISON=1` fills the work area with NaN before
every range that carries nothing over; `DSGB10_VIT_BUDGET_MS=x` sets an encode unit's budget (0.001: a block at a time);
`DSGB10_VIT_MIDTEST=1` starts a second image while the first is between its blocks, which must stop the engine.

## Memory: what left rank 0

What the encoder holds on each Spark, by the path in use:

| | the encoder in the engine (served) | the container encoder (`--vision container`) |
|---|---|---|
| rank 0, a line outside the store | the media worker, `--media-mib` 512 MiB (its container's cap) | the encoder container, `--encoder-mib` 1,260 MiB |
| every rank, outside the store | its quarter of the tower, 232.80 MiB of device memory; 6 MiB of pinned memory (used on rank 0) | nothing |
| every rank, inside the store | the work area, 128 chunks (356 MiB), from the start on; a running request's image rows (204 a chunk) | a running request's image rows (204 a chunk) |
| what a request carries | its pixels, 588 bytes a patch, in the front-end and in rank 0's frame | its rows, 10 KiB each, from the encoder to the front-end to rank 0, then broadcast to ranks 1 to 3 |

So rank 0 no longer holds the whole tower, PyTorch or the rows' copies, and nothing on this path can grow past its line while the
engine runs: every byte is a fixed line counted before the store is sized (the media worker held to its line by its container's
cap), or a part of the store. Each Spark carries its quarter of the tower and the work area.

Measured: with the vision weights loaded the store was 104 chunks smaller (8,523 against 8,615 and 8,639 on two starts without
them), and the load lowered free device memory by 285 to 294 MiB a Spark for 232.80 MiB of weights (52 to 62 MiB a Spark not
accounted for); the work area is exactly 128 chunks (the store's count after the start's hand-over is the start line's less the
work area's 128 and the Engram cache's chunks). Estimated from these lines, not measured as one comparison: on rank 0, which sets
every Spark's store, the 748 MiB less outside the store, less the weights' 104 chunks (289 MiB) and the 6 MiB buffer, less the
work area's 356 MiB, leaves the conversations about 100 MiB more on every Spark.

## Options

| Where | Option | Default | |
|---|---|---|---|
| engine | `--vision-weights FILE` | unset | each rank's quarter of the tower; without it the engine serves text and answers a pixel request with an error |
| engine | `--media-mib N` | 512 | the media worker's cap, a line outside the store on rank 0 when the vision weights are loaded; the same number as the container's `--memory` in `deploy/pico-media.service` |
| engine | `--encoder-mib N` | 1,260 | the encoder container's cap, the same line without the vision weights |
| engine | `--frame-stage-mib N` | 400 | the largest request frame the engine reads |
| engine | `--interleave-gap-ms N` | 600 | the longest a decoding user should wait between steps; encode units and row pieces are sized from it |
| front-end | `--vision engine\|container` | engine | where images are encoded: the engine (pixels from the media worker) or the encoder container (rows) |
| front-end | `--media-addr HOST:PORT` | unset | the media worker; without it image and clip requests are refused; an explicit `--vision engine` without it is an error |
| front-end | `--vision-addr HOST:PORT` | unset | the encoder container, with `--vision container` |
| front-end | `--max-image-rows N` | 40,000 | the rows (image tokens) of all one request's images, 1 to 110,000; the media worker's own `--max-rows` has the same default, and the lower applies |
| front-end | `--max-image-rows-in-flight N` | 80,000 | the rows of all image requests in flight, not below `--max-image-rows` |
| front-end | `--front-image-mib N` | 400 | the pixel bytes the image requests in flight count for against the front-end's table budget (`--front-table-mib`) |
| front-end | `--max-body-mb N` | 64 | the JSON body, 1 to 4,096 |
| media worker | `--max-images`, `--max-rows`, `--max-bytes`, `--max-request-bytes`, `--max-image-pixels`, `--max-video-bytes`, `--video-timeout-s` | 100, 40,000, 32 MiB, 72 MiB, 40,000,000, 128 MiB, 60 s | [media.md](media.md) |

As served (`deploy/pico.env.example`): the engine gets `--vision-weights /home/<user>/dsgb10/vision/vision.rank{rank}.dsw
--media-mib 512` and the gap's default; the front-end gets `--media-addr 127.0.0.1:18951` and encodes in the engine by default;
`deploy/pico-media.service` runs the worker on 127.0.0.1:18951 with `--max-images 100`, host networking, no GPU, a read-only root
and 512 MiB with no swap. The vision files need no GPU to make ([setup.md](setup.md), section 5.2).

## Limits and refusals

A request takes up to **100 images across all its messages**, earlier turns included; a clip's frames count as images. Image rows
and text must fit the context window together. Memory follows rows (image tokens), not the number of images: about 382 rows for a
768 x 768 image and 1,024 at most. Text requests are never counted, delayed or refused by these limits.

An image request that cannot run is answered in plain words and never reaches the engine: 400 for a bad image, too many images or
rows over the cap; 413 for an image over the media worker's pixel limit, a request whose pixels pass `--front-image-mib` (and
never more than the engine's 400 MiB frame) or a body over its limit; 429 with `Retry-After: 5` when the rows in flight are over
their limit or the front-end is over its own memory limit (`--front-mib`; the request is not queued, so its body is not held
while it waits); 503 when the media worker is down (text requests are unaffected). `input_image` (the Responses API) and audio
parts are refused.

A clip that cannot be used is answered 400: `cannot read the video: not a readable video`, `the clip has no video stream`,
`video url must be a data: URL or http(s)`, `video URL host H is not a public address`, `could not fetch the video from its URL`.
The decoder's own error text goes to the media worker's log. A clip sent inline must fit the JSON body (64 MiB: about 47 MiB of
video); a larger one goes by URL, up to 128 MiB.

## The pixels frame

A request carries its images as rows (flag 8, `kFlagImages`, the container encoder's) or as pixels (flag 32, `kFlagPixels`), never
both. After the prompt ids: `u32 n` (1 to 100); `n` spans (`u32 start`, `u32 rows`, `u64 hash`); `n` grids `u16 n_h, u16 n_w`
(patches; the array padded to 8 bytes); then each image's `588 x n_h x n_w` bytes of u8 (the media worker's padded image), each
padded to 8 bytes. The frame holds exactly these bytes. A span's rows are `ceil(n_h / 3) x (ceil(n_w / 3) + 1) + 2`, at most
1,024, and a grid is at most 9,216 patches. The flag is accepted on a connection of either protocol version, with or without the
prefix flag, and changes no response frame ([protocol.md](protocol.md)).

A frame whose bytes cannot be accounted for (both flags, `n` out of range, a byte count that is not exactly the layout's) is
malformed and closes the connection. A well-formed frame the engine will not run (a grid of 0 patches or more than 9,216, rows
that do not match the grid, rows over 1,024, an engine without the encoder or its work area) is answered DONE with an error and
the connection lives. The layout and its checks are `engine/server/pixels.h`; `engine/server/tests/pixel_client.py` sends one.

## Measured

On the four Dell Pro Max GB10 of this cluster. These are the measurements of the changes themselves, on the builds that brought
them in; the served build's speeds are in the README.

**One image's encode** (the whole tower and the aligner):

| Image | In the engine, four Sparks over the ring | pico-dgx-ds4's container on rank 0's GPU |
|---|---|---|
| 1,521 patches | 39.6 ms | |
| 768 x 768 (3,025 patches, 382 rows) | 77.8 ms | 133.7 ms |
| the largest (9,189 patches, 1,024 rows) | 270.2 ms | 564.1 ms |
| 32 frames of 768 x 768 (12,224 rows) | about 2.5 s (32 x 77.8 ms) | 4.28 s |

In the engine: the four Sparks with the engine stopped (`vit_ring_test`), where the gathers were 51 to 66 % of each image's time;
the 768 x 768 figure is the engine's own time for each of 32 frames with eight users decoding. The container: its own timer,
medians of five, the engine serving. A rank's compute alone (`bench_vit`, one Spark, the engine stopped) is 12.8, 27.3 and 141.8 ms
at 1,521, 3,025 and 9,189 patches, against 43.9, 103.4 and 496.5 ms for the same kernels unsplit on one GPU.

**Beside eight users decoding**, through the front-end's API: eight users stream their outputs and one image request arrives
among them; medians of five runs, ranges in brackets. "32 frames" is 32 frames of a video, 768 x 768 each, sent as 32 images
(12,224 rows), so no video is decoded in these runs:

| | The container, the rows processed in one run | The container, the rows processed in pieces | The engine encodes, the rows processed in pieces |
|---|---|---|---|
| one image: time to first token | 0.737 s (0.703 to 0.750) | 0.720 s (0.706 to 0.761) | 0.443 s (0.359 to 0.557) |
| one image: the users' longest pause | 0.326 s (0.323 to 0.331) | 0.322 s (0.313 to 0.462) | 0.510 s (0.507 to 0.524) |
| 32 frames: time to first token | 12.56 s (12.46 to 12.62) | 14.98 s (14.65 to 15.40) | 10.20 s (9.28 to 10.24) |
| 32 frames: the users' longest pause | 2.01 s (1.94 to 2.17) | 0.59 s (0.56 to 0.65) | 0.58 s (0.56 to 0.64) |
| 32 frames: the eight users' output while it is processed | 89.6 t/s | 105.5 t/s | 108.0 t/s |

With no image request the eight users' output was about 190 t/s together on all three builds, and their longest pause 0.31 s
(the first build's run without an image). Against the container with the rows read in one run, the engine's encoder gave the
image its first token 40 % sooner and the 32 frames 19 % sooner, and cut the users' longest pause during the frames by 71 %.
Reading the rows in pieces is what removed the two-second pause; with the container still encoding it added 19 % to the frames'
time to first token, and encoding in the engine took that back and more. One image costs the users a longer single pause than
with the container (0.51 against 0.33 s, inside the 600 ms gap): the container encoded in its own process while the engine kept
decoding, where the engine's encode takes a turn between their steps.

The three columns are three builds: the first two with the container encoder, the third a later build with the encoder in the
engine and the media worker (that build also carried a version of the split cache that is not in the served build). On the build
that brought the encoder in, timed by the engine itself from the request's arrival at rank 0's socket (no front-end, no media
worker; five runs): one image 0.285 s, the 32 frames 11.2 s, and the longest pause across an encode unit 557 to 567 ms against the
600 ms gap in every run; the users saw about 0.1 s more, the end of a decode run, which every prompt processed in pieces adds.

**Correctness.**

- Every image the engine encoded on four ranks was equal in bytes on every rank and to the unsplit run: whole images, ranges of
  blocks with decode steps and another request's pieces between them, two requests at once, cancels at 0.05 to 2 s, and a resume
  that encodes its image after its session was evicted.
- Each operation of six of the 32 blocks, from the first to the last, and of the patch embedding was checked on the GPU against
  the checkpoint's own PyTorch code on test images, each fed the reference's own input, and each of nine deliberately wrong builds
  was detected.
- The media worker's pixels, through the engine's table, are the reference's patches bit for bit (11 images, 4 formats).
- Text outputs are equal with and without the vision weights loaded.

**Memory.**

- The tower's quarter: 232.80 MiB of device memory a rank; the store 104 chunks smaller with it; the work area exactly 128 chunks
  ([Memory: what left rank 0](#memory-what-left-rank-0)).
- The media worker (one Spark, its container's peak memory under the unit's settings): 12 to 36 MiB idle; 388 MiB for a
  40-megapixel image, 421 MiB for 38 four-megapixel pages and a 40-megapixel image in one request, 186 MiB for a 47 MiB clip sent
  inline, 202 MiB for a 128 MiB clip by URL, all under its 512 MiB. Sampling 32 frames took about 1.5 s for the 47 MiB clip and
  1.8 s for the 128 MiB one (720p test clips, timed by the test client).
- Image rows in the store: a request of 39,600 rows held 195 chunks on every Spark and gave all of them back when it ended, the
  other Sparks' available memory unchanged; moving the rows from pinned host buffers into the store changed an image request's
  time to first token by +0.5 % (584.1 against 581.3 ms; measured with the container encoder).

**Image conversations.** The second turn of an image conversation took 2,042 of its 2,051 prompt tokens from the cache and
answered its first token in 0.25 s (the first turn, cold: 1.40 s); a third turn with another image in the same place took nothing
from the cache (0.50 s). Measured through the front-end on an earlier build that encoded with the container;
which conversation continues is the front-end's choice on either path.

## Tests

On any machine (no GPU):

- `engine/vision/tests`: `vit_host_test` (the weights' quarters, grids and host references), `vision_dsw_roundtrip` (the vision
  files cut, written and read back), `vit_gather_test` (the chunked host gather).
- `engine/model/tests/img_rows_test.cpp`, `engine/server/tests/img_hold_test.cpp` (the rows in the store's chunks: taken once,
  read by every pass, freed on every way a request ends), `vit_sched_test.cpp` (which unit is next, one image at a time, the mark
  cleared by every way a request ends), `interleave_test.cpp` (a timeline of encode units, row pieces and decode runs),
  `pixels_test.cpp` (the pixels frame), `pool_rules_test.cpp` (the image process's line).
- `frontend`: `server_pixels_test.rs` (the router with `--vision engine` against the mock engine and a stand-in media worker;
  `--vision container` as the control), the tests in `vision.rs`, `video.rs`, `session.rs` (`a_saved_table_keeps_its_images`) and
  `main.rs` (`the_vision_flag`).
- `tools/vision`, CPU only: `test_media_worker.py` (the ported grid over a sweep of sizes; the pixels through the engine's table
  equal to the reference's patches; the reference side needs PyTorch on the CPU), `test_media_worker_fetch.py` (an http(s) name
  resolved once and its address checked), `test_encoder_limits.py` (the request limits), `test_video_frames.py` (PyAV and
  Pillow), `test_video_errors.py` (the refusals in plain words; its decoder tests need PyAV), `test_vision_exactness.py` (the
  exactness tools; PyTorch on the CPU for the dump tool). Each runs as:

```bash
python3 -m unittest discover -s tools/vision -p test_media_worker.py -v
```

On one GPU, the engine stopped: `engine/vision/tests` `vit_test` (every kernel and the tower: the four-rank run stepped on one GPU
against the unsplit run in bytes on four grids, each rank's phase machine, and ten deliberately wrong builds that must give other
bytes), `bench_vit` (the kernels' times), `vit_npy_test` and `vit_teacher` with `tools/vision/dump_ref.py`, `vit_compare.py`,
`null_control.py` and `f0-gpu-run.sh` (each operation against the checkpoint's own code).

On the four Sparks: `vit_ring_test` (started by `engine/comm/run4-vit.sh`, the engine stopped: each rank's span rows over the real
fabric equal to the unsplit run's, and each image's time); `engine/vision/tests/vision_weights_gpu.py` (the vision files on four
Sparks: the set check and what the load costs); `engine/server/tests/vit_engine_test.py` on a test build (rows in bytes, ranges of
blocks, two requests, a request held behind another's image, cancels, the sum check, a resume, the work area at the start, the clip
beside decoding users) and `image_store_test.py` (the rows in the store: outputs, first token, nothing left behind, cancels,
resumes); `tools/vision/media_highwater.sh` (the media worker's peak memory under its cap, on a Spark that is not serving).

Against a running endpoint (the API key; for images and clips the media worker and `--media-addr`):

```bash
python3 tools/vision/smoke_100.py http://127.0.0.1:PORT/v1/chat/completions --key-file KEYFILE --output vision-100.json
python3 tools/vision/smoke_video.py http://127.0.0.1:PORT/v1/chat/completions clip.mp4 --key-file KEYFILE --output video.json
```

`smoke_100.py` makes numbered cards locally (Pillow and a TrueType font; `--font PATH` picks another), checks the first and last
image of a full batch, checks that 101 images and broken image bytes are answered 400, and checks text afterwards. `smoke_video.py`
tests streaming and non-streaming requests, bad inputs and the shared frame budget; with a clip that shows known codes in turn,
`--expected 1111 2222 3333` checks their order. `imgtest.py` (known-answer images and the time to first token an image adds),
`conc_img.py` (eight users decoding and one image or text request) and `engine/server/tests/image_overlap.py` (two image prompts
processed in pieces at once, each output equal to its own alone) are the others.

## The container encoder

pico-dgx-ds4's side encoder stays available: start the front-end with `--vision container --vision-addr HOST:PORT` instead of
`--media-addr`; the engine needs no `--vision-weights`, the request carries rows (flag 8), and the line outside the store on rank 0
is `--encoder-mib` (1,260). Its rows live in the store's chunks like the engine's (every rank takes chunks for them at the
request's first pass, rank 0 sends the rows once and keeps no copy, and every rank gives them back at the end), and its requests
are processed in pieces beside decoding users the same way. Its preprocessing, its attention shim (contiguous 4-D views, so PyTorch
uses its flash kernel) and its own run line are in [pico-dgx-ds4's vision.md](../ds41f-4xgx10-switchless/docs/vision.md):

```
docker run -d --name dsgx10-vision --restart unless-stopped --gpus all --net=host --memory=6g -e VISION_PRELOAD=1 \
  -v $MODEL:/model:ro -v $PWD/tools/vision:/app:ro --entrypoint python3 <image with torch + PIL + safetensors> \
  /app/vision_encoder.py --host 127.0.0.1 --port 18950 --max-images 100
dsfront serve ... --vision container --vision-addr 127.0.0.1:18950
```

The encoder's kind is part of every image's hash, so a conversation made under one encoder is never continued under the other:
image conversations start cold once when the path changes. The media worker's script also listens on 18950 when given no
`--port`; the served unit runs it on 18951, so the two can run side by side only on different ports.
