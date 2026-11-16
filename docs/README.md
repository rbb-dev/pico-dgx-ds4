# This repository's documents

The documents of this repository, for the engine it serves. Where pico-dgx-ds4's own records cover a subject (the engine,
batching, the rank files, the protocol, the ring), the page here holds what this repository changed or added and links the record
in its first line; operations.md and vision.md are whole pages of their own. The front page is the [README](../README.md); the served settings are
[`deploy/pico.env.example`](../deploy/pico.env.example).

**Running it**

- [setup.md](setup.md): from four Sparks, a switch and the checkpoint to a first answer: the files each Spark needs, the
  settings, the services, the media worker, fans and clock, the first checks.
- [operations.md](operations.md): the running service: start, stop, upgrade, rollback, logs, what the engine's log lines mean,
  fans and the GPU clock.
- [dashboard.md](dashboard.md): the `/dashboard` page: sign-in, its routes, the tiles and charts, the node sampler, the fan and
  clock box.
- The network: [NETWORK.md](../NETWORK.md) at the repository root.

**Memory and the conversation cache**

- [store.md](store.md): the memory store: what lives in it, how its size is chosen from `--pool-free-gib`, its tenants, what
  is checked at start and while serving.
- [split.md](split.md): the split cache: idle conversations written to the SSDs as each Spark's quarter, the modes and options
  of `--kv-split`, eviction and read-back.

**Prompts, decoding and the model**

- [engine.md](engine.md): what this repository changed in the engine: start-up, sessions and the chunk pool, memory given back
  on demand, the disk tier piece by piece, the split cache, context past 1M, the split index scan, the Engram cache in the display
  reservation, crash checkpoints, the file formats.
- [interleave.md](interleave.md): long prompts processed in pieces between the decoding users' steps, its settings and
  measurements.
- [batching.md](batching.md): a request that ends inside a batch, the draft count for several sessions, and copy drafting.
- [kernel-trims.md](kernel-trims.md): six kernel changes, each bit for bit the reference kernel's, with their measurements
  and the commands to check them.
- [layout.md](layout.md): the rank files' version 3 (the embedding cut by vocabulary), the vision files, the served checkpoint.
- [ring.md](ring.md): pico-dgx-ds4's ring settings that the served settings switch on, and the prefill's join clock.
- [protocol.md](protocol.md): the frames and fields added between the front-end and rank 0: the dashboard's, the ablit flag,
  the context past 1M, the suffix-only request, images.
- [ablit.md](ablit.md): the runtime abliteration behind the `-ablit` model name.

**Images and clips**

- [vision.md](vision.md): where each part of vision runs, what moved into the engine, and the path of an image or a clip from
  the client to the first token.
- [media.md](media.md): the media worker, the front-end's image options, the request frame with pixels and the engine's
  encoder.

pico-dgx-ds4's own records (the engine this one builds on, as it stood when each was written) are in
[`ds41f-4xgx10-switchless/docs/`](../ds41f-4xgx10-switchless/docs/), and pico-dgx-ds4's engine README is
[`ds41f-4xgx10-switchless/README.md`](../ds41f-4xgx10-switchless/README.md).
