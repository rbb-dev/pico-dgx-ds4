# The runtime abliteration

A published abliteration edits two weights of each layer, `attn.wo_b` and `ffn.shared_experts.w2`, as
`W' = W - s Q Q^T W`: Q an orthonormal basis of a few directions, s a strength per layer and weight. The engine applies the
same edit to those weights' outputs, `y' = y - s Q (Q^T y)`, on the rows of the requests that ask for it. The weights
stay the checkpoint's: one set of rank files serves the model as published and its abliterated variant, in the same
batch, and a request picks per request.

## Serving it

Start every rank with `--ablit FILE` (below). With the services of [setup.md](setup.md), put `ablit.bin` on rank 0 only: the start helper copies it to the workers at every start. The front-end then serves two model names from the one engine:
`--model-name NAME` runs the checkpoint as published, and `NAME-ablit` runs the same request with the edit applied.
`GET /v1/models` lists both, a request picks one with its `model` field (names match exactly), and the response's
`model` is the name the request asked for; any other `model` runs as `NAME`. Conversations on the two names never share
a session, since a session is keyed by its weights path too. `--no-ablit-alias` on the front-end serves `NAME` only.
The front-end does not know whether the engine has a file: without `--ablit` a request for `NAME-ablit` runs as published,
so serve the second name only with the file loaded.

## The file

The file for DeepSeek-V4.1-Flash-1017 (revision `1bae0e10`) is in this repository:
`ds41f-4xgx10-switchless/tools/ablit/ablit.bin` (778,624 bytes), made from the edit by mrfakename as below. Copy it to
`~/dsgb10/ablit.bin` on rank 0; `setup.md` gives its sha256. To make it again, or from another edit,
`tools/ablit/make_ablit.py` makes it from the edit and the checkpoint:

```bash
python3 tools/ablit/make_ablit.py --edit abliteration-edit.npz --ckpt ~/dsgb10/model --out ~/dsgb10/ablit.bin
# or straight from the Hub, reading only the 30 shared-expert w2 tensors it needs (about 350 MB):
python3 tools/ablit/make_ablit.py --edit abliteration-edit.npz \
    --hf deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170 --out ablit.bin
```

The edit used here is the `abliteration-edit.npz` in the Hub repository mrfakename/DeepSeek-V4.1-Flash-1017-abliterated (MIT; download that file first; a rank-2
subspace, wo_b strengths on layers 0-31 and shared-expert strengths on layers 10-39). Its Q, strengths and, per layer,
`G = W2^T Q` from the checkpoint's shared expert (`[2304][2]`) make a 778,624-byte file (the script prints the size and the sha256 of what it wrote). Every rank reads the same file
(`--ablit FILE`) and keeps its G rows `[576 r, 576 r + 576)`, the intermediate units it holds.

## What the engine does

- wo_b is row-parallel: each rank's partial sum is full length before the all-reduce, so each rank projects its own
  partial (model.cu `attention`) and the sum is the projected sum.
- The shared expert's output is added inside the MoE kernels to the routed experts' sum, so it is projected through its
  input: `Q^T (W2 h) = G^T h`, h the shared expert's activations as the MoE kernels left them (`moe_routed`'s workspace,
  `moe_prefill_shared` for the wide steps and prefill). The kernels reading h and correcting the MoE's partial run
  after the MoE (model/ablit.cu); the MoE kernels are unchanged.
- Prefill (a chunk is one session's): attention's sums are projected on each rank's own rows; the shared expert's dot
  products of every row are all-gathered (8 bytes a row) and each rank corrects its rows. CED's replay goes through the
  same code.
- A row's strength is its session's `Sampling::ablit` (the front-end's SUBMIT flag `kFlagAblit`) times the layer's s.
  The step kernels read it from the step input (`StepSamp::ablit`), so the captured graphs serve both kinds of rows; a
  row at 0 keeps its bits. Prefill skips the edit's launches when the chunk's session does not ask for it.
- Without `--ablit` nothing is launched.

Cost (estimated, not timed on its own): with `--ablit` on, one launch per edited site and layer and step (62 for this
edit), each a few microseconds; an edited row reads its 5,120 values once more.

## Tests

`model/tests/ablit_test` (one GB10): the file's per-rank G slices; the projected rows against a double-precision
reference; the rows without the edit bit for bit; the prefill pair against the reference with the gather done by copies.
