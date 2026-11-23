# Rank files: this repository's changes

pico-dgx-ds4's [layout.md](../ds41f-4xgx10-switchless/docs/layout.md) describes the per-rank weight files of the engine this repository
builds on: what each rank holds, the tile layouts, the file format and the tools that cut and check the files. This page
describes what this repository changed in them.

## Format version 3: the embedding cut by vocabulary

The embedding and the drafter's Markov embedding are cut by vocabulary rows, the rows the head's slice scores: rank `r` holds
rows `[32,320 r, 32,320 (r + 1))` of each. The two tables, 1.233 GiB and 63 MiB whole, become a quarter on every rank, which
frees about 0.97 GiB a rank (computed).

| Tensor | Shape | A rank holds | Layout |
|---|---|---|---|
| `embed.weight` | bf16 [129280, 5120] | rows [32320 r, +32320) | natural |
| `mtp.2.markov_head.embed` | bf16 [129280, 256] | rows [32320 r, +32320) | natural |

A lookup is an exchange: the rank that owns the token's id writes the row into its slot, the other three write bf16 -0.0, and
the all-reduce (an fp32 sum in rank order) returns the row bit for bit, since x + (-0.0) == x for every x
(`common/embed_shard.h`). Decode adds one all-reduce of R x 10 KiB before layer 0; the drafter one for its token rows and one for
each Markov step's row; prefill sends the rows in the all-to-all it already makes, to the rank whose chunk rows they are. An id
outside [0, 129,280) stops the kernel and fails the host's check.

A version 3 file holds 1,609 tensors and is about 71.4 GiB (version 2's 72.38 GiB less 0.97, computed, not read off a
file; `python3 engine/weights/toc.py FILE` prints a file's exact size). An engine refuses a file of another
version, so after an upgrade that changes the version the files are cut again with `dsw_convert` ([setup.md, 5.1](setup.md#5-the-files-each-spark-needs)).

Measured on the four Sparks at start, against the same build with the copies kept: the embedding cut gives the store 0.98 GiB a
rank, outputs are bit-equal and the store's digests equal on the four ranks (README, [Weights and copies](../README.md#weights-and-copies);
tests `embed_shard_test`, `dsw_dest_test`).

## The vision files

The model's vision tower is not in the rank file: each rank's quarter of it is a file of its own, `vision.rank<r>.dsw`
(232.80 MiB a rank), cut by `dsw_convert --vision` and checked by `dsw_verify --vision`; the engine loads it with
`--vision-weights` and the ranks check at start that the four files are one set ([media.md](media.md#the-engines-encoder),
[vision.md](vision.md)). The Engram tables stay outside both, in each Spark's own shards.

## Where the files live

Every Spark keeps its own files under `~/dsgb10/`: `weights/rank<r>.dsw`, `vision/vision.rank<r>.dsw` and the Engram shards in
`engram/`. All are cut on rank 0 from its one copy of the checkpoint and copied out; each Spark checks what arrived with
`dsw_load FILE --verify-crc --rank <r>` ([setup.md, section 5](setup.md#5-the-files-each-spark-needs)).

## The checkpoint

This repository serves `deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`
([setup.md, section 3](setup.md#3-the-checkpoint)); the confidence network that decides how many drafts a step checks was fitted
on it. The checkpoints that pico-dgx-ds4's files and reference hashes were made from are named in its layout.md.
