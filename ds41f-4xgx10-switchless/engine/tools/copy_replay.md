# copy_replay: copy drafting replayed on recorded conversations

The engine's own host matcher and row-count functions (`model/copy.h`: `index_range`, `propose`, `gate_rows`) run over recorded
conversations, with no GPU and no engine. At every step of every output it proposes copy tokens from the history so far and
scores them against what the model wrote next. Use it to check a survival table or a row-count rule against your own traffic before
it runs on the engine.

Files: `copy_replay.h` (the replay, and a `Rule` interface for a row-count rule) and `copy_replay.cpp` (the command line and the
self-test). CMake target `copy_replay`, host only. It includes only `model/copy.h`.

## Build and self-test

    cmake --build build --target copy_replay && ./build/copy_replay

With no arguments it runs its self-test: planted repeats with known answers, a stream with none, the finder of the model's
outputs, and the matcher against a plain latest-earlier-occurrence reference at every position of 20 random streams.

## Input

Files of little-endian int32 token ids, conversations one after another; a conversation starts at the begin id (0). The
model's output is what follows an assistant marker (128804) and the think or end-think id after it, up to and including the
next end id (1). `--spans last` (the default) scores only the last output of each conversation, the one the engine decoded
with its reasoning in place; `--spans all` scores every output (the template drops earlier outputs' reasoning, so earlier
turns are not exactly what was decoded). `--ids BOS,ASSISTANT,EOS` changes the ids.

The exact ids of a conversation are the engine's own: the files in `<disk dir>/ids/` hold the ids of the conversations
written to disk, each after a 24-byte header (an 8-byte magic, the count, a checksum); `tail -c +25 FILE > conv.i32` makes one
a stream file. Ids rebuilt from a request log are exact only where the log keeps the whole text.

Conversations are users' data: keep the files out of the repository and print aggregates only.

## Run

    copy_replay --table builtin --table copy-T.txt --table empirical \
        --group NAME --dir DIR [--group NAME2 --dir DIR2 --combined all] [--spans all] [--min-match N] [--json]

`--table` is repeatable: `builtin` (the engine's built-in table), `learned` (the engine's learner, on with copy drafting,
starting from the built-in table, with each conversation's own correction), a file (the format of `DSGB10_COPY_TABLE`), or
`empirical` (the counts of the data itself: a reference, not a candidate). `--gate ratio|marginal|thresh`, `--thresh`, `--step-ms` and
`--chain-prev` set the row-count rule and its cost model (defaults: the ratio rule, `server/cmd.h` `spec_row_ms`, no chain).

## What it reports

Per source: conversations, outputs, prompt and predicted tokens. With every proposed row verified (as `DSGB10_COPY_DUMP=2`
does), per match-length bucket: the proposals, how many leading rows were right, and for each depth j the observed chance
that copies 1..j are all right beside each table's value. Then for each table, the run with that rule choosing the rows: steps, tokens a step, rows
verified, accepted, wasted, right rows not verified, accepted copy tokens per 1,000 predicted tokens, and a modelled speedup
(plain decode steps at `step_ms[0]` against the rule's steps' `step_ms[rows]`). The same files and options give the same
output.

The counts are exact for the outputs as recorded: the engine draws each sampled token from the request's seed and its
position, so a draft is accepted exactly when it equals the token that was produced. The time is modelled.

## Limits

- Copies alone: no draft chain before them (`--chain-prev` scales every copy's chance by a fixed chain survival), so the
  speedup is a model of copy rows only.
- Proposal points follow the replay's own steps; a run with a chain would reach some positions in a different order.

## A row-count rule

Subclass `Rule` (`copy_replay.h`): `rows()` picks how many rows a step verifies; `outcome(bucket, depth, hit)` is called
for each accepted row and for the one token after them; `begin_conversation()` resets a conversation's state. Add it to
`run_source` beside `TableRule`.
