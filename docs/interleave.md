# Interleaved prefill

A prompt processed whole stops every session that is decoding for as long as the processing takes: about 20 s for a
200K-token prompt processed cold, about 97 s for 1M. With interleaved prefill a long prompt is processed in pieces, and the
sessions that decode get a window between the pieces. Nobody waits longer than a set gap between tokens, the
decoders get a set share of the time, and every prompt's tokens and log-probabilities are the ones it gets processed alone.

The code: [`interleave.h`](../ds41f-4xgx10-switchless/engine/server/interleave.h) (the decisions, pure functions, tested by
`tests/interleave_test.cpp` without a GPU) and [`serve.cpp`](../ds41f-4xgx10-switchless/engine/server/serve.cpp) (`il_pass`, the reader:
`pf_begin`, `pf_piece`, `pf_end`).

## What is processed in pieces

- **Rows to compute, not prompt length.** A prompt is long when its rows to compute (its length minus what its
  session or base gives back, the SSD tier included) are above `--interleave-whole-rows` (32,768). A 650K-token turn
  of a conversation whose first 640K are cached computes 10K rows and is processed whole.
- **Shorter prompts are processed whole:** the decoders wait for them (a 32K prompt takes about 3 s, estimated from
  the cold rate).
- **Nobody decoding: the same speed.** A long prompt with no session decoding is processed in pieces of 16,384 rows
  while the prefill's lent set of buffers is held (8,192 rows otherwise; see [the README](../README.md#long-prompts-processed-16384-rows-at-a-time)), back to
  back, as fast as processed whole (measured below); a session decoding with no long prompt waiting is unaffected.
- **A queued prompt holds nothing.** Long prompts wait in a queue, first come first served, with no slot and no
  memory; the front one begins when a slot can take it. A cancel ends a queued prompt at once.
- **A second reader.** A queued prompt with at most `--interleave-push-rows` (131,072) rows to compute starts beside
  the one being processed and processes its pieces first, so a 60K prompt does not wait behind a 1M one.
- **Resumes.** A paused request whose state left its slot, with more rows to process again than the whole limit, is processed
  again in pieces like a new long prompt; the shared session store keeps a paused session's state, so a resume
  continues from it. `resume_test.py` checks that resumes in turns are exact; no test drives a resume that is processed in pieces.
- **Images.** Each conversation's images (their spans and rows) are kept with its slot and switched in with it, so the
  pieces of two image prompts processed in turns each see their own images only.

## How the time is shared

The scheduler (`il::next`) chooses one thing a pass: admit an arrival, begin a reader, process a piece, or decode.

- **The piece size follows the gap.** A piece of R rows takes k x (76 ms + 0.092 ms x R) on the device (measured,
  with k a running correction from the pieces the engine times). The piece is the largest multiple of 1,024 rows
  whose time plus one decode step fits `--interleave-gap-ms` (600): 5,120 rows with one session decoding, 4,096 with
  three. `--interleave-rows N` fixes it instead.
- **The window follows the share.** After a piece of T ms the decoders get T x share / (100 - share) ms of decode
  (`--interleave-share`, 40 %: 0.67 x T), at most `--interleave-burst-max-ms` (1,000). The window's end is set at its
  first step, and a decode run ends at the step that crosses it.
- **Arrivals and pieces alternate.** While a prompt is processed, a new request is admitted between two pieces, never two in
  a row, so a stream of short prompts cannot starve the long one; a short prompt waits for at most one piece and one
  window, then its own processing.
- **A CED prompt's pieces** (prefill in the Causal Encoder-Decoder mode, [ced.md](../ds41f-4xgx10-switchless/docs/ced.md)) never leave 1 to 127 rows for the last one (the window's replay needs 128 or none), and the
  first piece of a prompt that restarts off a 1,024 boundary ends on one.

## Settings

| Flag (engine) | Default | Live | |
|---|---|---|---|
| `--interleave` / `--no-interleave` | on | yes | off: every prompt processed whole; `--exact` turns it off |
| `--interleave-whole-rows N` | 32,768 | yes | processed whole up to this many rows to compute (at least `--prefill-rows`) |
| `--interleave-push-rows N` | 131,072 | yes | a second reader up to this many rows to compute; 0: none |
| `--interleave-gap-ms N` | 600 | yes | the longest silence of a decoding session (at least 100) |
| `--interleave-share N` | 40 | yes | % of the time decoders get while a prompt is processed (10 to 90) |
| `--interleave-burst-max-ms N` | 1,000 | yes | one decode window at most (at least 50) |
| `--interleave-rows N` | 0 | no | a fixed piece size (0: from the gap) |

Live settings go to the engine in a `CONFIG` frame ([protocol.md](protocol.md#the-dashboards-frames)); the dashboard's ENGINE SETTINGS box sets
them (SPLIT LONG PROMPTS, TIME FOR OTHERS, MAX PAUSE FOR OTHERS, JUMP THE QUEUE UP TO, NO SPLIT UP TO; [dashboard.md](dashboard.md)). They last until the engine restarts; its flags set them again.
A value out of its flag's range is refused by the front-end and ignored by the engine.

## What the request's DONE says

`queue_us` runs from the SUBMIT to the start of the prompt's processing, so a long prompt's wait in the queue is queue
time; `prefill_us` runs from the start of its processing to its first token, its pieces and the decode windows between them.
A prompt cancelled while queued has all its time as queue time.

## Measured

Four GB10s (TP=4), `--slots 32`, the tests' synthetic token ids; each long timing against a control on the same build
with `--no-interleave` or processed alone in the same run. Each line is one test in `engine/server/tests`, run through rank
0's socket (`image_overlap.py` through the front-end).

| Case (test) | The users decoding | The long prompt | Longest silence | Exact | Processed whole (`--no-interleave`) |
|---|---|---|---|---|---|
| 200K cold while one session decodes (`overlap_test.py`) | 79.3 t/s (36 % of the 222.8 it makes alone) | first token 36.6 s (+82 %) | 0.60 s | the session's tokens and the prompt's first token equal to alone | the session stops 20.19 s; the prompt 20.2 s |
| 200K cold, three sessions decoding, a 4K prompt every second (`stream_test.py`) | 42.8, 36.6, 29.0 t/s; each 4K prompt's first token 0.97 to 1.12 s | 62.0 s (+207 % on 20.2 alone) | 0.80 s | | every session stops 20.2 s; a 4K prompt waits up to 20 s (computed) |
| a 60K prompt arrives while a 300K one is processed (`push_test.py`) | | 60K: first token 11.9 s, 44 s before the 300K one | 0.62 s | first tokens equal | 60K waits behind the 300K: 57.5 s (alone 6.5 s) |
| cancels (`cancel_test.py`) | | queued: DONE 450 ms after the cancel, all its time queue time; being processed: DONE in 449 ms, its rows kept for its next turn | 0.60 s | first token equal | |
| two image prompts processed in turns while a session decodes (`image_overlap.py`, through the front-end's HTTP API) | | two prompts of about 47K and 40K tokens | | outputs and top-2 log-probabilities equal | one image set for the whole model: the image prompt sees the other's image |
| the share, changed live (`config_test.py`) | share 20 %: 47.0 t/s; share 80 %: 159.4 t/s | 35.7 s and 94.2 s | | | |
| a 40K prompt waiting behind a 100K one, the second reader off (`queue_test.py`) | | DONE: queue 20.12 s (its wait), prefill 8.62 s (its own processing) | | | |
| one prompt processed alone, nobody decoding (`bit_probe.py`, timings) | | 393K 40.27 s, 200K 20.16 s | | equal | 40.18 s, 20.13 s (+0.2 %, inside the 4-5 % noise) |
| decode with no long prompt (a bench: chat and code at one user, three users) | 80.6, 130.9, 183.6 t/s | | | | 80.7, 130.8, 183.1 t/s (the same build with `--no-interleave`) |

The cost is the long prompt's own time: it shares the device with the decoders, so at the default share it takes
about 80 % longer with one session decoding. The share setting moves time between the two. Two sessions decoding
together in one batch after a long prompt differ from their runs alone in their last bits; that is the batch path's, the
same with interleave off.
