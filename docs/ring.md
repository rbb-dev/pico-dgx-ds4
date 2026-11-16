# The ring: this repository's changes

pico-dgx-ds4's [ring.md](../ds41f-4xgx10-switchless/docs/ring.md) describes the collectives of the engine this repository builds on:
the all-reduce, all-gather and reduce-scatter over ibverbs between the four Sparks, their settings and their measurements. This
page describes what the served settings switch on there and what this repository added.

## pico-dgx-ds4's ring settings in the served settings

The served settings ([`deploy/pico.env.example`](../deploy/pico.env.example)) run two of pico-dgx-ds4's ring options:

- **`DSGB10_RING_DIRECT=2`**: on one switch every rank writes to every other rank itself, with no relays, and merged. Leave it
  out on a cabled ring. `ring_test` reads it too (`engine/comm/ring.cpp`: 0 off, 1 direct, 2 direct and merged), so the ring check
  of [setup.md, section 7](setup.md#7-sudo-the-services-and-the-ring-check) runs once as the fleet file describes and once in this
  mode.
- **`DSGB10_RING_RSBAL=1`**, pico-dgx-ds4's balanced reduce-scatter (the ring's `rs_bal`): in direct mode, for reduce-scatter posts of
  `split_domains_from` bytes or more, it sets how a neighbour's quarter is spread over the card's two PCIe functions (PCI
  domains). With 0, the default, the whole quarter goes on domain 1, so domain 0 carries 0.5 and domain 1 2.5 quarters of a
  post; with 1 each neighbour's quarter goes in two halves, one per domain (1.5 quarters each); with 2 next's quarter goes on
  domain 0 and prev's on domain 1 (1.5 each, whole messages). Only the queue pair a byte travels on changes: the buffers, the
  flags and every sum are the same. Measured with it alone added to the settings served before it: equal at 16 users, and a long
  cold prompt 2.3 % shorter (engine clock, not a clean boot) (README, [Kernels](../README.md#kernels)).

pico-dgx-ds4's ring statistics are an option of the same table: `DSGB10_COMM_STATS=S` (the ring's `stats_s`, default 0) prints the
proxy-side statistics per collective class (all-exchange or reduce-scatter, by size) to stderr every S seconds while there is
traffic. `ring_test` takes both as `--rs-bal` and `--comm-stats`.

## The prefill's join clock (`DSGB10_PF_STALL`)

`DSGB10_PF_STALL=1` (an engine setting, off by default: no event is created or recorded) times how long a prompt chunk's main
stream stands at its two joins with the comm stream (`rs_join`: after the attention's all-to-all, and after the MoE's last
column group) on layers 0 to 19: two timing events bracket each wait, and one pair recorded back to back at the chunk's start
reads what a pair costs by itself. Rank 0 prints one line after a request's `turn:` line:

```
serve: req N joins: C chunks: attention X ms, MoE Y ms (P pairs), floor Z ms a pair
```

A decode step is a CUDA graph and has no join clock.
