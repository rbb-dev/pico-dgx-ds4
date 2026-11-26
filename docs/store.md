# The memory store

On a GB10 the CPU and the GPU share one pool of memory, and everything the engine needs beyond the weights draws on it: the
conversations' caches, the images' rows, the Engram row cache, the buffers of the image encoder, the front-end process and
the box's own page cache. This engine keeps an account of all of it. The part it sizes and hands out is the **store**: the
chunk pool of [engine.md](engine.md#sessions-and-the-store), made once at start on every Spark, plus the groups of it that are lent
out. This page says what lives in the store, how its size is chosen, what is checked at every start, and what the engine
does when memory runs short. The user's view is in the [README](../README.md#the-memory-store).

## The unit

The store is one array of equal **chunks** (the engine's log and STATS call a chunk a number). A chunk is one 8,192-entry
slice of each of four pool-wide arrays (compressed KV, its scales, index keys, their scales): 2,916,352 B (2.78 MiB) on each
Spark, and the same count on all four. Every chunk is created and mapped before the engine serves; serving creates, maps
and unmaps none (captured CUDA graphs keep their addresses). A **group** is 64 chunks, the smallest set whose slices of all
four arrays are whole 2 MiB physical allocations (178 MiB); it is the unit the store lends out.

Rank 0 decides every change and sends it as a command; every rank applies the same operations in the same order. Decisions
read only the allocator and the store's records, never timing or free memory, so all four ranks hold identical sets.
`DSGB10_STORE_CHECK=1` prints the store's digest after each window, state and reservation command, and the four ranks' lines
must agree. The store's bookkeeping lives in a host-only core (`engine/model/store_core.{h,cpp}`) reached through a thin
device interface; an optional trace records every store operation, and traces recorded on the Sparks are replayed against the
core on any machine (`engine/model/tests/store_core_test.cpp`).

## What lives in it

Every conversation is a session of the store, a line is a slot (the dashboard says LINES), and the SSD tier is the engine's
"disk tier" (`--disk-gib`).

| Holder | What it holds | Gives way |
|---|---|---|
| a running request's conversation | its list of chunks and its reservation | never |
| a running request's image rows | chunks, inside its reservation (204 rows a chunk) | never; freed when the request ends |
| a conversation on no running request | its pieces: cache entries and bookmarks | to the SSD tier or dropped, least recently used first |
| a ghost ([split.md](split.md)) | nothing: a free chunk whose bytes are still intact | taken like a free chunk, after the plain ones |
| a window past 1M | groups lent out | closed on demand, when no request needs it |
| the Engram row cache | groups lent out | by groups, down to its floor, last |
| the image encoder's work area | two groups lent out | never while the engine runs |

A running request is in flight and not paused between turns. Everything else in the store is cache, and it gives way in one
order: free chunks; stored conversations, least recently used first (the last piece first); conversations on idle or paused
lines (the line is detached, the conversation stored); and for a request's growth, a copy or a window, a window no request
needs. A consumer that is short only of what running requests hold waits ([README](../README.md#requests-that-wait-for-memory)).
By default nothing is freed on a timer; memory goes back when a consumer asks (`--window-idle-s` adds a timer for larger windows).

## Tenants

A **tenant** is a borrower that is not a request. The store lends it whole groups; it holds them as device or host memory and
gives them back. The start prints a line per tenant (`store tenant N, NAME (host memory|device memory, required|cache|on
demand): G groups of 178 MiB`).

| Tenant | Memory | Standing | Sized by |
|---|---|---|---|
| 0, windows | device | on demand | a request's factor: the 2M and 4M windows' graphs and score buffers, opened when first needed |
| 1, Engram cache | host | cache | `--engram-cache-mib`, between `--engram-cache-min-mib` and `--engram-cache-max-mib`; live from the dashboard |
| 3, work area | device | required | the image encoder's buffers, two groups, taken once at start |
| 4, prefill rows | device | on demand | `--prefill-rows-max`: a second complete set of the prefill's row buffers for 16,384 rows, 23 groups (about 4,065 MiB a Spark), taken once after start |

The prefill rows tenant is the first to give way. It is asked for only when the store's top 23 groups (the groups at the end of the store, which a take would have to empty) hold nothing in use and the
free groups are at least the set's 23 plus `--prefill-loan-high-groups` (8), so 8 stay free after the take; it never evicts a
conversation. When the free groups fall under `--prefill-loan-low-groups` (2) it goes back at the first moment no prompt is processed on
it, before any cached conversation is evicted, and is asked for again only at the set plus the high mark. A failed allocation on
any Spark is undone on every Spark and not retried for 300 s. The set's rules are in the README's
[Long prompts processed 16,384 rows at a time](../README.md#long-prompts-processed-16384-rows-at-a-time).

Tenant ids 2 and 5 to 7 are unused; 8 to 15 are reserved for models.

A hand-over is a collective every rank joins; it runs at a run boundary, between decode runs.
A take that cannot be satisfied does one of three things: `no room` (the store could not still hold one conversation of
`--max-ctx`), `busy` (running requests hold what the cut takes; nothing leaves the store, and the conversations evicted on the
way stay evicted) or `failed` (the driver's accounting did not match, so the groups are back or owed). After a hand-over every
rank proves that memory really moved: the pool's handles and the box's available memory are read before and after, and the
change must match within the larger of a measured null and 256 MiB. A failed size is not tried again for five minutes unless
the setting changes. A run that has to pause for a tenant's change, a setting or owed groups says so in the log (`serve: the
run was paused for ...`), because every pause costs each decoder a whole step.

## The sizing rule

`--pool-free-gib G` is how much memory the engine leaves to the host when it starts: on every Spark, G GiB of device memory and of
available host memory, **with every line outside the store at its cap**. It is a reservation made once, at start; the store keeps its
size while the engine runs. The offer of a rank is

    (smaller of: free device memory less G, available host memory less G less the Engram cache still to grow)
      less the sum of the lines' room (each line's cap less what it holds at the read)

rounded down to whole groups; every rank takes the smallest offer. The lines:

Each line is held one of four ways: a **hard cap** (the line cannot pass it), a **refusal** (what would pass it is refused or
dropped), a **governor** (the process sheds load when its reading passes the cap, so the cap can be passed between readings) or an
**estimate** (a reservation sized from measurements, with nothing that stops the line passing it).

| Line | Cap | Setting | Held by |
|---|---|---|---|
| glibc heap | 1 GiB | | estimate: the engine caps glibc's arenas at 2 and trims after each checkpoint, which keeps the heap small but is not a byte limit |
| Engram cache still to grow | its size less what it holds | `--engram-cache-mib` | hard cap: the cache never holds more than its size; what it has still to grow is counted in the offer |
| front-end process (rank 0) | 840 MiB (the default); 0 refuses a `--pool-free-gib` start | `--front-mib` (engine and `dsfront serve`) | governor: the front-end's own memory governor, read every 10 s; `dsfront serve --front-mib 0` is not watched |
| image process outside the engine (rank 0) | 512 MiB (media worker) or 1,260 MiB (encoder container) | `--media-mib`, `--encoder-mib` | hard cap: its container's memory limit |
| front-end output queue (rank 0) | 64 MiB a connection | | refusal: the connection is dropped past it |
| request frames (rank 0) | twice `--frame-stage-mib` (400) | `--frame-stage-mib` | refusal: a larger frame is answered `too-large` and skipped unread |
| server token ids (rank 0) | 1,024 MiB | `--ids-host-mib` | governor: idle conversations' ids go to files past it |
| state gather buffer | computed | | |
| the kernel's own growth | 0.75 GiB | `--pool-slack-gib` | not enforced; shown as observed plus margin |
| the split cache's buffers | computed | | counted |

### What stays free

With the store full and every line outside it at its cap, every Spark keeps G GiB free. Rank 0 usually has the least room,
because the front-end, the media worker, the token ids and the request frames all live there, so its lines are the ones that bind.
G is a floor: an idle engine has more free than G, because the lines outside the store are not at their caps then.

The Engram cache's memory is part of the memory read at start until its groups are lent, and then it is the store's tenant; the
offer subtracts only the part still to grow.

### The start table and the refusals

Rank 0 prints, after sizing, a table of the lines outside the store (each with its cap and what it holds at the read; a line that
differs by more than 64 MiB between ranks is marked), the store's size, and what each Spark keeps free with the store full and every
line at its cap. The start is refused, with the reason in the log, when

- `--pool-free-gib` is given without `--front-mib` (give the engine `--front-mib` and `dsfront serve` the same value),
- one of the engine's own lines already holds more than its cap at the read, or
- the store is smaller than one conversation of `--max-ctx` plus the Engram cache's groups, one largest image request's rows
  and the largest window a request may open.

A rank projected to have under 2 GiB available once the store and the lines are full gets a warning line. A start by
`--pool-tokens` alone leaves nothing aside by this rule and needs no front-end cap.

The start lines to look for (rank 0; the numbers are placeholders, each start prints its own):

```
chunk pool: N numbers of 2.78 MiB (X GiB, Y positions) on every rank (--pool-free-gib G: N on this rank; free device / available host memory before it, ranks 0-3: ...; the Engram cache's 0.0 to grow)
store citizens: Engram cache E groups (64 numbers each); windows W groups at most; KV and image rows the rest
store tenant 1, Engram cache (host memory, cache): E groups of 178 MiB (M MiB held by its body), 0 in flight
rank 0: ready after N s; free device memory X GiB, available host memory Y GiB, resident Z GiB
```

## The memory watch

While serving, every 10 seconds each Spark reads its available memory. Rank 0 prints `memory: rank R has X GiB available, under
the G GiB left to the host; lines over their cap: ...` at each new low under the setting, and nothing while memory stays above it;
STATS carries the smallest reading (`mem_avail_min_bytes`) and the count of new lows (`mem_avail_low_count`), and the dashboard's
NODE HEALTH shows each Spark's available memory. The reading goes beside the device's work: inside a decoding run between two
steps (one bit of the step's control word), and while a long prompt is processed, on the chunk's own command; a command of its own
is issued only when neither is running. A run is never ended for the watch.

The watch only reports: nothing gives memory back or takes more while serving. The sizing at start and the lines' caps are what
keep G free.

## Settings

| Setting | Default | What it does |
|---|---|---|
| `--pool-free-gib G` | unset | the reservation above. G is what the box keeps for the operating system, the page cache and anything else not under the engine's account; the engine's own lines are already inside the sum, so G need not allow for them. A larger G gives a smaller store |
| `--pool-tokens N` | unset | a store of N positions (N x 5 / 16,384 chunks) |
| `--pool-pick smaller\|larger` | smaller | which one decides when both are given |
| `--pool-slack-gib G` | 0.75 | the kernel-drift line |
| `--front-mib N` | no default: 0 refuses a `--pool-free-gib` start | the front-end process's cap (give `dsfront serve` the same `--front-mib`; `dsfront serve --front-mib 0` is not watched) |
| `--media-mib N` / `--encoder-mib N` | 512 / 1,260 | the image process's cap |
| `--frame-stage-mib N` | 400 | the largest request frame |
| `--ids-host-mib N` | 1,024 | token ids kept in memory before idle conversations' ids go to files |
| `--grow-ahead N` | 131,072 | the response horizon a request reserves |
| `--snap-every N` | 1,024 | positions between window bookmarks |
| `--engram-cache-mib`, `-max-mib`, `-min-mib` | 4,096; 16,384; the start size | the Engram tenant |
| `--window-idle-s S` | 0 | close a larger window after S seconds with no request of its factor (0: only when the store needs it) |
