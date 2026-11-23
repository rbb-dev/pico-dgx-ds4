// Front-end <-> rank-0 protocol over a Unix stream socket: frame layouts (see docs/protocol.md).
// Every frame is a FrameHeader followed by `size` payload bytes: a fixed struct, then arrays.
// Little-endian, no padding between arrays. Plain C++, no CUDA.
#pragma once
#include <cstddef>
#include <cstdint>

namespace fe {

constexpr uint32_t kVersion = 4;
// The version of a front-end that sends kFlagPrefix and reads DoneTail. The engine accepts both versions on a connection (HelloFe.version);
// a connection of kVersion sees only the frames without SubmitPrefix and DoneTail (server/frames.h).
constexpr uint32_t kVersionPrefix = kVersion + 1;
constexpr const char* kDefaultSocket = "/tmp/dsgb10-engine.sock";

enum MsgType : uint16_t {
  // front-end -> rank 0
  kHello = 0x01,    // HelloFe
  kSubmit = 0x02,   // SubmitHeader, [SubmitPrefix with kFlagPrefix], uint32 stop_ids[n_stop_ids], uint32 prompt[n_prompt]
  kCancel = 0x03,   // Cancel
  kRelease = 0x04,  // Release
  kStatsReq = 0x05, // StatsReq: ask for one Stats frame (the dashboard is being watched)
  kConfig = 0x06,   // Config: interleaved prefill's settings, applied at once (the dashboard's live settings)
  // rank 0 -> front-end
  kHelloEngine = 0x81,  // HelloEngine, char name[name_len]
  kTokens = 0x82,       // TokensHeader, uint32 ids[n], [float logprob[n]], [TopLogprob top[n * n_top]]
  kDone = 0x83,         // DoneHeader, [DoneTail], char error[error_len]
  kProgress = 0x84,     // Progress: after each prefill chunk of a request
  kStats = 0x85,        // StatsFrame: the answer to a StatsReq
};

struct FrameHeader {
  uint32_t size;  // payload bytes after this header
  uint16_t type;  // MsgType
  uint16_t reserved;
};

struct HelloFe {
  uint32_t version;
  uint32_t reserved;
};

enum SubmitFlags : uint32_t {
  kFlagSeed = 1,      // use SubmitHeader::seed
  kFlagKeep = 2,      // keep the session's state after DONE
  kFlagLogprobs = 4,  // send logprobs in TOKENS
  // vision (docs/vision.md): after prompt[], uint32 n_images, ImageSpan spans[n_images], then the spans' rows,
  // bf16 [sum rows][5120] in span order. Only image requests set it; a text SUBMIT is unchanged.
  kFlagImages = 8,
  // ablated weights path: the request runs with the ablation direction applied. Sessions are kept per flag: the
  // front-end never continues or forks a session with a request of the other flag.
  kFlagAblit = 16,
};

// An image's span of the prompt: positions [start, start + rows), every id 129264 (<｜deepseek_image｜>);
// its rows (the encoder's: IMAGE_START, the aligner rows with IMAGE_NEW_LINE after each grid row,
// IMAGE_END) replace the token embeddings there, the span routes with ffn.gate.bias_vl and is dead to Engram.
struct ImageSpan {
  uint32_t start;
  uint32_t rows;
  uint64_t hash;  // of the image's bytes (logs)
};
constexpr int kImageDim = 5120;
constexpr uint32_t kMaxImages = 100;
constexpr uint32_t kMaxImageRows = 1100;
constexpr uint32_t kImageTokenId = 129264;

struct SubmitHeader {
  uint64_t req_id;           // unique per connection
  uint64_t session_id;       // state to continue or create; never 0
  uint64_t base_session_id;  // 0, or a resident session a new session may fork from (up to the LCP)
  uint64_t seed;
  float temperature;         // 0 = greedy
  float top_p;               // 1 = off
  uint32_t top_k;            // 0 = off
  uint32_t max_new_tokens;   // >= 1, already clamped to the context by the front-end
  uint32_t flags;            // SubmitFlags
  uint32_t n_top_logprobs;   // 0..20, with kFlagLogprobs
  uint32_t prefix_hint;      // prompt tokens the front-end expects to match the session (advisory)
  uint32_t n_stop_ids;       // ids that end generation (EOS is in here)
  uint32_t n_prompt;         // >= 1 (the suffix's count with kFlagPrefix: any, the built prompt needs one id)
  uint32_t reserved;
};

struct Cancel {
  uint64_t req_id;
};

struct Release {
  uint64_t session_id;
};

struct HelloEngine {
  uint32_t version;
  uint32_t max_context;   // prompt + output tokens per request
  uint32_t vocab_size;
  uint32_t eos_id;
  uint32_t max_sessions;  // resident sessions rank 0 can hold (advisory)
  uint32_t name_len;      // bytes of model/engine name that follow
  uint64_t keep_positions;  // positions of conversation the engine can keep (the pool and the disk tier): the
                            // front-end's session table keeps as much (its matches live as long as the engine's state)
};

enum TokensFlags : uint32_t {
  kTokLogprobs = 1,  // float logprob[n] follows the ids
  kTokTop = 2,       // TopLogprob top[n * n_top_logprobs] follows
};

struct TokensHeader {
  uint64_t req_id;
  uint32_t n;
  uint32_t flags;  // TokensFlags
};

struct TopLogprob {
  uint32_t id;
  float logprob;
};

enum DoneReason : uint32_t {
  kStop = 0,       // sampled a stop id (sent as the last token)
  kLength = 1,     // max_new_tokens reached
  kCancelled = 2,  // CANCEL
  kContext = 3,    // context window full
  kError = 4,      // rejected or failed; error text follows
  kTooLarge = 5,   // a frame larger than the engine reads (--frame-stage-mib): skipped, not read; error text follows (the limit)
  kPrefixGone = 6, // a kFlagPrefix request whose prompt could not be built (no such entry, another epoch, n_prefix past what the
                   // entry reported, ids unreadable): nothing was done, no DoneTail; the reason's text follows
};

struct DoneHeader {
  uint64_t req_id;
  uint32_t reason;             // DoneReason
  uint32_t prompt_tokens;
  uint32_t cached_tokens;      // prompt tokens whose state was reused
  uint32_t completion_tokens;  // ids sent in TOKENS for this request
  uint64_t queue_us;           // SUBMIT received -> prefill start
  uint64_t prefill_us;         // prefill start -> first token
  uint64_t decode_us;          // first token -> DONE
  uint32_t error_len;
  uint32_t reserved;
};

// On a connection of kVersionPrefix only: directly after the DoneHeader of every DONE of a request that ran (not of one cancelled in the queue, rejected, too large or
// prefix-gone): the history entry's epoch and the positions it reports. A front-end reads it when the frame holds it:
// size == sizeof(DoneHeader) + sizeof(DoneTail) + error_len (a DONE without one is sizeof(DoneHeader) + error_len). epoch 0 and
// n_hist 0: the session was not kept (no kFlagKeep, or released). The next request continues the entry with kFlagPrefix, epoch and
// n_prefix <= n_hist.
struct DoneTail {
  uint64_t epoch;
  uint32_t n_hist;
  uint32_t reserved;  // 0
};

// Prefill progress of one request, sent after each prefill chunk (and so also after the last one): the
// prompt tokens computed so far, of the tokens this request has to compute (the prompt without the
// positions reused from a session or the disk tier).
struct Progress {
  uint64_t req_id;
  uint32_t computed;
  uint32_t total;
};

struct StatsReq {
  uint32_t reserved0;  // 0
  uint32_t reserved1;  // 0
};

// Interleaved prefill's settings (the engine's --interleave-* flags), changed while it serves: version, the number of
// i64 fields that follow, then the fields in this order; a field < 0, or one not sent, keeps its value. The engine checks
// each against its flag's range and ignores one out of range; the next Stats frame shows what it runs.
constexpr uint32_t kConfigVersion = 3;
constexpr uint32_t kConfigFields = 15;
struct Config {
  uint32_t version;  // kConfigVersion
  uint32_t n_fields;
  int64_t on;            // 0 off, 1 on
  int64_t whole_rows;    // a prompt with at most this many rows to compute is read whole
  int64_t push_rows;     // a second reader's rows to compute, at most
  int64_t gap_ms;        // the longest a decoding session waits between steps
  int64_t share;         // % of the time decoders get while a prompt is read (10-90)
  int64_t burst_max_ms;  // one decode window, at most
  int64_t ssd_gib;       // the disk tier's size in GiB (at most its file's size at the start; smaller evicts to fit)
  int64_t keep_days;     // a conversation unused this many days is dropped (1-3650)
  int64_t extend;        // the context past 1,048,576 tokens: 0 off (the 1,048,576-token window only), 1 on (a factor per request)
  int64_t engram_mib;    // the Engram row cache's slots in MiB, the display reservation's included (0 off; applied at a run boundary)
  int64_t engram_mib_spark[4];  // reserved: the Engram cache's size of Spark 0-3 (-1 keeps it); not read by this engine
  int64_t test_tenant_groups;   // fault builds only (DSGB10_TEST_FAULTS): the groups of the dummy tenant 15; a normal build never reads it
};

// The engine's own counters for the dashboard: version, the number of u64 fields that follow, then the fields in this
// order. A front-end reads min(n_fields, kStatsFields) of them; a field it does not get is 0. The pool's counts are
// rank 0's (every rank's are equal); counters count from the engine's start.
constexpr uint32_t kStatsVersion = 2;
constexpr uint32_t kStatsFields = 100;
struct StatsFrame {
  uint32_t version;   // kStatsVersion
  uint32_t n_fields;  // kStatsFields
  // the chunk pool (docs/engine.md, "The chunk pool"): numbers, each one slice of every cache array on each rank
  uint64_t pool_numbers;            // numbers the pool holds
  uint64_t pool_free;               // of them no conversation names
  uint64_t pool_held;               // named by a conversation's caches or bookmarks
  uint64_t pool_shared;             // of those, named by more than one (shared beginnings)
  uint64_t pool_reserved;           // kept for the running requests' horizons, not yet held
  uint64_t number_bytes;            // one number's bytes on each rank
  uint64_t pool_positions;          // positions of conversation the whole pool holds (a conversation takes 5 numbers per 16,384 positions)
  // conversations
  uint64_t conv_on_lines;           // held by a line (decoding, or reading its prompt)
  uint64_t conv_in_memory;          // idle, their chunks in the pool
  uint64_t conv_on_disk;            // in the disk tier
  uint64_t conv_sharing;            // naming at least one shared number
  uint64_t lines;                   // sequences the engine decodes at once
  // the disk tier
  uint64_t disk_cap_bytes;          // 0: no disk tier
  uint64_t disk_used_bytes;
  uint64_t spills_total;            // session states written to disk
  uint64_t spill_bytes_total;
  uint64_t spills_pending;          // queued for writing, not yet committed
  uint64_t restores_total;          // states read back from disk
  uint64_t restore_bytes_total;
  uint64_t restored_tokens_total;   // prompt tokens restored from disk instead of computed
  uint64_t restore_wait_ns_total;   // time requests waited for restores
  uint64_t restore_wait_max_ns;     // the longest single wait
  // interleaved prefill: the settings in force (Config), and the long prompts being read and waiting
  uint64_t il_on;
  uint64_t il_whole_rows;
  uint64_t il_push_rows;
  uint64_t il_gap_ms;
  uint64_t il_share;
  uint64_t il_burst_max_ms;
  uint64_t il_reading;
  uint64_t il_queued;
  uint64_t keep_days;               // a conversation unused this many days is dropped (CONFIG keep_days)
  uint64_t extend;                  // the context past 1,048,576 tokens (CONFIG extend): 0 off, 1 on
  uint64_t ctx_limit;               // the longest context a request may use: the start window, 1,048,576 with extend 0
  uint64_t ranks;                   // the Sparks the engine runs on: each holds the same pool and the same disk tier
  uint64_t image_numbers;           // numbers the running requests' image rows hold now (each 204 rows; they are not in pool_held's conversations)
  // the Engram row cache, a borrower of the pool's memory (rank 0's; the cache's contents differ by rank, its size does not)
  uint64_t engram_cache_bytes;      // the cache's slots now, bytes (the display reservation's included)
  uint64_t engram_groups;           // groups of the pool its block holds
  uint64_t engram_pending_mib;      // the size a change not yet landed is going to (running, or waiting for memory or a back-off); 0: none
  uint64_t engram_lookups;          // rows looked up in the cache since the start
  uint64_t engram_hits;             //   and found
  uint64_t engram_lend_ok_total;    // changes of its size that landed
  uint64_t engram_lend_failed_total;  // that failed (the cache stayed at its old size)
  uint64_t lent_owed_groups;        // groups a borrower gave back that the pool could not make into numbers yet
  uint64_t engram_huge_pct;         // % of its block's resident bytes in huge pages at the last change's end
  uint64_t engram_evictions;        // rows the cache dropped to make room, since the start
  // the store's sizing rule and the memory watch (rank 0's; the store's size is every rank's)
  uint64_t store_numbers;           // numbers the store was sized to at the start, a whole number of groups of 64
  uint64_t mem_avail_min_bytes;     // the smallest of the four ranks' MemAvailable at the watch's last reading (0: no reading yet)
  uint64_t mem_avail_low_count;     // new lows under the memory --pool-free-gib leaves to the host, reported since the start
  uint64_t ids_host_bytes;          // the server's token ids held in memory now, bytes (the rest are in the ids files)
  uint64_t ids_file_sessions;       // conversations whose ids are in a file now
  uint64_t mem_avail_bytes[4];      // each rank's MemAvailable at the watch's last reading, rank order (0: none yet)
  // the pool's tenants (lend.h): the groups each holds, by id
  uint64_t tenant_groups[8];        // tenants 0-7: 0 the windows, 1 the Engram cache, 2 not assigned, 3 a work area, 4 the prefill's lent row buffers
  uint64_t tenant_models_groups;    // the groups of the models loaded while the engine runs (tenants 8-15), together
  // the cache conversation split (--kv-split cache; model/store_core.h)
  uint64_t kv_split;               // 1 when --kv-split cache is on
  uint64_t conv_split;             // conversations with a piece that is a quarter slot
  uint64_t quarter_slots;          // quarter slots in use
  uint64_t quarter_numbers;        // chunk numbers open as quarter numbers
  uint64_t ghost_numbers;          // free numbers that still hold a split piece's bytes (ghosts)
  uint64_t split_pieces_total;     // pieces split since the start
  uint64_t split_deferred_total;   // pieces a split left whole for want of a quarter number
  uint64_t revives_total;          // pieces a join took back from their ghosts (nothing crossed the ring)
  uint64_t joins_total;            // join commands that had a piece to bring back
  uint64_t join_bytes_total;       // bytes that came from the other three Sparks in joins
  uint64_t join_ns_total;          // time the engine thread spent in joins, ns
  uint64_t join_max_ns;            // the longest join command, ns
  uint64_t lend_busy_total;        // times a window or tenant change answered busy
  uint64_t join_short_total;       // joins that stopped for want of a number
  // the split cache's tiers (server/split_pace.h): L1 whole conversations, L2 quarter slots, L3 the SSD
  uint64_t split_level;            // the last pass's level: 0 none, 1 aging, 2 reclaim
  uint64_t split_pressure_pm;      // the pressure between the marks, thousandths
  uint64_t split_age_now_s;        // the idle age a conversation must have for a demotion now, seconds
  uint64_t split_demote_pps;       // pieces demoted a second over the last 10 s
  uint64_t l2_numbers;             // chunk numbers that hold quarter slots
  uint64_t l2_dropped_total;       // conversations given back from L2 to the SSD since the start
  // the imager (model/store_core.h)
  uint64_t clean_numbers;          // held numbers whose quarter image is on the SSD (an eviction gives them back writing nothing)
  uint64_t image_pending;          // image jobs queued, not yet settled
  uint64_t split_rides_total;      // rides of the image commands inside decoding runs
  uint64_t returns_read_total;     // pieces a return had to read from the SSD (the others were taken back from their ghosts)
  uint64_t image_waits_total;      // takes that waited for an image (a broken count: always 0)
  uint64_t split_reader_waits_total;  // at most: times a prompt was read alone while a request waited and images were pending (it waits for the reader's end; 0 is conclusive)
  // copy drafting's learner (DSGB10_COPY_ADAPT; rank 0's counts, every rank learns the same): proposals scored since the start and the
  // outcomes it holds, the first copy token right, copy rows verified and accepted, and the shared first-token chance by match
  // bucket (8-15, 16-31, 32-63, 64 and longer) in thousandths
  uint64_t copy_adapt;                // 1 when the gate learns, else 0 (and the others are 0)
  uint64_t copy_proposals_total;      // proposals the gate made (a proposal is scored a step later, unless that step is not the verify of it)
  uint64_t copy_scored_total;
  uint64_t copy_first_right_total;
  uint64_t copy_rows_verified_total;
  uint64_t copy_rows_accepted_total;
  uint64_t copy_chance_pm[4];
  uint64_t copy_failing_requests_total;  // requests that ended with 16 or more proposals scored and the first token right in under half of them
};

static_assert(sizeof(FrameHeader) == 8);
static_assert(sizeof(HelloFe) == 8);
static_assert(sizeof(SubmitHeader) == 72);
static_assert(offsetof(SubmitHeader, temperature) == 32);
static_assert(offsetof(SubmitHeader, n_prompt) == 64);
static_assert(sizeof(HelloEngine) == 32);
static_assert(offsetof(HelloEngine, keep_positions) == 24);
static_assert(sizeof(TokensHeader) == 16);
static_assert(sizeof(TopLogprob) == 8);
static_assert(sizeof(ImageSpan) == 16);
static_assert(sizeof(DoneHeader) == 56);
static_assert(sizeof(SubmitPrefix) == 16);
static_assert(sizeof(DoneTail) == 16);
static_assert(offsetof(DoneHeader, queue_us) == 24);
static_assert(sizeof(Progress) == 16);
static_assert(sizeof(StatsReq) == 8);
static_assert(sizeof(StatsFrame) == 8 + 8 * kStatsFields);
static_assert(offsetof(StatsFrame, pool_numbers) == 8);
static_assert(offsetof(StatsFrame, lines) == 8 + 8 * 11);
static_assert(offsetof(StatsFrame, keep_days) == 8 + 8 * 30);
static_assert(offsetof(StatsFrame, extend) == 8 + 8 * 31);
static_assert(offsetof(StatsFrame, ctx_limit) == 8 + 8 * 32);
static_assert(offsetof(StatsFrame, ranks) == 8 + 8 * 33);
static_assert(offsetof(StatsFrame, image_numbers) == 8 + 8 * 34);
static_assert(offsetof(StatsFrame, engram_cache_bytes) == 8 + 8 * 35);
static_assert(offsetof(StatsFrame, engram_huge_pct) == 8 + 8 * 43);
static_assert(offsetof(StatsFrame, engram_evictions) == 8 + 8 * 44);
static_assert(offsetof(StatsFrame, store_numbers) == 8 + 8 * 45);
static_assert(offsetof(StatsFrame, ids_file_sessions) == 8 + 8 * 49);
static_assert(offsetof(StatsFrame, mem_avail_bytes) == 8 + 8 * 50);
static_assert(offsetof(StatsFrame, mem_avail_bytes) + sizeof(StatsFrame::mem_avail_bytes) == offsetof(StatsFrame, tenant_groups));
static_assert(offsetof(StatsFrame, tenant_groups) == 8 + 8 * 54);
static_assert(offsetof(StatsFrame, tenant_models_groups) == 8 + 8 * 62);
static_assert(offsetof(StatsFrame, kv_split) == 8 + 8 * 63);
static_assert(offsetof(StatsFrame, copy_failing_requests_total) + sizeof(StatsFrame::copy_failing_requests_total) == sizeof(StatsFrame));
static_assert(sizeof(Config) == 8 + 8 * kConfigFields);

}  // namespace fe
