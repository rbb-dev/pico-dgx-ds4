//! The dashboard's data: what the front-end sees of the engine (requests, token and prefill counters, the STATS frame)
//! and what the samplers report, served under /dashboard (docs/dashboard.md). The token path only bumps atomics or
//! takes a lock for a few instructions; rates, series and JSON are made by the 1 Hz task or a handler.

use crate::protocol::{ConfigFrame, Done, DoneReason, EngineStats};
use crate::server::App;
use axum::extract::{ConnectInfo, State};
use axum::http::request::Parts;
use axum::http::{HeaderMap, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::body::Bytes;
use axum::routing::get;
use axum::Router;
use serde_json::{json, Map, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const PAGE: &str = include_str!("dashboard.html");

/// The rates are taken over this long.
const RATE_WINDOW: Duration = Duration::from_secs(3);
/// The series keep this many buckets of `BUCKET_S` seconds (15 minutes).
const BUCKETS: u64 = 90;
const BUCKET_S: u64 = 10;
/// The CACHE series keeps twice the other series' window: the chart's running lines look back a whole window from each point it draws.
const CACHE_BUCKETS: u64 = 2 * BUCKETS;
/// The finished requests the page's TTFT figures are taken from.
const RECENT: usize = 200;
/// The dashboard counts as watched for this long after a dashboard request; only then is the engine
/// asked for STATS and are the samplers polled.
const WATCH_S: u64 = 10;
/// While nobody watches, the engine is asked for STATS this often (seconds), so the CACHE chart has its history when a page opens.
const IDLE_STATS_S: u64 = 10;
const NODE_POLL: Duration = Duration::from_secs(2);
const NODE_TIMEOUT: Duration = Duration::from_millis(1500);
/// A sampler's answer is one JSON line; a longer one is refused.
const NODE_MAX_LINE: u64 = 1 << 20;

fn unix_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// What the page shows of a request beside its counters.
#[derive(Clone, Debug)]
pub struct ReqMeta {
    pub key_label: String,
    pub ablit: bool,
    pub prompt_tokens: u32,
    pub endpoint: &'static str,
    pub priority: &'static str,
}

/// One request in flight. The counters are bumped from the token path; nothing here is a lock.
pub struct ReqLive {
    pub id: u64,
    pub meta: ReqMeta,
    t_submit: Instant,
    generated: AtomicU32,
    /// prompt tokens computed so far and the tokens to compute (from PROGRESS; 0 until the first PROGRESS arrives)
    computed: AtomicU32,
    total: AtomicU32,
    progressed: AtomicBool,
    decoding: AtomicBool,
    /// set once the request has been counted into the CACHE series
    counted: AtomicBool,
}

impl ReqLive {
    fn new(id: u64, meta: ReqMeta) -> ReqLive {
        ReqLive {
            id,
            meta,
            t_submit: Instant::now(),
            generated: AtomicU32::new(0),
            computed: AtomicU32::new(0),
            total: AtomicU32::new(0),
            progressed: AtomicBool::new(false),
            decoding: AtomicBool::new(false),
            counted: AtomicBool::new(false),
        }
    }

    /// `decode` once a token has come, `prefill` once the engine has reported prefill progress, else
    /// `queued`.
    fn state(&self) -> &'static str {
        if self.decoding.load(Ordering::Relaxed) {
            "decode"
        } else if self.progressed.load(Ordering::Relaxed) {
            "prefill"
        } else {
            "queued"
        }
    }

    /// The prompt tokens reused from a session or the disk tier: the prompt less the tokens the engine reports it has to
    /// compute (PROGRESS `total`); the whole prompt when the request decodes without a PROGRESS (a request that reuses the
    /// whole prompt sends none); unknown while it is queued.
    fn cached_tokens(&self) -> Option<u32> {
        if self.progressed.load(Ordering::Relaxed) {
            Some(self.meta.prompt_tokens.saturating_sub(self.total.load(Ordering::Relaxed)))
        } else if self.decoding.load(Ordering::Relaxed) {
            Some(self.meta.prompt_tokens)
        } else {
            None
        }
    }

    /// The KV positions the request holds as far as the front-end can tell: its prompt and output while
    /// it decodes, the prompt tokens computed so far while it prefills, none while it is queued.
    fn kv_tokens(&self) -> u64 {
        match self.state() {
            "decode" => self.meta.prompt_tokens as u64 + self.generated.load(Ordering::Relaxed) as u64,
            "prefill" => self.computed.load(Ordering::Relaxed) as u64,
            _ => 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Recent {
    pub ttft_s: f64,
    pub prompt_tokens: u32,
    pub cached_tokens: u32,
    pub ablit: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct Bucket {
    idx: u64,
    sum: f64,
    n: u32,
    peak: f64,
}

/// Fixed buckets of `BUCKET_S` seconds with their mean and peak, `BUCKETS` of them.
#[derive(Default)]
pub struct Buckets {
    b: VecDeque<Bucket>,
}

/// A series' answer: the mean and the peak of each point (None: no reading), and the end of the last
/// point in Unix milliseconds.
#[derive(Debug, PartialEq)]
pub struct SeriesAnswer {
    pub points: Vec<Option<f64>>,
    pub peaks: Vec<Option<f64>>,
    pub last_ms: u64,
}

impl Buckets {
    /// A reading `v` at `t_ms`: into the bucket that holds the time; buckets older than the window go.
    pub fn add(&mut self, t_ms: u64, v: f64) {
        let idx = t_ms / (BUCKET_S * 1000);
        match self.b.back_mut() {
            Some(last) if last.idx == idx => {
                last.sum += v;
                last.n += 1;
                last.peak = last.peak.max(v);
            }
            Some(last) if last.idx > idx => return, // the clock went back: that reading is lost
            _ => self.b.push_back(Bucket { idx, sum: v, n: 1, peak: v }),
        }
        while self.b.front().is_some_and(|f| f.idx + BUCKETS <= idx) {
            self.b.pop_front();
        }
    }

    /// `seconds` of history as points of `step` seconds (multiples of the bucket, at most the whole
    /// window), the last point ending with the bucket that holds `now_ms`. A point with no reading in
    /// it is None.
    pub fn query(&self, now_ms: u64, seconds: u64, step: u64) -> SeriesAnswer {
        let k = (step.max(BUCKET_S) / BUCKET_S).min(BUCKETS);
        let n = ((seconds / BUCKET_S).clamp(1, BUCKETS) / k).max(1);
        let last = now_ms / (BUCKET_S * 1000);
        let (mut points, mut peaks) = (Vec::new(), Vec::new());
        for g in 0..n {
            let end = last as i64 - ((n - 1 - g) * k) as i64;
            let start = end - k as i64 + 1;
            let (mut sum, mut cnt, mut peak) = (0.0, 0u32, f64::MIN);
            for x in self.b.iter().filter(|x| (x.idx as i64) >= start && (x.idx as i64) <= end) {
                sum += x.sum;
                cnt += x.n;
                peak = peak.max(x.peak);
            }
            if cnt > 0 {
                points.push(Some(sum / cnt as f64));
                peaks.push(Some(peak));
            } else {
                points.push(None);
                peaks.push(None);
            }
        }
        SeriesAnswer { points, peaks, last_ms: (last + 1) * BUCKET_S * 1000 }
    }
}

/// Counter samples over the rate window: the rate is the change from the oldest sample inside it.
#[derive(Default)]
struct Window {
    s: VecDeque<(Instant, u64, u64)>,
}

impl Window {
    /// Add a reading of the two counters and return (decode, prefill) per second.
    fn push(&mut self, now: Instant, decode: u64, prefill: u64) -> (f64, f64) {
        self.s.push_back((now, decode, prefill));
        while self.s.len() > 2 && now.duration_since(self.s[1].0) >= RATE_WINDOW {
            self.s.pop_front();
        }
        let (t0, d0, p0) = self.s[0];
        let dt = now.duration_since(t0).as_secs_f64();
        if dt <= 0.0 {
            return (0.0, 0.0);
        }
        (decode.saturating_sub(d0) as f64 / dt, prefill.saturating_sub(p0) as f64 / dt)
    }
}

/// The latest answer of one rank's sampler.
struct NodeSample {
    at: Instant,
    at_ms: u64, // the reading's wall-clock time, the same on every status call: a page tells a new reading by it
    json: Value,
}

/// The Engram cache's lookups and hits as the engine's STATS counted them, sampled once a STATS: the hit rate is the change
/// over the last 60 seconds, not the whole run's.
#[derive(Default)]
pub struct HitWindow {
    samples: VecDeque<(Instant, u64, u64)>,
}

const HIT_WINDOW: Duration = Duration::from_secs(60);

impl HitWindow {
    pub fn push(&mut self, now: Instant, lookups: u64, hits: u64) {
        // counters that went back mean the engine restarted: start over
        if self.samples.back().is_some_and(|&(_, l, h)| lookups < l || hits < h) {
            self.samples.clear();
        }
        self.samples.push_back((now, lookups, hits));
        while self.samples.front().is_some_and(|&(t, _, _)| now.duration_since(t) > HIT_WINDOW) {
            self.samples.pop_front();
        }
    }

    /// hits / lookups between the oldest sample in the window and the newest; None when no row was looked up in between.
    pub fn rate(&self, now: Instant) -> Option<f64> {
        let &(_, l1, h1) = self.samples.back()?;
        let &(_, l0, h0) = self.samples.iter().find(|&&(t, _, _)| now.duration_since(t) <= HIT_WINDOW)?;
        let dl = l1 - l0;
        if dl == 0 {
            return None;
        }
        Some((h1 - h0) as f64 / dl as f64)
    }
}

pub struct Telemetry {
    started: Instant,
    connected_ms: AtomicU64,
    decode_ids: AtomicU64,
    prefill_tokens: AtomicU64,
    watched_ms: AtomicU64,
    live: Mutex<HashMap<u64, Arc<ReqLive>>>,
    recent: Mutex<VecDeque<Recent>>,
    window: Mutex<Window>,
    rates: Mutex<(f64, f64)>,
    decode_series: Mutex<Buckets>,
    prefill_series: Mutex<Buckets>,
    stats: Mutex<Option<(Instant, EngineStats)>>,
    engram_hits: Mutex<HitWindow>,
    cache: Mutex<CacheSeries>,
    cache_prompts: AtomicU64,
    cache_prompt_tokens: AtomicU64,
    cache_cached_tokens: AtomicU64,
    cache_seen: Mutex<[u64; 3]>,
    stats_asked_ms: AtomicU64,
    node_addrs: Mutex<Vec<String>>,
    nodes: Mutex<Vec<Option<NodeSample>>>,
}

impl Default for Telemetry {
    fn default() -> Self {
        Telemetry::new()
    }
}

impl Telemetry {
    pub fn new() -> Telemetry {
        Telemetry {
            started: Instant::now(),
            connected_ms: AtomicU64::new(0),
            decode_ids: AtomicU64::new(0),
            prefill_tokens: AtomicU64::new(0),
            watched_ms: AtomicU64::new(0),
            live: Mutex::new(HashMap::new()),
            recent: Mutex::new(VecDeque::new()),
            window: Mutex::new(Window::default()),
            rates: Mutex::new((0.0, 0.0)),
            decode_series: Mutex::new(Buckets::default()),
            prefill_series: Mutex::new(Buckets::default()),
            stats: Mutex::new(None),
            engram_hits: Mutex::new(HitWindow::default()),
            cache: Mutex::new(CacheSeries::default()),
            cache_prompts: AtomicU64::new(0),
            cache_prompt_tokens: AtomicU64::new(0),
            cache_cached_tokens: AtomicU64::new(0),
            cache_seen: Mutex::new([0; 3]),
            stats_asked_ms: AtomicU64::new(0),
            node_addrs: Mutex::new(Vec::new()),
            nodes: Mutex::new(Vec::new()),
        }
    }

    /// A request goes to the engine.
    pub fn register(&self, id: u64, meta: ReqMeta) -> Arc<ReqLive> {
        let live = Arc::new(ReqLive::new(id, meta));
        self.live.lock().unwrap().insert(id, live.clone());
        live
    }


    pub fn generated_tokens(&self) -> u64 {
        self.decode_ids.load(Ordering::Relaxed)
    }

    /// `n` ids of a request arrived. Atomics only.
    pub fn tokens(&self, live: &ReqLive, n: usize) {
        self.decode_ids.fetch_add(n as u64, Ordering::Relaxed);
        live.generated.fetch_add(n as u32, Ordering::Relaxed);
        live.decoding.store(true, Ordering::Relaxed);
        self.count_prompt(live);
    }

    /// The request's prompt goes into the CACHE series once, when its answer is first known: its first progress report (the prompt
    /// less the tokens to compute is cached) or, with none, its first output token (the whole prompt is cached). Atomics only.
    fn count_prompt(&self, live: &ReqLive) {
        if live.counted.swap(true, Ordering::Relaxed) {
            return;
        }
        let Some(cached) = live.cached_tokens() else { live.counted.store(false, Ordering::Relaxed); return };
        self.cache_prompts.fetch_add(1, Ordering::Relaxed);
        self.cache_prompt_tokens.fetch_add(live.meta.prompt_tokens as u64, Ordering::Relaxed);
        self.cache_cached_tokens.fetch_add(cached as u64, Ordering::Relaxed);
    }

    /// A prefill chunk of a request finished: `computed` of `total` prompt tokens. Atomics only.
    pub fn progress(&self, live: &ReqLive, computed: u32, total: u32) {
        let before = live.computed.swap(computed, Ordering::Relaxed);
        live.total.store(total, Ordering::Relaxed);
        live.progressed.store(true, Ordering::Relaxed);
        self.prefill_tokens.fetch_add(computed.saturating_sub(before) as u64, Ordering::Relaxed);
        self.count_prompt(live);
    }

    /// A request ended (an error, or a request the engine could not build from its prefix, is not a finished request for the TTFT figures).
    pub fn finish(&self, d: &Done) {
        let live = self.live.lock().unwrap().remove(&d.req_id);
        if matches!(d.reason, DoneReason::Error | DoneReason::TooLarge | DoneReason::PrefixGone) {
            return;
        }
        let ablit = live.is_some_and(|l| l.meta.ablit);
        let r = Recent { ttft_s: (d.queue_us + d.prefill_us) as f64 / 1e6, prompt_tokens: d.prompt_tokens, cached_tokens: d.cached_tokens, ablit };
        let mut q = self.recent.lock().unwrap();
        q.push_front(r);
        q.truncate(RECENT);
    }

    /// The connection to the engine dropped: nothing is in flight any more.
    pub fn clear_live(&self) {
        self.live.lock().unwrap().clear();
    }

    pub fn set_stats(&self, s: EngineStats) {
        self.set_stats_at(unix_ms(), s);
    }

    /// A STATS frame that arrived at `now_ms`.
    pub fn set_stats_at(&self, now_ms: u64, s: EngineStats) {
        let now = Instant::now();
        self.engram_hits.lock().unwrap().push(now, s.engram_lookups, s.engram_hits);
        {
            let mut c = self.cache.lock().unwrap();
            c.add_stats(now_ms, CacheCounters::of(&s));
            c.add_pool(now_ms, PoolFill::of(&s));
        }
        *self.stats.lock().unwrap() = Some((now, s));
    }

    /// The CACHE chart's points: `seconds` of history in points of `step` seconds.
    pub fn cache_series(&self, now_ms: u64, seconds: u64, step: u64) -> CacheAnswer {
        self.cache.lock().unwrap().query(now_ms, seconds, step)
    }

    /// The sum of the CACHE chart's buckets over the last `seconds`.
    pub fn cache_window(&self, now_ms: u64, seconds: u64) -> CacheBucket {
        self.cache.lock().unwrap().window(now_ms, seconds)
    }

    /// Whether the engine is asked for STATS at `now_ms`: every second while the dashboard is watched, else every `IDLE_STATS_S`
    /// seconds, so the CACHE chart has its history when a page opens.
    pub fn stats_due(&self, now_ms: u64) -> bool {
        let at = self.watched_ms.load(Ordering::Relaxed);
        let watched = at != 0 && now_ms.saturating_sub(at) < WATCH_S * 1000;
        let asked = self.stats_asked_ms.load(Ordering::Relaxed);
        let due = watched || asked == 0 || now_ms.saturating_sub(asked) >= IDLE_STATS_S * 1000;
        if due {
            self.stats_asked_ms.store(now_ms, Ordering::Relaxed);
        }
        due
    }

    /// The Engram cache's hit rate over the last minute of STATS (None: no row looked up in it).
    pub fn engram_hit_rate(&self) -> Option<f64> {
        self.engram_hits.lock().unwrap().rate(Instant::now())
    }

    pub fn stats(&self) -> Option<EngineStats> {
        self.stats.lock().unwrap().map(|(_, s)| s)
    }

    /// How long ago the last STATS arrived (None: none yet).
    pub fn stats_age(&self) -> Option<Duration> {
        self.stats.lock().unwrap().as_ref().map(|(t, _)| t.elapsed())
    }

    /// The front-end's connection to the engine came up (true) or went down (false): the page shows how long it has held.
    pub fn connected(&self, up: bool) {
        self.connected_ms.store(if up { unix_ms() } else { 0 }, Ordering::Relaxed);
    }

    /// Seconds since the front-end connected to the engine; None while it is not connected.
    pub fn connected_s(&self) -> Option<u64> {
        match self.connected_ms.load(Ordering::Relaxed) { 0 => None, at => Some(unix_ms().saturating_sub(at) / 1000) }
    }

    /// A dashboard request came in: the dashboard counts as watched for the next `WATCH_S` seconds.
    pub fn watch(&self) {
        self.watched_ms.store(unix_ms(), Ordering::Relaxed);
    }

    pub fn watched(&self) -> bool {
        let at = self.watched_ms.load(Ordering::Relaxed);
        at != 0 && unix_ms().saturating_sub(at) < WATCH_S * 1000
    }

    /// The 1 Hz step: the rates over the window, and a reading into each series.
    pub fn tick(&self, now: Instant, now_ms: u64) {
        let r = self.window.lock().unwrap().push(now, self.decode_ids.load(Ordering::Relaxed), self.prefill_tokens.load(Ordering::Relaxed));
        *self.rates.lock().unwrap() = r;
        self.decode_series.lock().unwrap().add(now_ms, r.0);
        self.prefill_series.lock().unwrap().add(now_ms, r.1);
        // every second the front-end runs is recorded in the CACHE series, so a quiet bucket is a zero bucket
        self.cache.lock().unwrap().touch(now_ms);
        // the requests counted since the last second, into the CACHE series
        let now = [self.cache_prompts.load(Ordering::Relaxed), self.cache_prompt_tokens.load(Ordering::Relaxed), self.cache_cached_tokens.load(Ordering::Relaxed)];
        let before = std::mem::replace(&mut *self.cache_seen.lock().unwrap(), now);
        if now != before {
            self.cache.lock().unwrap().add_prompts(now_ms, now[0] - before[0], now[1] - before[1], now[2] - before[2]);
        }
    }

    pub fn rates(&self) -> (f64, f64) {
        *self.rates.lock().unwrap()
    }

    pub fn series(&self, name: &str, now_ms: u64, seconds: u64, step: u64) -> Option<SeriesAnswer> {
        let b = match name {
            "decode_tps" => &self.decode_series,
            "prefill_tps" => &self.prefill_series,
            _ => return None,
        };
        Some(b.lock().unwrap().query(now_ms, seconds, step))
    }

    fn live_requests(&self) -> Vec<Arc<ReqLive>> {
        let mut v: Vec<_> = self.live.lock().unwrap().values().cloned().collect();
        v.sort_by_key(|l| l.t_submit);
        v
    }

    fn store_node(&self, i: usize, json: Value) {
        let mut n = self.nodes.lock().unwrap();
        if n.len() <= i {
            n.resize_with(i + 1, || None);
        }
        n[i] = Some(NodeSample { at: Instant::now(), at_ms: unix_ms(), json });
    }
}

// ------------------------------------------------------------------ the CACHE series

/// The STATS counters the cache series is made of: the Engram cache's lookups and hits, and the SSD read-backs (prompt tokens read
/// back instead of computed, the time requests waited, the pieces a return read from the SSDs and the pieces it took back from memory).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheCounters {
    pub lookups: u64,
    pub hits: u64,
    pub read_tokens: u64,
    pub wait_ns: u64,
    pub pieces_read: u64,
    pub revives: u64,
}

impl CacheCounters {
    pub fn of(s: &EngineStats) -> CacheCounters {
        CacheCounters { lookups: s.engram_lookups, hits: s.engram_hits, read_tokens: s.restored_tokens_total, wait_ns: s.restore_wait_ns_total,
                        pieces_read: s.returns_read_total, revives: s.revives_total }
    }

    /// True when any counter is lower than in `before`: the engine started again.
    fn went_back(&self, before: &CacheCounters) -> bool {
        self.lookups < before.lookups || self.hits < before.hits || self.read_tokens < before.read_tokens || self.wait_ns < before.wait_ns
            || self.pieces_read < before.pieces_read || self.revives < before.revives
    }
}

/// The context cache's pool as the last STATS of a bucket saw it: the chunks taken and all of them (taken and free), a Spark each, and
/// what one chunk is on all the engine's Sparks together. The CONTEXT CACHE tile's numbers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolFill {
    pub held: u64,
    pub total: u64,
    pub chunk_bytes: u64,
}

impl PoolFill {
    pub fn of(s: &EngineStats) -> Option<PoolFill> {
        let total = s.pool_held + s.pool_free;
        if total == 0 { None } else { Some(PoolFill { held: s.pool_held, total, chunk_bytes: s.number_bytes * s.ranks.max(1) }) }
    }
}

/// What one bucket of the CACHE chart holds: the prompts that arrived (their tokens, and how many of those the engine did not
/// compute), and the change in the engine's counters over the STATS samples taken in it. A bucket is also what a window adds up to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheBucket {
    idx: u64,
    pub prompts: u64,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub lookups: u64,
    pub hits: u64,
    pub read_tokens: u64,
    pub wait_ns: u64,
    pub pieces_read: u64,
    pub revives: u64,
    /// the tokens read back from the SSDs that belong to this bucket's prompts (at most the prompt tokens not computed): the read-back
    /// counted by the engine in this bucket or in a bucket just after or before it, matched by time
    pub ssd_used: u64,
    /// read-back counted in the three buckets before this one, for which no prompt turned up
    pub read_unmatched: u64,
    /// STATS samples whose change counted into the bucket; 0: the engine's counters were not read in it
    pub samples: u32,
    /// the engine's counters went back in the bucket (it started again): nothing of its counters is counted there
    pub restarted: bool,
    /// the context cache's pool at the bucket's last STATS (a level, not a sum); None: no STATS saw it
    pub pool: Option<PoolFill>,
}

/// Shares of all prompt tokens that arrived, in per cent: found in memory, read back from the SSDs, computed. They add up to 100.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CacheShares {
    pub memory: f64,
    pub ssd: f64,
    pub computed: f64,
}

impl CacheBucket {
    fn add(&mut self, o: &CacheBucket) {
        self.prompts += o.prompts;
        self.prompt_tokens += o.prompt_tokens;
        self.cached_tokens += o.cached_tokens;
        self.lookups += o.lookups;
        self.hits += o.hits;
        self.read_tokens += o.read_tokens;
        self.wait_ns += o.wait_ns;
        self.pieces_read += o.pieces_read;
        self.revives += o.revives;
        self.ssd_used += o.ssd_used;
        self.read_unmatched += o.read_unmatched;
        self.samples += o.samples;
        self.restarted |= o.restarted;
        if o.pool.is_some() { self.pool = o.pool; }
    }

    /// The prompt tokens the engine did not compute: at most the prompt's.
    fn not_computed(&self) -> u64 {
        self.cached_tokens.min(self.prompt_tokens)
    }

    /// None when no prompt arrived. Memory is what was not computed less what the SSDs gave back, so a share is never under 0
    /// or over 100.
    pub fn shares(&self) -> Option<CacheShares> {
        if self.prompt_tokens == 0 {
            return None;
        }
        let p = self.prompt_tokens as f64;
        let cached = self.not_computed();
        let ssd = self.ssd_used.min(cached);
        Some(CacheShares { memory: 100.0 * (cached - ssd) as f64 / p, ssd: 100.0 * ssd as f64 / p, computed: 100.0 * (self.prompt_tokens - cached) as f64 / p })
    }

    /// The Engram cache's hits of its lookups, in per cent; None when the bucket's samples saw no lookup.
    pub fn engram_pct(&self) -> Option<f64> {
        if self.samples == 0 || self.restarted || self.lookups == 0 { None } else { Some(100.0 * self.hits.min(self.lookups) as f64 / self.lookups as f64) }
    }
}

/// The CACHE chart's buckets: the same 90 of 10 seconds as the other series.
#[derive(Default)]
pub struct CacheSeries {
    b: VecDeque<CacheBucket>,
    prev: Option<CacheCounters>,
    /// read-back tokens counted by the engine that no prompt has taken yet, and the bucket that counted the oldest of them
    carry: u64,
    carry_idx: u64,
    /// the newest buckets with prompts, and with Engram lookups, that have left the 180: the running lines hold their value through
    /// any quiet time, so a window of traffic is still known after the traffic has left the series
    older_prompts: VecDeque<CacheBucket>,
    older_lookups: VecDeque<CacheBucket>,
}

/// The cache series' answer: a bucket a point (None: nothing in it), the end of the last point in Unix milliseconds, and the buckets
/// with traffic from before the first point (each with the end of its 10 seconds), up to a window of each kind of traffic.
#[derive(Debug, PartialEq)]
pub struct CacheAnswer {
    pub points: Vec<Option<CacheBucket>>,
    pub last_ms: u64,
    pub before: Vec<(u64, CacheBucket)>,
}

impl CacheSeries {
    /// The bucket that holds `t_ms`, made when it is not there; None when the time is older than the newest bucket.
    fn at(&mut self, t_ms: u64) -> Option<&mut CacheBucket> {
        let idx = t_ms / (BUCKET_S * 1000);
        match self.b.back() {
            Some(last) if last.idx > idx => return None,
            Some(last) if last.idx == idx => {}
            _ => self.b.push_back(CacheBucket { idx, ..Default::default() }),
        }
        while self.b.front().is_some_and(|f| f.idx + CACHE_BUCKETS <= idx) {
            if let Some(old) = self.b.pop_front() {
                for (keep, has) in [(&mut self.older_prompts, old.prompt_tokens > 0), (&mut self.older_lookups, old.lookups > 0)] {
                    if has {
                        keep.push_back(old);
                        while keep.len() as u64 > BUCKETS { keep.pop_front(); }
                    }
                }
            }
        }
        self.b.back_mut()
    }

    /// The front-end was running at `t_ms`: the bucket that holds it exists, so a quiet bucket is a zero bucket and only a time the
    /// front-end did not run has no bucket.
    pub fn touch(&mut self, t_ms: u64) {
        if let Some(idx) = self.at(t_ms).map(|b| b.idx) {
            self.settle(idx);
        }
    }

    /// Gives the read-back no prompt has taken to the prompts it belongs to. The engine counts a read-back when it reads and the front-end
    /// counts a prompt when its answer is first known, so the two land in the same bucket or in neighbours: the newest bucket `k` takes first,
    /// then the two before it, each up to its prompt tokens not computed. A read-back no prompt of `k - 2` to `k` took, once it is three
    /// buckets old, goes to `k`'s `read_unmatched` (a request that never reported). A bucket's prompt tokens never change after its 10 seconds.
    fn settle(&mut self, k: u64) {
        if self.carry == 0 {
            return;
        }
        if k >= self.carry_idx + 3 {
            if let Some(b) = self.b.iter_mut().rev().find(|b| b.idx == k) {
                b.read_unmatched += self.carry;
                self.carry = 0;
            }
            return;
        }
        for idx in [Some(k), k.checked_sub(1), k.checked_sub(2)].into_iter().flatten() {
            if let Some(b) = self.b.iter_mut().rev().find(|b| b.idx == idx) {
                let take = self.carry.min(b.not_computed().saturating_sub(b.ssd_used));
                b.ssd_used += take;
                self.carry -= take;
            }
        }
    }

    /// The context cache's pool as a STATS at `t_ms` saw it: the bucket keeps its last reading.
    pub fn add_pool(&mut self, t_ms: u64, pool: Option<PoolFill>) {
        if let (Some(b), Some(_)) = (self.at(t_ms), pool) { b.pool = pool; }
    }

    /// `prompts` requests arrived in the second at `t_ms`, with `prompt_tokens` tokens of which `cached_tokens` were not computed.
    pub fn add_prompts(&mut self, t_ms: u64, prompts: u64, prompt_tokens: u64, cached_tokens: u64) {
        let Some(b) = self.at(t_ms) else { return };
        b.prompts += prompts;
        b.prompt_tokens += prompt_tokens;
        b.cached_tokens += cached_tokens;
        let idx = b.idx;
        self.settle(idx);
    }

    /// A STATS sample at `t_ms`: its change from the sample before goes into the bucket. The first sample has no change; counters
    /// that went back mark the bucket as one where the engine started again and count nothing.
    pub fn add_stats(&mut self, t_ms: u64, c: CacheCounters) {
        let before = self.prev.replace(c);
        let Some(b) = self.at(t_ms) else { return };
        let idx = b.idx;
        let Some(before) = before else { return };
        if c.went_back(&before) {
            b.restarted = true;
            // a read-back still waiting for its prompt was counted by the engine that has gone
            self.carry = 0;
            return;
        }
        b.samples += 1;
        b.lookups += c.lookups - before.lookups;
        b.hits += c.hits - before.hits;
        b.read_tokens += c.read_tokens - before.read_tokens;
        b.wait_ns += c.wait_ns - before.wait_ns;
        b.pieces_read += c.pieces_read - before.pieces_read;
        b.revives += c.revives - before.revives;
        let read = c.read_tokens - before.read_tokens;
        if read > 0 {
            if self.carry == 0 {
                self.carry_idx = idx;
            }
            self.carry += read;
        }
        self.settle(idx);
    }

    /// `seconds` of history as points of `step` seconds (a multiple of the bucket, at most the whole window), the last point ending
    /// with the bucket that holds `now_ms`: the same grouping as `Buckets::query`. A point where nothing happened is None.
    pub fn query(&self, now_ms: u64, seconds: u64, step: u64) -> CacheAnswer {
        let k = (step.max(BUCKET_S) / BUCKET_S).min(CACHE_BUCKETS);
        let n = ((seconds / BUCKET_S).clamp(1, CACHE_BUCKETS) / k).max(1);
        let last = now_ms / (BUCKET_S * 1000);
        let mut points = Vec::new();
        for g in 0..n {
            let end = last as i64 - ((n - 1 - g) * k) as i64;
            let start = end - k as i64 + 1;
            let mut sum = CacheBucket::default();
            let mut any = false;
            for x in self.b.iter().filter(|x| (x.idx as i64) >= start && (x.idx as i64) <= end) {
                sum.add(x);
                any = true;
            }
            points.push(if any { Some(sum) } else { None });
        }
        // the buckets with traffic before the first point: the newest window of each kind, oldest first
        let first = last as i64 - ((n - 1) * k) as i64 - k as i64 + 1;
        let older = self.older_prompts.iter().chain(self.older_lookups.iter()).chain(self.b.iter());
        let mut prompts: Vec<&CacheBucket> = older.clone().filter(|x| (x.idx as i64) < first && x.prompt_tokens > 0).collect();
        let mut lookups: Vec<&CacheBucket> = older.filter(|x| (x.idx as i64) < first && x.lookups > 0).collect();
        for v in [&mut prompts, &mut lookups] {
            v.sort_by_key(|x| x.idx);
            v.dedup_by_key(|x| x.idx);
            let cut = v.len().saturating_sub(BUCKETS as usize);
            v.drain(..cut);
        }
        let mut before: Vec<&CacheBucket> = prompts.into_iter().chain(lookups).collect();
        before.sort_by_key(|x| x.idx);
        before.dedup_by_key(|x| x.idx);
        CacheAnswer { points, last_ms: (last + 1) * BUCKET_S * 1000, before: before.into_iter().map(|x| ((x.idx + 1) * BUCKET_S * 1000, *x)).collect() }
    }

    /// The sum of the buckets of the last `seconds`: what the CACHE HITS tile shows, from the same buckets as the chart.
    pub fn window(&self, now_ms: u64, seconds: u64) -> CacheBucket {
        let a = self.query(now_ms, seconds, seconds);
        a.points.into_iter().flatten().fold(CacheBucket::default(), |mut s, b| { s.add(&b); s })
    }
}

/// A bucket as the page's chart reads it: its raw sums. The prompt tokens split into memory, SSD and computed (they add up to
/// `prompt_tokens`; the bars' parts and the running lines' sums), Engram's rows looked up and found, and the read-back facts for the hover.
/// null: the front-end recorded nothing in it; a bucket with no prompt is an object with 0 tokens.
pub fn cache_point_json(b: &Option<CacheBucket>) -> Value {
    let Some(b) = b else { return Value::Null };
    let (memory, ssd, computed) = (b.not_computed() - b.ssd_used.min(b.not_computed()), b.ssd_used.min(b.not_computed()), b.prompt_tokens - b.not_computed());
    json!({"memory_tok": memory, "ssd_tok": ssd, "computed_tok": computed,
           "prompts": b.prompts, "prompt_tokens": b.prompt_tokens, "cached_tokens": b.cached_tokens,
           "lookups": b.lookups, "hits": b.hits, "read_tokens": b.read_tokens, "read_unmatched": b.read_unmatched,
           "wait_ns": b.wait_ns, "pieces_read": b.pieces_read, "revives": b.revives,
           "counters_read": b.samples > 0 && !b.restarted, "restarted": b.restarted,
           "pool": b.pool.map_or(Value::Null, |p| json!({"held": p.held, "total": p.total, "chunk_bytes": p.chunk_bytes}))})
}

/// The CACHE series' answer as the route sends it: a point a bucket (null: nothing in it), the step and the end of the last point.
pub fn cache_answer_json(a: &CacheAnswer, step_s: u64) -> Value {
    let before: Vec<Value> = a.before.iter().map(|(end_ms, b)| { let mut v = cache_point_json(&Some(*b)); v["end_ms"] = json!(end_ms); v }).collect();
    json!({"points": a.points.iter().map(cache_point_json).collect::<Vec<_>>(), "step_s": step_s, "last_ms": a.last_ms, "before": before})
}

/// The `cache` block of the status: the window's sums, and the prompts counted since the front-end started.
pub fn cache_json(t: &Telemetry, now_ms: u64) -> Value {
    let mut v = cache_window_json(&t.cache_window(now_ms, BUCKETS * BUCKET_S), BUCKETS * BUCKET_S);
    v["since_start"] = json!({"prompts": t.cache_prompts.load(Ordering::Relaxed), "prompt_tokens": t.cache_prompt_tokens.load(Ordering::Relaxed),
                              "cached_tokens": t.cache_cached_tokens.load(Ordering::Relaxed)});
    v
}

/// The window's sums as the CACHE HITS tile reads them: shares of all prompt tokens and the read-back facts. `shares` is null
/// when no prompt arrived in the window.
pub fn cache_window_json(w: &CacheBucket, window_s: u64) -> Value {
    let shares = w.shares().map_or(Value::Null, |s| json!({"memory": r1(s.memory), "ssd": r1(s.ssd), "computed": r1(s.computed), "hit": r1(r1(s.memory) + r1(s.ssd))}));
    json!({"window_s": window_s, "prompts": w.prompts, "prompt_tokens": w.prompt_tokens, "cached_tokens": w.cached_tokens,
           "read_tokens": w.read_tokens, "read_unmatched": w.read_unmatched, "pieces_read": w.pieces_read, "revives": w.revives, "wait_ns": w.wait_ns,
           "counters_read": w.samples > 0, "shares": shares})
}

// ------------------------------------------------------------------ JSON

fn r1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

pub fn requests_json(t: &Telemetry) -> Value {
    let now = Instant::now();
    let in_flight: Vec<Value> = t
        .live_requests()
        .iter()
        .map(|l| {
            json!({"id": l.id, "state": l.state(), "prompt_tokens": l.meta.prompt_tokens,
                   "generated_tokens": l.generated.load(Ordering::Relaxed), "key_label": l.meta.key_label,
                   "age_s": now.duration_since(l.t_submit).as_secs_f64(), "kv_tokens": l.kv_tokens(),
                   "computed": l.computed.load(Ordering::Relaxed), "to_compute": l.total.load(Ordering::Relaxed),
                   "cached_tokens": l.cached_tokens(),
                   "endpoint": l.meta.endpoint, "priority": l.meta.priority, "ablit": l.meta.ablit})
        })
        .collect();
    let recent: Vec<Value> = t
        .recent
        .lock()
        .unwrap()
        .iter()
        .map(|r| json!({"ttft_s": r.ttft_s, "prompt_tokens": r.prompt_tokens, "cached_tokens": r.cached_tokens, "ablit": r.ablit}))
        .collect();
    json!({"in_flight": in_flight, "recent": recent})
}

/// The engine's sessions as the page's card reads them: the chunk pool (numbers, free, held, shared, reserved, number bytes,
/// positions), the conversations (on lines, in memory, on the SSD, sharing a beginning) and the lines. The window is
/// STATS ctx_limit, else HELLO's max_context; ranks are the Sparks the engine runs on.
pub fn sessions_json(s: &EngineStats, usable_window: u32) -> Value {
    let window = if s.ctx_limit > 0 { s.ctx_limit } else { usable_window as u64 };
    json!({"lines": s.lines, "usable_window": window, "ranks": s.ranks,
           "pool": {"numbers": s.pool_numbers, "free": s.pool_free, "held": s.pool_held, "shared": s.pool_shared,
                    "reserved": s.pool_reserved, "number_bytes": s.number_bytes, "positions": s.pool_positions,
                    "image_numbers": s.image_numbers},
           "conversations": {"on_lines": s.conv_on_lines, "in_memory": s.conv_in_memory, "on_disk": s.conv_on_disk,
                             "sharing": s.conv_sharing, "split": s.conv_split},
           "split": {"on": s.kv_split != 0, "quarter_slots": s.quarter_slots, "quarter_numbers": s.quarter_numbers,
                     "ghost_numbers": s.ghost_numbers, "pieces_total": s.split_pieces_total, "deferred_total": s.split_deferred_total,
                     "revives_total": s.revives_total, "joins_total": s.joins_total, "join_short_total": s.join_short_total,
                     "join_bytes_total": s.join_bytes_total, "join_ns_total": s.join_ns_total, "join_max_ns": s.join_max_ns,
                     "lend_busy_total": s.lend_busy_total}})
}

/// The Engram cache's size in MiB, to the nearest: its rows (264 bytes) leave it a few bytes short of a whole MiB, and the sizes the
/// page offers are whole MiB. The tile and the settings list both read this.
pub fn engram_mib(s: &EngineStats) -> u64 { (s.engram_cache_bytes + (1 << 19)) >> 20 }

/// The Engram row cache as the page's tile reads it: its size, the groups of the pool it holds, the size a change under way goes to,
/// lookups and hits since the engine's start with the hit rate of the last minute (`rate`, null without lookups), evictions, size changes
/// landed and failed, the groups owed back to the pool and the huge-page share. Null for an engine that sends none of it.
pub fn engram_json(s: &EngineStats, rate: Option<f64>) -> Value {
    if s.engram_cache_bytes == 0 && s.engram_groups == 0 && s.engram_pending_mib == 0 && s.engram_lookups == 0
        && s.engram_lend_ok_total == 0 && s.engram_lend_failed_total == 0 {
        return Value::Null;
    }
    json!({"cache_bytes": s.engram_cache_bytes, "size_mib": engram_mib(s), "groups": s.engram_groups,
           "pending_mib": s.engram_pending_mib, "lookups": s.engram_lookups, "hits": s.engram_hits, "hit_rate_1m": rate,
           "evictions": s.engram_evictions, "lend_ok": s.engram_lend_ok_total, "lend_failed": s.engram_lend_failed_total,
           "owed_groups": s.lent_owed_groups, "huge_pct": s.engram_huge_pct})
}

/// The numbers one group of the pool holds (the engine's `kGroupNums`): a tenant's groups times this times a number's bytes
/// is its memory on one Spark.
pub const GROUP_NUMBERS: u64 = 64;

/// The store's memory as its readers take it (the soak's watcher, a STORE card): the store's numbers, the server's ids held on the
/// host, every rank's free memory at the engine's last watch, the groups owed back to the pool and each tenant's groups. Null for an
/// engine that sends none of it.
pub fn memory_json(s: &EngineStats) -> Value {
    if s.store_numbers == 0 && s.mem_avail_bytes == [0; 4] {
        return Value::Null;
    }
    json!({"store_numbers": s.store_numbers, "ids_host_bytes": s.ids_host_bytes, "ids_file_sessions": s.ids_file_sessions,
           "engram_cache_bytes": s.engram_cache_bytes, "mem_avail_bytes": s.mem_avail_bytes.to_vec(),
           "owed_groups": s.lent_owed_groups, "tenant_groups": s.tenant_groups.to_vec(), "tenant_models_groups": s.tenant_models_groups,
           "number_bytes": s.number_bytes, "group_numbers": GROUP_NUMBERS, "ranks": s.ranks})
}

/// The sizes the SSD CACHE SIZE list offers, in GiB: 128 to 1024 in steps of 128 (the engine takes any size up to the file it made).
pub const SSD_SIZES_GIB: [u64; 8] = [128, 256, 384, 512, 640, 768, 896, 1024];

/// The SSD tier as the page's card reads it: `spill` (its state) and `prompt_cache` (its counters).
pub fn tier_json(s: Option<&EngineStats>) -> (Value, Value) {
    let Some(s) = s else { return (json!({"state": "unknown"}), Value::Null) };
    let state = if s.disk_cap_bytes > 0 { "on" } else { "off" };
    let cache = json!({
        "spill": [{"stored_bytes": s.disk_used_bytes, "cap_bytes": s.disk_cap_bytes, "points": s.conv_on_disk,
                   "writes_committed": s.spills_total, "written_bytes": s.spill_bytes_total, "writes_pending": s.spills_pending}],
        "restores_begun": s.restores_total, "disk_hits": s.restores_total, "restored_tokens": s.restored_tokens_total,
        "restore_waits": s.restores_total, "restore_wait_ns": s.restore_wait_ns_total, "restore_wait_max_ns": s.restore_wait_max_ns});
    (json!({"state": state}), cache)
}

/// The ranks' samples as the page reads them, each with its age, and the zone names of the first sample
/// that has them.
pub fn node_health_json(t: &Telemetry) -> Value {
    let addrs = t.node_addrs.lock().unwrap().len();
    let nodes = t.nodes.lock().unwrap();
    let now = Instant::now();
    let mut zone_names = Value::Array(Vec::new());
    let mut ranks = Vec::new();
    for i in 0..addrs {
        let mut o = match nodes.get(i).and_then(|n| n.as_ref()) {
            Some(n) => {
                let mut o = n.json.as_object().cloned().unwrap_or_default();
                o.insert("age_ms".into(), json!(now.duration_since(n.at).as_millis() as u64));
                o.insert("at_ms".into(), json!(n.at_ms));
                if zone_names.as_array().is_some_and(|z| z.is_empty()) {
                    if let Some(z) = o.get("zone_names").filter(|z| z.as_array().is_some_and(|a| !a.is_empty())) {
                        zone_names = z.clone();
                    }
                }
                o
            }
            None => {
                let mut o = Map::new();
                o.insert("age_ms".into(), Value::Null);
                o.insert("at_ms".into(), Value::Null);
                o
            }
        };
        o.insert("rank".into(), json!(i));
        ranks.push(Value::Object(o));
    }
    json!({"ranks": ranks, "zone_names": zone_names})
}

/// Each rank's device memory by use, from the samplers' `memory` blocks.
pub fn ledger_json(t: &Telemetry) -> Value {
    let nodes = t.nodes.lock().unwrap();
    let (mut ranks, mut entries) = (Vec::new(), Vec::new());
    for (i, n) in nodes.iter().enumerate() {
        let Some(m) = n.as_ref().and_then(|n| n.json.get("memory")) else { continue };
        let g = |k: &str| m.get(k).and_then(Value::as_u64);
        let Some(total) = g("device_bytes") else { continue };
        ranks.push(json!({"rank": i, "device_bytes": total}));
        for (key, id) in [("weights_bytes", "weights"), ("kv_bytes", "store"), ("work_bytes", "work")] {
            if let Some(b) = g(key) {
                entries.push(json!({"rank": i, "feature_id": id, "device_bytes": b}));
            }
        }
    }
    json!({"ranks": ranks, "entries": entries})
}

pub fn status_json(app: &App) -> Value {
    let t = app.engine.telemetry();
    let info = app.engine.info();
    let reqs = t.live_requests();
    let waiting = reqs.iter().filter(|l| l.state() != "decode").count();
    let (dec, pre) = t.rates();
    let now = Instant::now();
    // the oldest request that is not decoding and has reported progress
    let pass = reqs.iter().find(|l| l.state() == "prefill").map(|l| {
        json!({"computed": l.computed.load(Ordering::Relaxed), "total": l.total.load(Ordering::Relaxed),
               "seconds": now.duration_since(l.t_submit).as_secs_f64()})
    });
    let stats = t.stats();
    let (spill, cache) = tier_json(stats.as_ref());
    let mut health = json!({
        "max_batch_size": info.as_ref().map(|i| i.max_sessions),
        "live": {"connections": reqs.len(), "waiting": waiting, "decode_tokens_per_second": r1(dec),
                 "prefill_tokens_per_second": r1(pre), "prompt_pass": pass, "generated_tokens": t.generated_tokens()},
        "spill": spill});
    if !cache.is_null() {
        health["prompt_cache"] = cache;
    }
    let kv = match (&stats, &info) {
        (Some(s), Some(i)) => sessions_json(s, i.max_context),
        _ => Value::Null,
    };
    json!({"at_ms": unix_ms(), "model_name": app.cfg.model_name,
           "ablit": {"alias": app.ablit_name()},
           "engine": {"connected": info.is_some(), "name": info.as_ref().map(|i| i.name.clone()), "uptime_s": t.started.elapsed().as_secs(), "connected_s": t.connected_s()},
           "health": health, "sessions": kv, "node_health": node_health_json(t),
           "engram": stats.as_ref().map_or(Value::Null, |s| engram_json(s, t.engram_hit_rate())),
           "memory": stats.as_ref().map_or(Value::Null, memory_json),
           "cache": cache_json(t, unix_ms()),
           "interleave": settings_json(stats.as_ref())})
}

/// Interleaved prefill's settings as the engine runs them (its last STATS), and the long prompts being processed and waiting
pub fn settings_json(s: Option<&EngineStats>) -> Value {
    match s {
        Some(s) if s.il_share > 0 => json!({"on": s.il_on == 1, "whole_rows": s.il_whole_rows, "push_rows": s.il_push_rows,
                                            "gap_ms": s.il_gap_ms, "share": s.il_share, "burst_max_ms": s.il_burst_max_ms,
                                            "reading": s.il_reading, "queued": s.il_queued,
                                            "ssd_gib": s.disk_cap_bytes >> 30, "ssd_choices": SSD_SIZES_GIB, "keep_days": s.keep_days,
                                            "extend": s.extend == 1, "engram_mib": engram_mib(s)}),
        _ => Value::Null,
    }
}

/// A settings change from the dashboard: a JSON object of the fields to change, each checked against the range the
/// engine accepts for it; anything else is refused, so the engine never sees a value it would ignore.
pub fn parse_settings(v: &Value, max_ctx: i64) -> Result<ConfigFrame, String> {
    let Some(o) = v.as_object() else { return Err("a JSON object of settings".into()) };
    let mut c = ConfigFrame::default();
    for (k, x) in o {
        let int = |lo: i64, hi: i64| -> Result<i64, String> {
            let n = x.as_i64().ok_or_else(|| format!("{k}: a whole number"))?;
            if (lo..=hi).contains(&n) { Ok(n) } else { Err(format!("{k}: {n} is outside {lo}..{hi}")) }
        };
        match k.as_str() {
            "on" => c.on = x.as_bool().ok_or("on: true or false")? as i64,
            "share" => c.share = int(10, 90)?,
            "gap_ms" => c.gap_ms = int(100, 60000)?,
            "burst_max_ms" => c.burst_max_ms = int(50, 60000)?,
            "whole_rows" => c.whole_rows = int(8192, max_ctx)?,
            "push_rows" => c.push_rows = int(0, max_ctx)?,
            "ssd_gib" => c.ssd_gib = int(1, 131071)?,
            "keep_days" => c.keep_days = int(1, 3650)?,
            "extend" => c.extend = x.as_bool().ok_or("extend: true or false")? as i64,
            "engram_mib" => c.engram_mib = int(0, 131072)?,
            _ => return Err(format!("{k}: not a setting (on, share, gap_ms, burst_max_ms, whole_rows, push_rows, ssd_gib, keep_days, extend, engram_mib)")),
        }
    }
    if c == ConfigFrame::default() {
        return Err("no setting to change".into());
    }
    Ok(c)
}

async fn settings_get(State(app): State<Arc<App>>, headers: HeaderMap) -> Response {
    if let Err(r) = authed(&app, &headers) {
        return *r;
    }
    let t = app.engine.telemetry();
    t.watch();
    json_response(StatusCode::OK, settings_json(t.stats().as_ref()))
}

async fn settings_post(State(app): State<Arc<App>>, headers: HeaderMap, body: Bytes) -> Response {
    if let Err(r) = authed(&app, &headers) {
        return *r;
    }
    if let Err(r) = json_body(&headers) {
        return *r;
    }
    let max_ctx = app.engine.info().map_or(1 << 20, |i| i.max_context as i64);
    let v: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return failure(StatusCode::BAD_REQUEST, &format!("settings: {e}")),
    };
    match parse_settings(&v, max_ctx) {
        Ok(c) => {
            app.engine.configure(c);
            app.engine.request_stats();
            json_response(StatusCode::OK, json!({"sent": v}))
        }
        Err(e) => failure(StatusCode::BAD_REQUEST, &e),
    }
}

/// The fans of the Sparks (src/fans.rs): without `--fan-tool` the box is absent and the answer says so.
async fn fans_get(State(app): State<Arc<App>>, headers: HeaderMap) -> Response {
    if let Err(r) = authed(&app, &headers) {
        return *r;
    }
    app.engine.telemetry().watch();
    if !app.fans.present() {
        return json_response(StatusCode::OK, json!({"at_ms": unix_ms(), "tool": false, "profiles": [], "wanted": null, "mixed": false, "pinned": null, "nodes": []}));
    }
    match app.fans.status().await {
        Ok(v) => json_response(StatusCode::OK, v),
        Err(e) => failure(StatusCode::BAD_GATEWAY, &format!("fans: {e}")),
    }
}

/// `{"profile": "<id>"}` (a profile of the fixed list) or `{"gpu_clock": "<MHz>"}` (one of four), set on all four Sparks by the tool.
/// 409 while the fan tool reports the fans and clock as pinned.
async fn fans_post(State(app): State<Arc<App>>, parts: Parts, body: Bytes) -> Response {
    if let Err(r) = authed(&app, &parts.headers) {
        return *r;
    }
    if let Err(r) = json_body(&parts.headers) {
        return *r;
    }
    if !app.fans.present() {
        return failure(StatusCode::SERVICE_UNAVAILABLE, "no fan tool: start dsfront with --fan-tool");
    }
    let v: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return failure(StatusCode::BAD_REQUEST, &format!("fans: {e}")),
    };
    let change = match crate::fans::parse_fan_request(&v) {
        Ok(c) => c,
        Err(e) => return failure(StatusCode::BAD_REQUEST, &e),
    };
    let what = match change {
        crate::fans::FanChange::Profile(p) => format!("profile={p}"),
        crate::fans::FanChange::GpuClock(c) => format!("gpu_clock={c}"),
    };
    let peer = parts.extensions.get::<ConnectInfo<std::net::SocketAddr>>().map_or("unknown".to_string(), |c| c.0.ip().to_string());
    match app.fans.set(change).await {
        crate::fans::SetOutcome::Done(v) => {
            let n = v["nodes"].as_array().map_or(0, |a| a.len());
            let ok = v["nodes"].as_array().map_or(0, |a| a.iter().filter(|x| x["ok"] == json!(true)).count());
            eprintln!("fan: {what} ok={ok}/{n} peer={peer}");
            json_response(StatusCode::OK, v)
        }
        crate::fans::SetOutcome::Pinned(why) => {
            eprintln!("fan: {what} refused, pinned ({why}) peer={peer}");
            failure(StatusCode::CONFLICT, &format!("the fans and the clock are pinned: {why}"))
        }
        crate::fans::SetOutcome::Failed(e) => {
            eprintln!("fan: {what} failed ({e}) peer={peer}");
            failure(StatusCode::BAD_GATEWAY, &format!("fans: {e}"))
        }
    }
}

// ------------------------------------------------------------------ routes

pub fn routes() -> Router<Arc<App>> {
    Router::new()
        .route("/dashboard", get(page))
        .route("/dashboard/", get(page))
        .route("/dashboard/api/v1/status", get(status))
        .route("/dashboard/api/v1/requests", get(requests))
        .route("/dashboard/api/v1/series", get(series))
        .route("/dashboard/api/v1/ledger", get(ledger))
        .route("/dashboard/api/v1/settings", get(settings_get).post(settings_post))
        .route("/dashboard/api/v1/fans", get(fans_get).post(fans_post))
        .layer(axum::middleware::map_response(framing_headers))
}

/// No page of another site may frame the dashboard: every answer of its routes says so. No script policy is set: the page's script is inline.
async fn framing_headers(mut r: Response) -> Response {
    let h = r.headers_mut();
    h.insert("x-frame-options", axum::http::HeaderValue::from_static("DENY"));
    h.insert("content-security-policy", axum::http::HeaderValue::from_static("frame-ancestors 'none'"));
    r
}

/// A request's host name (the Host header without its port), lower case.
fn host_of(v: &str) -> String {
    let v = v.trim();
    let h = if let Some(rest) = v.strip_prefix('[') { rest.split(']').next().unwrap_or("") } else { v.rsplit_once(':').map_or(v, |(h, p)| if p.chars().all(|c| c.is_ascii_digit()) { h } else { v }) };
    h.to_ascii_lowercase()
}

/// The dashboard answers only requests that name this server (Host one of `dashboard_hosts`), so a name rebound to this
/// address by a page elsewhere gets 403. The page itself carries no data; every data route and the settings need the
/// API key (`authed`: the page asks for it once and keeps it in the browser until its user signs out).
fn guard(app: &App, headers: &HeaderMap) -> Result<(), Box<Response>> {
    let host = headers.get("host").and_then(|v| v.to_str().ok()).unwrap_or("");
    if !app.cfg.dashboard_hosts.iter().any(|h| *h == host_of(host)) {
        return Err(Box::new(failure(StatusCode::FORBIDDEN, "this host name is not one the dashboard answers (--dashboard-host)")));
    }
    Ok(())
}

/// The Host check, then the API key (the same check as the API's)
fn authed(app: &App, headers: &HeaderMap) -> Result<(), Box<Response>> {
    guard(app, headers)?;
    crate::server::check_auth(app, headers)
}

/// A POST's body is JSON sent as `Content-Type: application/json` (else 415). Without a CORS preflight, which this server
/// never answers, a page on another site can make a browser send only a form or plain text here, so a dashboard run
/// without an API key takes no change from another site.
fn json_body(headers: &HeaderMap) -> Result<(), Box<Response>> {
    let ct = headers.get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("");
    if ct.split(';').next().is_some_and(|t| t.trim().eq_ignore_ascii_case("application/json")) {
        Ok(())
    } else {
        Err(Box::new(failure(StatusCode::UNSUPPORTED_MEDIA_TYPE, "send the body as Content-Type: application/json")))
    }
}

fn json_response(status: StatusCode, v: Value) -> Response {
    (status, [("content-type", "application/json"), ("cache-control", "no-store")], v.to_string()).into_response()
}

fn failure(status: StatusCode, msg: &str) -> Response {
    json_response(status, json!({"error": {"message": msg, "code": status.as_u16()}}))
}

async fn page(State(app): State<Arc<App>>, headers: HeaderMap) -> Response {
    if let Err(r) = guard(&app, &headers) {
        return *r;
    }
    app.engine.telemetry().watch();
    (StatusCode::OK, [("content-type", "text/html; charset=utf-8"), ("cache-control", "no-store"), ("x-content-type-options", "nosniff")], PAGE).into_response()
}

async fn status(State(app): State<Arc<App>>, headers: HeaderMap) -> Response {
    if let Err(r) = authed(&app, &headers) {
        return *r;
    }
    app.engine.telemetry().watch();
    json_response(StatusCode::OK, status_json(&app))
}

async fn requests(State(app): State<Arc<App>>, headers: HeaderMap) -> Response {
    if let Err(r) = authed(&app, &headers) {
        return *r;
    }
    let t = app.engine.telemetry();
    t.watch();
    json_response(StatusCode::OK, requests_json(t))
}

async fn ledger(State(app): State<Arc<App>>, headers: HeaderMap) -> Response {
    if let Err(r) = authed(&app, &headers) {
        return *r;
    }
    let t = app.engine.telemetry();
    t.watch();
    json_response(StatusCode::OK, ledger_json(t))
}

/// `name=...&seconds=900&step=10` (no percent-decoding: the values are words and numbers).
fn query_param<'a>(uri: &'a Uri, key: &str) -> Option<&'a str> {
    uri.query()?.split('&').find_map(|kv| kv.split_once('=').filter(|(k, _)| *k == key).map(|(_, v)| v))
}

async fn series(State(app): State<Arc<App>>, headers: HeaderMap, uri: Uri) -> Response {
    if let Err(r) = authed(&app, &headers) {
        return *r;
    }
    let t = app.engine.telemetry();
    t.watch();
    let num = |k: &str, default: u64| -> Result<u64, ()> { query_param(&uri, k).map_or(Ok(default), |v| v.parse().map_err(|_| ())) };
    let (Ok(seconds), Ok(step)) = (num("seconds", 900), num("step", BUCKET_S)) else {
        return failure(StatusCode::BAD_REQUEST, "seconds and step must be whole numbers");
    };
    let name = query_param(&uri, "name").unwrap_or("decode_tps");
    if name == "cache" {
        let step_s = (step.max(BUCKET_S) / BUCKET_S).min(BUCKETS) * BUCKET_S;
        return json_response(StatusCode::OK, cache_answer_json(&t.cache_series(unix_ms(), seconds, step), step_s));
    }
    match t.series(name, unix_ms(), seconds, step) {
        Some(a) => {
            let step_s = (step.max(BUCKET_S) / BUCKET_S).min(BUCKETS) * BUCKET_S;
            json_response(StatusCode::OK, json!({"points": a.points, "peaks": a.peaks, "step_s": step_s, "last_ms": a.last_ms}))
        }
        None => failure(StatusCode::BAD_REQUEST, "name is decode_tps, prefill_tps or cache"),
    }
}

// ------------------------------------------------------------------ background

/// The 1 Hz task (counters into rates and series; a STATS request to the engine while the dashboard is
/// watched) and the sampler poller (every 2 s while watched, one connection per rank).
pub fn spawn(app: Arc<App>, node_addrs: Vec<String>) {
    let t = app.engine.telemetry().clone();
    *t.node_addrs.lock().unwrap() = node_addrs.clone();
    {
        let (t, engine) = (t.clone(), app.engine.clone());
        tokio::spawn(async move {
            let mut every = tokio::time::interval(Duration::from_secs(1));
            every.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                every.tick().await;
                t.tick(Instant::now(), unix_ms());
                if t.stats_due(unix_ms()) {
                    engine.request_stats();
                }
            }
        });
    }
    if node_addrs.is_empty() {
        return;
    }
    tokio::spawn(async move {
        let mut every = tokio::time::interval(NODE_POLL);
        every.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            every.tick().await;
            if !t.watched() {
                continue;
            }
            let polls: Vec<_> = node_addrs
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    let (t, a) = (t.clone(), a.clone());
                    tokio::spawn(async move {
                        if let Ok(Ok(v)) = tokio::time::timeout(NODE_TIMEOUT, sample(&a)).await {
                            t.store_node(i, v);
                        }
                    })
                })
                .collect();
            for p in polls {
                let _ = p.await;
            }
        }
    });
}

/// One sampler's answer: connect, send `GET\n`, read the one JSON line.
pub async fn sample(addr: &str) -> Result<Value, String> {
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
    let mut s = tokio::net::TcpStream::connect(addr).await.map_err(|e| e.to_string())?;
    s.write_all(b"GET\n").await.map_err(|e| e.to_string())?;
    let mut line = String::new();
    BufReader::new(s.take(NODE_MAX_LINE)).read_line(&mut line).await.map_err(|e| e.to_string())?;
    let v: Value = serde_json::from_str(line.trim()).map_err(|e| e.to_string())?;
    if v.is_object() {
        Ok(v)
    } else {
        Err("not a JSON object".into())
    }
}

// ------------------------------------------------------------------ mock sampler

/// `dsfront mock-node-stats`: a rank's sampler stand-in for tests. One JSON line to every connection that
/// sends `GET\n`, numbers that move with the time.
pub async fn mock_node_stats(port: u16, rank: u32) -> std::io::Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    eprintln!("mock node sampler: rank {rank} on 127.0.0.1:{port}");
    let t0 = Instant::now();
    loop {
        let (s, _) = listener.accept().await?;
        let line = mock_node_json(rank, t0.elapsed().as_secs_f64()).to_string();
        tokio::spawn(async move {
            let (rd, mut wr) = s.into_split();
            let mut cmd = String::new();
            if tokio::time::timeout(Duration::from_secs(2), BufReader::new(rd).read_line(&mut cmd)).await.is_ok() && cmd.trim() == "GET" {
                let _ = wr.write_all(format!("{line}\n").as_bytes()).await;
            }
        });
    }
}

/// A plausible sample of rank `rank` at `t` seconds.
pub fn mock_node_json(rank: u32, t: f64) -> Value {
    let w = (t / 7.0 + rank as f64).sin();
    let gib = 1u64 << 30;
    json!({"rank": rank, "mem_available_bytes": (12.0 + w) * gib as f64, "mem_total_bytes": 121.6 * gib as f64,
           "zones_c": [60.0 + 3.0 * w, 58.5 + rank as f64], "zone_names": ["cpu-thermal", "soc-thermal"],
           "gpu": {"temp_c": 60.0 + 2.0 * w, "util_pct": 35.0 + 30.0 * w, "clock_mhz": 2190, "clock_max_mhz": 3003,
                   "reasons": [], "power_w": 16.0 + w},
           "load1": 0.3 + 0.1 * w.abs(), "cpus": 20, "cpu_busy_pct": 3.0 + w.abs(),
           "nvme": {"temp_c": 53.0 + rank as f64, "wear_pct": 0, "root_free_bytes": 3.0e12, "root_total_bytes": 3.9e12,
                    "available_spare_pct": 100, "spare_threshold_pct": 5, "media_errors": 0, "critical_warning": 0,
                    "data_units_written": 1_000_000, "data_unit_bytes": 512000, "power_on_hours": 300, "unsafe_shutdowns": 2},
           "rails": [{"rx_bytes": (t * 2.0e8) as u64, "tx_bytes": (t * 2.1e8) as u64, "errors": 0, "discards": 0},
                     {"rx_bytes": (t * 1.9e8) as u64, "tx_bytes": (t * 2.0e8) as u64, "errors": 0, "discards": 0}],
           "memory": {"device_bytes": (121.6 * gib as f64) as u64, "weights_bytes": (74.2 * gib as f64) as u64,
                      "work_bytes": (18.1 * gib as f64) as u64, "kv_bytes": (15.3 * gib as f64) as u64}})
}

#[cfg(test)]
mod settings_tests {
    use super::*;

    #[test]
    fn settings_are_checked_against_the_engine_flags() {
        let c = parse_settings(&json!({"share": 40, "gap_ms": 600}), 1 << 20).unwrap();
        assert_eq!((c.share, c.gap_ms, c.on, c.whole_rows), (40, 600, -1, -1), "only the given fields change");
        assert_eq!(parse_settings(&json!({"on": false}), 1 << 20).unwrap().on, 0);
        assert_eq!(parse_settings(&json!({"push_rows": 131072, "whole_rows": 32768}), 1 << 20).unwrap().push_rows, 131072);
        let c = parse_settings(&json!({"ssd_gib": 128, "keep_days": 14}), 1 << 20).unwrap();
        assert_eq!((c.ssd_gib, c.keep_days, c.share), (128, 14, -1), "the SSD tier's size and expiry");
        let c = parse_settings(&json!({"extend": true}), 1 << 20).unwrap();
        assert_eq!((c.extend, c.share), (1, -1), "extending past 1M on");
        assert_eq!(parse_settings(&json!({"extend": false}), 1 << 20).unwrap().extend, 0);
        for bad in [json!({"extend": "on"}), json!({"extend": 1}), json!({"ssd_gib": 0}), json!({"keep_days": 0}), json!({"keep_days": 4000}), json!({"ssd_gib": "big"}),
                    json!({"share": 95}), json!({"share": 5}), json!({"gap_ms": 50}), json!({"whole_rows": 4096}),
                    json!({"whole_rows": 2_000_000}), json!({"share": "40"}), json!({"on": 1}), json!({"speed": 1}),
                    json!({}), json!([1])] {
            assert!(parse_settings(&bad, 1 << 20).is_err(), "{bad} must be refused");
        }
    }

    #[test]
    fn settings_json_reads_the_stats() {
        assert_eq!(settings_json(None), Value::Null);
        assert_eq!(settings_json(Some(&EngineStats::default())), Value::Null, "an engine without interleave");
        let s = EngineStats { il_on: 1, il_share: 40, il_gap_ms: 600, il_reading: 1, il_queued: 2, disk_cap_bytes: 128 << 30,
                              keep_days: 30, ..Default::default() };
        let v = settings_json(Some(&s));
        assert_eq!((v["on"].as_bool(), v["share"].as_u64(), v["queued"].as_u64()), (Some(true), Some(40), Some(2)));
        assert_eq!((v["ssd_gib"].as_u64(), v["keep_days"].as_u64()), (Some(128), Some(30)));
        assert_eq!(v["extend"], false, "off in these stats");
        let v = settings_json(Some(&EngineStats { il_share: 40, extend: 1, ..Default::default() }));
        assert_eq!(v["extend"], true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: u64 = 1000;

    #[test]
    fn host_names_lose_their_port() {
        assert_eq!(host_of("192.0.2.10:8000"), "192.0.2.10");
        assert_eq!(host_of("LocalHost"), "localhost");
        assert_eq!(host_of("[::1]:8000"), "::1");
        assert_eq!(host_of("api.example.com"), "api.example.com");
    }

    #[test]
    fn buckets_keep_mean_and_peak_of_ten_seconds() {
        let mut b = Buckets::default();
        // bucket 100 (t = 1000..1010 s): 2, 4, 9; bucket 101: 6
        for (t, v) in [(1000, 2.0), (1003, 4.0), (1009, 9.0), (1010, 6.0)] {
            b.add(t * S, v);
        }
        let a = b.query(1011 * S, 20, 10);
        assert_eq!(a.points, vec![Some(5.0), Some(6.0)]);
        assert_eq!(a.peaks, vec![Some(9.0), Some(6.0)]);
        assert_eq!(a.last_ms, 1020 * S, "the last point ends with its bucket");
    }

    #[test]
    fn series_have_gaps_windows_and_steps() {
        let mut b = Buckets::default();
        b.add(1000 * S, 10.0); // bucket 100
        b.add(1030 * S, 20.0); // bucket 103
        b.add(1031 * S, 40.0);
        // five points ending with bucket 104: 100, 101, 102, 103, 104
        let a = b.query(1045 * S, 50, 10);
        assert_eq!(a.points, vec![Some(10.0), None, None, Some(30.0), None]);
        assert_eq!(a.peaks, vec![Some(10.0), None, None, Some(40.0), None]);
        // a 20 s step groups buckets (103, 104) and (101, 102) and (99, 100): the mean of the readings in them
        let a = b.query(1045 * S, 60, 20);
        assert_eq!(a.points, vec![Some(10.0), None, Some(30.0)]);
        // the window is 90 buckets: older readings are gone, and a longer ask is cut to it
        b.add(1930 * S, 1.0); // bucket 193: bucket 100 is 93 back
        let a = b.query(1935 * S, 5000, 10);
        assert_eq!(a.points.len(), 90);
        assert_eq!(a.points.iter().flatten().count(), 1);
        assert_eq!(a.points[89], Some(1.0));
        assert!(b.b.iter().all(|x| x.idx + 90 > 193));
        // a point with no reading is None, a zero reading is 0
        let mut z = Buckets::default();
        z.add(5 * S, 0.0);
        assert_eq!(z.query(5 * S, 10, 10).points, vec![Some(0.0)]);
    }

    #[test]
    fn rates_are_taken_over_three_seconds() {
        let mut w = Window::default();
        let t0 = Instant::now();
        let at = |s: f64| t0 + Duration::from_secs_f64(s);
        assert_eq!(w.push(at(0.0), 0, 0), (0.0, 0.0));
        let (d, p) = w.push(at(1.0), 100, 4096);
        assert_eq!((d, p), (100.0, 4096.0));
        w.push(at(2.0), 200, 4096);
        w.push(at(3.0), 300, 4096);
        // at 4 s the window starts at 1 s: (400 - 100) / 3
        let (d, p) = w.push(at(4.0), 400, 4096);
        assert!((d - 100.0).abs() < 1e-9, "{d}");
        assert_eq!(p, 0.0, "the prefill burst left the window");
        // idle: no change, so zero
        for s in 5..9 {
            w.push(at(s as f64), 400, 4096);
        }
        assert_eq!(w.push(at(9.0), 400, 4096), (0.0, 0.0));
    }

    #[test]
    fn the_store_memory_reaches_the_json() {
        let s = EngineStats { store_numbers: 6912, ids_host_bytes: 5 << 20, mem_avail_bytes: [11 << 30, 15 << 30, 15 << 30, 16 << 30],
                              lent_owed_groups: 1, tenant_groups: [0, 0, 0, 2, 0, 0, 0, 0], tenant_models_groups: 3, ..Default::default() };
        let m = memory_json(&s);
        assert_eq!((m["store_numbers"].as_u64(), m["ids_host_bytes"].as_u64(), m["owed_groups"].as_u64()), (Some(6912), Some(5 << 20), Some(1)));
        assert_eq!((m["mem_avail_bytes"][0].as_u64(), m["mem_avail_bytes"][3].as_u64()), (Some(11 << 30), Some(16 << 30)));
        assert_eq!((m["tenant_groups"][3].as_u64(), m["tenant_models_groups"].as_u64()), (Some(2), Some(3)));
        // the fields the STORE card's page reads by name
        for k in ["store_numbers", "ids_host_bytes", "ids_file_sessions", "engram_cache_bytes", "mem_avail_bytes", "tenant_groups",
                  "tenant_models_groups", "number_bytes", "group_numbers", "ranks"] {
            assert!(m.get(k).is_some(), "the STORE card reads {k}");
        }
        assert_eq!(m["group_numbers"].as_u64(), Some(GROUP_NUMBERS));
        assert_eq!(memory_json(&EngineStats::default()), Value::Null, "an engine that sends none of it: null");
    }

    #[test]
    fn the_engram_cache_reaches_the_json() {
        let s = EngineStats { pool_numbers: 8913, number_bytes: 2_916_352, image_numbers: 6, lines: 32, ranks: 4, engram_cache_bytes: 2032 << 20,
                              engram_groups: 14, engram_pending_mib: 4096, engram_lookups: 1000, engram_hits: 900, engram_lend_ok_total: 3,
                              engram_lend_failed_total: 1, lent_owed_groups: 2, engram_huge_pct: 97, engram_evictions: 55, ..Default::default() };
        assert_eq!(sessions_json(&s, 5)["pool"]["image_numbers"].as_u64(), Some(6), "the images' numbers beside the pool's");
        let e = engram_json(&s, Some(0.9));
        assert_eq!((e["cache_bytes"].as_u64(), e["size_mib"].as_u64(), e["groups"].as_u64(), e["pending_mib"].as_u64()),
                   (Some(2032 << 20), Some(2032), Some(14), Some(4096)));
        assert_eq!((e["lookups"].as_u64(), e["hits"].as_u64(), e["evictions"].as_u64()), (Some(1000), Some(900), Some(55)));
        assert_eq!((e["lend_ok"].as_u64(), e["lend_failed"].as_u64(), e["owed_groups"].as_u64(), e["huge_pct"].as_u64()),
                   (Some(3), Some(1), Some(2), Some(97)));
        assert_eq!(e["hit_rate_1m"].as_f64(), Some(0.9));
        assert_eq!(engram_json(&s, None)["hit_rate_1m"], Value::Null, "no lookups in the last minute: no rate");
        // an engine that sends none of it (34 fields): the block is null, so the page hides the tile
        assert_eq!(engram_json(&EngineStats { engram_lookups: 0, ..EngineStats::default() }, None), Value::Null);
        // the setting: the cache's size now, in MiB; and a change is checked and sent
        let v = settings_json(Some(&EngineStats { il_share: 40, engram_cache_bytes: 4096 << 20, ..Default::default() }));
        assert_eq!(v["engram_mib"].as_u64(), Some(4096));
        for ok in [0i64, 2032, 4096, 6912] {
            assert_eq!(parse_settings(&json!({"engram_mib": ok}), 1 << 20).unwrap().engram_mib, ok);
        }
        let c = parse_settings(&json!({"engram_mib": 6912}), 1 << 20).unwrap();
        assert_eq!((c.engram_mib, c.share, c.extend), (6912, -1, -1), "only the given field changes");
        for bad in [json!({"engram_mib": -1}), json!({"engram_mib": "4096"}), json!({"engram_mib": 1e9 as i64}), json!({"engram_mib": true})] {
            assert!(parse_settings(&bad, 1 << 20).is_err(), "{bad} must be refused");
        }
    }

    #[test]
    fn the_hit_rate_is_over_the_last_minute() {
        let mut w = HitWindow::default();
        let t0 = Instant::now();
        let at = |s: u64| t0 + Duration::from_secs(s);
        assert_eq!(w.rate(at(0)), None, "no sample");
        w.push(at(0), 1000, 500);
        assert_eq!(w.rate(at(0)), None, "one sample has no change");
        w.push(at(10), 1000, 500);
        assert_eq!(w.rate(at(10)), None, "no lookups since: no rate, not 0");
        w.push(at(20), 1100, 590);
        assert_eq!(w.rate(at(20)), Some(0.9), "90 of 100 rows since the first sample");
        // the old rows leave the window: at 100 s only the samples from 40 s on count
        w.push(at(40), 1200, 590);
        w.push(at(100), 1400, 790);
        assert_eq!(w.rate(at(100)), Some(1.0), "200 of 200 rows since the sample 60 s back");
        // the engine restarted: the counters began again
        w.push(at(101), 50, 5);
        assert_eq!(w.rate(at(101)), None, "counters went back: start over");
        w.push(at(102), 150, 105);
        assert_eq!(w.rate(at(102)), Some(1.0));
    }

    #[test]
    fn the_hit_rate_is_kept_by_the_telemetry() {
        let t = Telemetry::new();
        assert_eq!(t.engram_hit_rate(), None);
        t.set_stats(EngineStats { engram_lookups: 100, engram_hits: 50, ..Default::default() });
        t.set_stats(EngineStats { engram_lookups: 200, engram_hits: 100, ..Default::default() });
        assert_eq!(t.engram_hit_rate(), Some(0.5));
    }

    fn stats() -> EngineStats {
        EngineStats {
            pool_numbers: 8913,
            number_bytes: 2_916_352,
            pool_positions: 8913 * 16384 / 5,
            disk_cap_bytes: 256 << 30,
            disk_used_bytes: 5 << 30,
            conv_on_disk: 41,
            spills_total: 400,
            spill_bytes_total: 9 << 30,
            spills_pending: 2,
            restores_total: 10,
            restored_tokens_total: 123456,
            restore_wait_ns_total: 50_000_000,
            restore_wait_max_ns: 9_000_000,
            lines: 32,
            ..Default::default()
        }
    }

    #[test]
    fn stats_map_to_the_sessions_and_the_ssd_tier() {
        let s = EngineStats { pool_free: 8000, pool_held: 913, pool_shared: 40, pool_reserved: 120, conv_on_lines: 3, conv_in_memory: 7,
                              conv_sharing: 2, ..stats() };
        let ss = sessions_json(&s, 1_048_576);
        assert_eq!((ss["lines"].as_u64(), ss["usable_window"].as_u64()), (Some(32), Some(1_048_576)));
        // the window is the engine's live limit when it sends one (it follows extend), HELLO's otherwise; the ranks are its
        let live = sessions_json(&EngineStats { ctx_limit: 1_048_576, ranks: 4, ..s }, 4_194_304);
        assert_eq!((live["usable_window"].as_u64(), live["ranks"].as_u64()), (Some(1_048_576), Some(4)));
        assert_eq!(sessions_json(&EngineStats { ctx_limit: 0, ..s }, 4_194_304)["usable_window"].as_u64(), Some(4_194_304));
        let p = &ss["pool"];
        assert_eq!((p["numbers"].as_u64(), p["free"].as_u64(), p["held"].as_u64()), (Some(8913), Some(8000), Some(913)));
        assert_eq!((p["shared"].as_u64(), p["reserved"].as_u64(), p["number_bytes"].as_u64()), (Some(40), Some(120), Some(2_916_352)));
        assert_eq!(p["positions"].as_u64(), Some(29_206_118), "8,913 numbers hold 29.2M positions (5 numbers each 16,384)");
        let c = &ss["conversations"];
        assert_eq!((c["on_lines"].as_u64(), c["in_memory"].as_u64(), c["on_disk"].as_u64(), c["sharing"].as_u64()),
                   (Some(3), Some(7), Some(41), Some(2)));
        // the split (--kv-split cache): off by default, its counts when on
        assert_eq!(ss["split"]["on"].as_bool(), Some(false));
        let sp = sessions_json(&EngineStats { kv_split: 1, conv_split: 5, quarter_slots: 40, ghost_numbers: 9, joins_total: 2, ..s }, 1_048_576);
        assert_eq!((sp["split"]["on"].as_bool(), sp["conversations"]["split"].as_u64(), sp["split"]["quarter_slots"].as_u64()),
                   (Some(true), Some(5), Some(40)));
        assert_eq!((sp["split"]["ghost_numbers"].as_u64(), sp["split"]["joins_total"].as_u64()), (Some(9), Some(2)));
        // nothing of the fixed slots: no slots x window sum, no store of its own memory
        for gone in ["slots", "slots_big", "ctx_big", "positions", "store_cap_bytes", "store_used_bytes"] {
            assert!(ss.get(gone).is_none(), "{gone}");
        }
        let (spill, cache) = tier_json(Some(&s));
        assert_eq!(spill["state"], "on");
        let t = &cache["spill"][0];
        assert_eq!((t["stored_bytes"].as_u64(), t["cap_bytes"].as_u64(), t["points"].as_u64()), (Some(5 << 30), Some(256 << 30), Some(41)));
        assert_eq!((t["writes_committed"].as_u64(), t["written_bytes"].as_u64(), t["writes_pending"].as_u64()), (Some(400), Some(9 << 30), Some(2)));
        assert_eq!((cache["restores_begun"].as_u64(), cache["restored_tokens"].as_u64()), (Some(10), Some(123456)));
        assert_eq!((cache["restore_wait_ns"].as_u64(), cache["restore_wait_max_ns"].as_u64()), (Some(50_000_000), Some(9_000_000)));
        // no disk tier, and no STATS yet
        assert_eq!(tier_json(Some(&EngineStats { disk_cap_bytes: 0, ..s })).0["state"], "off");
        assert_eq!(tier_json(None), (json!({"state": "unknown"}), Value::Null));
        assert_eq!(sessions_json(&EngineStats::default(), 5)["pool"]["positions"], 0);
    }

    /// One line of the page's script that starts with `prefix`, whole.
    fn page_line(prefix: &str) -> String {
        let page = include_str!("dashboard.html");
        page.lines().find(|l| l.trim_start().starts_with(prefix)).unwrap_or_else(|| panic!("the page has no line starting {prefix:?}")).to_string()
    }

    /// A top-level function of the page's script, from `function NAME(` to the closing brace in column 0.
    fn page_function(name: &str) -> String {
        let page = include_str!("dashboard.html");
        let start = page.find(&format!("function {name}(")).unwrap_or_else(|| panic!("the page has no function {name}"));
        let end = page[start..].find("\n}\n").expect("the function ends in column 0") + 3;
        page[start..start + end].to_string()
    }

    /// Runs a script in node and returns what it printed; None when this host has no node (the page's own checks run where it has).
    fn node(src: &str) -> Option<String> {
        let out = std::process::Command::new("node").arg("-e").arg(src).output().ok()?;
        assert!(out.status.success(), "node: {}", String::from_utf8_lossy(&out.stderr));
        Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }

    #[test]
    fn a_long_legend_wraps_inside_its_head_and_the_hover_box_fits_a_phone() {
        // the CACHE legend wraps inside its head at the 1100 px layout, and on a phone the hover box wraps inside its plot
        let page = include_str!("dashboard.html");
        assert!(page_line(".legend {").contains("flex-wrap: wrap"), "the legend's keys wrap to a second line");
        assert!(page.contains(".plots .phead:has(.legend) { height: auto; min-height: 13px; }"), "a head with a legend grows when its legend wraps");
        assert!(page.contains("@media (max-width: 600px) { .plot .tip { white-space: normal; max-width: calc(100% - 2px); box-sizing: border-box; } }"), "on a phone the box wraps inside its plot");
    }

    #[test]
    fn the_whole_page_script_parses() {
        // the other page tests run single functions; one stray quote anywhere else breaks every tile in the browser, so the whole
        // script is compiled here (vm.Script: parsed, nothing run)
        let page = include_str!("dashboard.html");
        let (open, close) = (page.find("<script>").expect("the page has its script") + "<script>".len(), page.rfind("</script>").unwrap());
        let src = format!(
            "const vm = require('vm'); try {{ new vm.Script({}); console.log('parsed'); }} catch (e) {{ console.log('SyntaxError: ' + e.message); }}",
            serde_json::to_string(&page[open..close]).unwrap()
        );
        let Some(out) = node(&src) else { eprintln!("no node here"); return };
        assert_eq!(out, "parsed", "the page's script does not parse: {out}");
    }

    #[test]
    fn the_pool_tiles_parts_add_up_to_its_value_and_the_images_are_not_counted_twice() {
        // `held` is every taken chunk, the images' rows among them: 500 taken (of which 40 images) + 300 free = 800, not 840
        let src = format!("{}\nconsole.log(JSON.stringify(poolParts({{held: 500, free: 300, image_numbers: 40, numbers: 800}})));", page_function("poolParts"));
        let Some(out) = node(&src) else { eprintln!("no node here: the pool parts are checked where node runs"); return };
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!((v["taken"].as_u64(), v["free"].as_u64(), v["total"].as_u64(), v["images"].as_u64()), (Some(500), Some(300), Some(800), Some(40)));
        assert_eq!(v["taken"].as_u64().unwrap() + v["free"].as_u64().unwrap(), v["total"].as_u64().unwrap(), "the parts are the value");
        // the tile prints the total and both parts from these same counts, never the store's start-time share
        let page = include_str!("dashboard.html");
        // the merged CONTEXT CACHE tile: taken of total from those counts, no POOL box or tile, never the start-time store share
        assert!(page.contains("used.set(gib(SPARKS * pp.taken * nb), `of ${gib(SPARKS * pp.total * nb)} GiB`,"));
        assert!(!page.contains("'POOL'") && !page.contains("pool.store_numbers") && !page.contains("FREE NOW"));
    }

    #[test]
    fn a_missing_reading_is_a_dash_and_a_count_of_one_is_singular() {
        let src = format!("const whole = new Intl.NumberFormat('en-AU', {{ maximumFractionDigits: 0 }});\n{}\n{}\n{}\n{}\nconsole.log(JSON.stringify([gib(null), gib(undefined), gib(2 ** 31), pl(1, 'group'), pl(2, 'group'), pl(0, 'group')]));",
            page_line("const one ="), page_line("const none ="), page_line("const gib ="), page_line("const pl ="));
        let Some(out) = node(&src) else { eprintln!("no node here"); return };
        assert_eq!(out, r#"["-","-","2.0","1 group","2 groups","0 groups"]"#);
    }

    #[test]
    fn the_pages_words_say_what_each_number_is() {
        let page = include_str!("dashboard.html");
        let has = |t: &str| assert!(page.contains(t), "the page lost: {t}");
        let gone = |t: &str| assert!(!page.contains(t), "the page still says: {t}");
        // the token lists: the MiB held in memory and the conversations in files are two sets, and it is the server on Spark 1
        has("Their token lists take ${one.format(Number(m.ids_host_bytes) / 2 ** 20)} MiB in the server on Spark 1");
        // one name for the chunk pool; images are inside the taken figure; the split cache's state is on the page
        gone("'STORE'");
        gone("Not counted here");
        has("are inside the taken figure and freed when each request ends");
        has("Split cache: ${sp.on ? 'on' : 'off'}");
        // the node table's split: work is a remainder, and the pool figure says which moment it is
        has("a remainder");
        has("led.live ?");
        // a lines row is a request in order, with its user; queued requests are counted as waiting, not in use
        has("Request ${i + 1} in order: user ${x.key_label}; holds ${whole.format(x.kv_tokens)} tokens of context.");
        gone("Seat ${i + 1}");
        has("in use, ${waiting} waiting (a queued request has no line yet)."); gone("lines in use"); gone("class: 'bsub'");  // the LINES box has no title or count: the PHASE tip carries it
        // the clock: the chip's top clock, its reasons in words
        has("CLOCK_REASONS[x] || x");
        gone("the highest clock NVML names");
        // the zones' trip is said on Hot only
        assert_eq!(page.matches("The zones trip at ${TEMP_TRIP_C} °C.").count(), 2, "the Hot column's head tip and its cell tip");
        assert!(!page.contains("bright red at 90 and over; the zones trip"));
        // chart hover: idle seconds count as 0, the peak is the best 3 s rate; local time
        has("averaged over ${state.step} s (idle seconds count as 0)");
        has("best 3 s rate ${fmt(pk)} ${unit}");
        has("Times are local time.");
        // one spelling of the rate unit, no internal names, no KV jargon
        has("text: 'tok/s'");
        has("['TOK/S', 'Output tokens a second");
        gone("'t/s'");
        has("['PROMPT', 'Prompt tokens of the request.'], ['OUTPUT'");
        gone("no ablit name");
        gone("ablated");
        gone("KV positions");
        gone("KV images");
        gone("display reservation");
        // LENT OUT has no term that can never fill; the chunk is 2.78 MiB, not 2.8
        gone("models loaded while");
        gone("tenant_models_groups");
        has("const mib2 = new Intl.NumberFormat('en-AU', { minimumFractionDigits: 2, maximumFractionDigits: 2 });");
        gone("one.format(nb / 2 ** 20)");
        // restores, pieces, rails, wait
        has("requests waited ${whole.format(Number(w.wait_ns) / 1e6)} ms in all");
        has("split-cache quarters and bookmarks of any size");
        has("receive errors and transmit discards since the driver loaded");
        has("(m.waitS == null ? '-' : `~${one.format(m.waitS)} s`)");
        // a setting started off its list prints as its list does
        has("FMT_OF[key] ? FMT_OF[key](Number(v)) : v");
        // an Engram cache that is off says off when the engine reports its other figures
        has("st.memory ? 'off' : 'not reported'");
    }

    #[test]
    fn the_one_ssd_tile_is_per_spark_and_carries_the_held_tokens() {
        // one SSD CACHE tile in the top row: the held cache in GiB a Spark as its value, the size a Spark beside it, the share full and
        // the four Sparks together in the qualifier; the second row's SSD CACHE HELD tile is gone
        let page = include_str!("dashboard.html");
        assert!(!page.contains("store-ssd") && !page.contains("SSD CACHE HELD"));
        assert_eq!(page.matches("etile('ssd', 'SSD CACHE', ").count(), 1);
        assert!(page.contains("ssd.set(gib(Number(t.stored_bytes)), `of ${whole.format(capGib)} GiB/Spark`, ssdQual(SPARKS, Number(t.stored_bytes)),"));
        assert!(page.contains("const state = sp.state === 'on' ? 'on' : sp.state === 'off' ? 'off' : String(sp.state);"), "one wording for the state");
        assert!(!page.contains("'Enabled'") && !page.contains("'Disabled'"));
        assert!(page.contains("ssd_gib: (v) => v + ' GiB a Spark'"), "the size setting says a Spark too");
        assert!(page.contains("so the ${SPARKS} together hold"), "the four together is a second sentence of the tip");
    }

    #[test]
    fn token_counts_are_binary_with_one_decimal_and_a_whole_value_has_no_point_zero() {
        let src = format!("const whole = new Intl.NumberFormat('en-AU', {{ maximumFractionDigits: 0 }});\n{}\n{}\n{}\n{}\nconsole.log(JSON.stringify([4194304, 1048576, 25585254, 131072, 8192, 353000, 900, null].map(posFmt)));",
            page_line("const one ="), page_line("const none ="), page_line("const binFmt ="), page_line("const posFmt ="));
        let Some(out) = node(&src) else { eprintln!("no node here"); return };
        assert_eq!(out, r#"["4M","1M","24.4M","128K","8K","344.7K","900","0"]"#);
        let page = include_str!("dashboard.html");
        assert!(page.contains("rows: (v) => posFmt(v) + ' tokens',"), "the settings use the same format");
        assert!(!page.contains("toFixed(2) + ' M'") && !page.contains("v / 1e6"));
    }

    #[test]
    fn engram_size_rounds_to_the_nearest_mib() {
        // the engine's 8,070,857 rows of 264 bytes are 184 bytes short of 2,032 MiB: the list's 2032, not a floored 2031
        let v = settings_json(Some(&EngineStats { il_share: 40, engram_cache_bytes: 2_130_706_248, ..Default::default() }));
        assert_eq!(v["engram_mib"].as_u64(), Some(2032));
        // the top row's ENGRAM CACHE tile reads size_mib: the same 2032 (the tile's own formatting only prints that number)
        let e = engram_json(&EngineStats { engram_cache_bytes: 2_130_706_248, ..Default::default() }, None);
        assert_eq!(e["size_mib"].as_u64(), Some(2032));
    }

    #[test]
    fn taken_tokens_are_the_pools_tokens_in_proportion_to_its_taken_chunks() {
        // 800 chunks hold 25,585,254 tokens; 20 are taken: 20 x 25,585,254 / 800 = 639,631
        let src = format!("{}\nconsole.log(JSON.stringify([poolTokens({{numbers: 800, held: 20, positions: 25585254}}), poolTokens({{numbers: 0, held: 5, positions: 9}}), poolTokens({{numbers: 800, held: 0, positions: 25585254}})]));", page_function("poolTokens"));
        let Some(out) = node(&src) else { eprintln!("no node here"); return };
        assert_eq!(out, "[639631,0,0]");
        let page = include_str!("dashboard.html");
        assert!(page.contains("`${one.format(100 * pp.taken / Math.max(1, pp.total))} % · ${posFmt(tokTaken)} of ${posFmt(Number(p.positions))} tok`"));
    }

    #[test]
    fn the_gpu_clock_cell_is_the_clock_now_and_amber_only_for_a_hold_down_that_is_not_the_limit() {
        let src = format!("const whole = new Intl.NumberFormat('en-AU', {{ maximumFractionDigits: 0 }});\n{}\n{}\n{}\nconsole.log(JSON.stringify([gpuClockCell({{clock_mhz: 2190.4, clock_max_mhz: 3003}}), gpuClockCell({{}}), heldDown({{reasons: ['applications_clocks_setting']}}), heldDown({{reasons: ['applications_clocks_setting', 'sw_power_cap']}}), heldDown({{reasons: []}}), heldDown({{}})]));",
            page_line("const none ="), page_line("function gpuClockCell"), page_line("function heldDown"));
        let Some(out) = node(&src) else { eprintln!("no node here"); return };
        assert_eq!(out, r#"["2,190","-",false,true,false,false]"#);
    }

    #[test]
    fn in_force_names_the_nodes_that_differ_and_what_they_run() {
        let page = include_str!("dashboard.html");
        assert!(page.contains("`${who} runs ${profileLabel(n.profile)}`") && page.contains("`${who} not answering`"));
        assert!(!page.contains("'fan-profile'"), "the Profile column repeated the FAN PROFILE dropdown");
        let src = format!("const profileLabel = (x) => x;\n{}\nconsole.log(JSON.stringify([inForceText({{wanted: 'max', mixed: true, nodes: [{{rank: 0, ok: true, profile: 'max'}}, {{rank: 1, ok: true, profile: 'medium'}}, {{rank: 2, ok: false}}]}}), inForceText({{wanted: 'max', nodes: [{{rank: 0, ok: true, profile: 'max'}}]}})]));", page_function("inForceText"));
        let Some(out) = node(&src) else { eprintln!("no node here"); return };
        assert_eq!(out, r#"["Spark 2 runs medium; Spark 3 not answering","-"]"#);
    }

    #[test]
    fn the_engram_and_ssd_tiles_do_not_repeat_their_settings() {
        let page = include_str!("dashboard.html");
        // ENGRAM CACHE: the value is the last minute's hit rate, the size is in the tip only; SSD CACHE: held GiB, the size in the tip only
        assert!(page.contains("t.set(e.hit_rate_1m == null ? '-' : one.format(100 * Number(e.hit_rate_1m)) + ' %'"));
        assert!(page.contains("since start: ${posFmt(Number(e.lookups))} rows${going}"));
        // every top-strip qualifier is short enough for one line at 1920 px (dash-tiles.mjs measures it with worst-case values)
        for t in ["no lookups, last minute", "lentQual(SPARKS, gb, [gg(0), gg(1), 0, gg(3)])", "memory ${one.format(sh.memory)} · SSD ${one.format(sh.ssd)}"] { assert!(page.contains(t), "{t}"); }
        assert!(!page.contains("t.set(whole.format(Number(e.size_mib))"));
        assert!(page.contains("`of ${whole.format(capGib)} GiB/Spark`"), "the SSD tile says what it is of");
    }

    #[test]
    fn the_front_end_knows_how_long_it_has_been_connected_to_the_engine() {
        let t = Telemetry::new();
        assert_eq!(t.connected_s(), None, "not connected before the first HELLO");
        t.connected(true);
        assert!(t.connected_s().is_some_and(|s| s < 2));
        t.connected(false);
        assert_eq!(t.connected_s(), None, "a lost connection clears it");
    }

    #[test]
    fn the_pages_labels_name_the_units_of_each_figure() {
        let page = include_str!("dashboard.html");
        let has = |t: &str| assert!(page.contains(t), "the page lost: {t}");
        let gone = |t: &str| assert!(!page.contains(t), "the page still says: {t}");
        gone("refs.sub");  // the model name is SERVED AS, the line count is LINES'
        has("'connected ' + dur(en.connected_s)"); gone("front-end up");
        has("'MAX CONTEXT'"); has("'longest per request'"); gone("extend on"); gone("'CAPACITY'");
        has("'CONVERSATIONS'"); has("saved to SSD"); gone("on SSD only");
        gone("'SSD READ-BACKS'"); gone("'RESTORES'");
        has("'hit, 1 min'"); gone("'hit, last minute'"); gone("'MiB/Spark'"); gone("· ${pl(e.groups, 'group')}${going}");
        gone("'TOKEN IDS'"); gone("'SERVER IDS'");
        has("ENGINE SETTINGS"); gone("LONG PROMPTS AND SSD");
        has("['PROMPT', "); has("['OUTPUT', "); has("chart('DECODE tok/s'"); has("chart('PREFILL tok/s'");
        has("['ns', 'NVMe', 'Free / total', 'TB', SPACE_TIP]");
        has("engram_mib: (v) => whole.format(Number(v)) + ' MiB a Spark'");
        has("JUMP THE QUEUE UP TO");
        has("${il.reading} long ${il.reading === 1 ? 'prompt' : 'prompts'} being processed · ${il.queued} waiting");
        // LENT OUT is a tile of the one top strip; the POOL box and FREE NOW are not on the page
        has("kv.by.kv, store.by.ten"); gone("data-box': 'store'");
        // IN FORCE and CLOCK IN FORCE are hidden unless the Sparks differ or one did not take the choice
        has("shownField.hidden = !showP;"); has("clockField.hidden = !clockOdd(f);");
    }

    #[test]
    fn one_gap_between_the_boxes_and_a_lines_box_without_a_title() {
        let page = include_str!("dashboard.html");
        // one 6 px gap: around the board, between its boxes and between the two plots; the empty STORE row is gone
        assert!(page.contains("main#view { max-width: 1900px; padding: 6px; gap: 6px; }"));
        assert!(page.contains("\"engine engine\" \"lines plots\" \"health health\" \"reading fans\"; gap: 6px;"));
        assert!(page.contains(".plots { grid-area: plots; display: flex; flex-direction: column; gap: 6px;"));
        assert!(!page.contains("\"store store\"") && !page.contains("\"engine\" \"store\""), "no empty store row");
        // the rows of LINES sit 2 px apart, and its title is removed
        assert!(page.contains(".lgrid { display: flex; flex-direction: column; gap: 2px; }"));
        assert!(page.contains("b.el.querySelector('.bhead').remove();"));
    }

    #[test]
    fn the_ssd_size_list_is_128_to_1024_in_steps_of_128_with_the_engines_value_selected() {
        let s = EngineStats { il_share: 40, disk_cap_bytes: 384 << 30, ..Default::default() };
        let v = settings_json(Some(&s));
        let choices: Vec<u64> = v["ssd_choices"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
        assert_eq!(choices, vec![128, 256, 384, 512, 640, 768, 896, 1024]);
        assert!(choices.contains(&v["ssd_gib"].as_u64().unwrap()), "the engine's current size is one of the options, selected");
        // every option is a value the POST accepts
        for c in choices { assert!(parse_settings(&json!({"ssd_gib": c}), 1 << 20).is_ok()); }
    }

    // ---- the CACHE series, the CACHE HITS tile and the page's CACHE chart

    /// STATS counters for the cache series: lookups, hits, tokens read back, wait ns, pieces read, pieces taken back.
    fn cc(lookups: u64, hits: u64, read_tokens: u64, wait_ns: u64, pieces_read: u64, revives: u64) -> CacheCounters {
        CacheCounters { lookups, hits, read_tokens, wait_ns, pieces_read, revives }
    }

    #[test]
    fn the_cache_buckets_are_sums_of_the_requests_and_the_stats_changes_of_ten_seconds() {
        let mut c = CacheSeries::default();
        // bucket 100 (1000..1010 s): a first sample (no change yet), two prompts, a second sample 1840 rows looked up of which 1700 found
        c.add_stats(1000 * S, cc(5000, 4000, 100, 7_000_000, 10, 20));
        c.add_prompts(1002 * S, 2, 300_000, 280_000);
        c.add_stats(1008 * S, cc(6840, 5700, 100, 7_000_000, 10, 20));
        // bucket 101: one prompt of 100,000 tokens of which 90,000 were not computed, 60,000 read back from the SSDs: 12 ms wait, 2 pieces read, 5 from memory
        c.add_prompts(1011 * S, 1, 100_000, 90_000);
        c.add_stats(1015 * S, cc(6840, 5700, 60_100, 19_000_000, 12, 25));
        // bucket 102: prompts and no lookups and no read-back
        c.add_prompts(1021 * S, 3, 50_000, 0);
        c.add_stats(1022 * S, cc(6840, 5700, 60_100, 19_000_000, 12, 25));
        let a = c.query(1035 * S, 40, 10);
        assert_eq!(a.points.len(), 4, "buckets 100 to 103");
        assert_eq!(a.last_ms, 1040 * S);
        let b0 = a.points[0].expect("bucket 100");
        assert_eq!((b0.prompts, b0.prompt_tokens, b0.cached_tokens, b0.lookups, b0.hits), (2, 300_000, 280_000, 1840, 1700));
        assert_eq!(b0.engram_pct().map(|x| (x * 10.0).round() / 10.0), Some(92.4), "1700 of 1840 rows found");
        assert_eq!((b0.read_tokens, b0.ssd_used, b0.samples > 0), (0, 0, true), "counters read, nothing read back: 0, not 'not reported'");
        let s0 = b0.shares().unwrap();
        assert!((s0.memory - 93.333).abs() < 0.01 && s0.ssd == 0.0 && (s0.computed - 6.667).abs() < 0.01, "{s0:?}");
        let b1 = a.points[1].expect("bucket 101");
        assert_eq!((b1.read_tokens, b1.wait_ns, b1.pieces_read, b1.revives), (60_000, 12_000_000, 2, 5));
        let s1 = b1.shares().unwrap();
        assert!((s1.memory - 30.0).abs() < 1e-9 && (s1.ssd - 60.0).abs() < 1e-9 && (s1.computed - 10.0).abs() < 1e-9, "{s1:?}");
        assert_eq!(b1.engram_pct(), None, "a bucket with prompts and no lookups has no Engram point, not 0");
        let b2 = a.points[2].expect("bucket 102");
        assert_eq!(b2.shares().map(|s| (s.memory, s.ssd, s.computed)), Some((0.0, 0.0, 100.0)), "a cold prompt: memory 0, a dot at 0");
        assert_eq!(a.points[3], None, "nothing in bucket 103: no point, never a 0");
    }

    #[test]
    fn a_read_back_never_takes_memory_under_zero_or_more_than_was_not_computed() {
        let mut c = CacheSeries::default();
        c.add_stats(2000 * S, cc(0, 0, 0, 0, 0, 0));
        // the read-back of 80,000 tokens is counted in bucket 200; its prompt (90,000, of which 85,000 not computed) in bucket 201
        c.add_stats(2005 * S, cc(0, 0, 80_000, 4_000_000, 3, 0));
        c.add_prompts(2012 * S, 1, 90_000, 85_000);
        let a = c.query(2015 * S, 20, 10);
        let (b0, b1) = (a.points[0].unwrap(), a.points[1].unwrap());
        assert_eq!((b0.read_tokens, b0.ssd_used), (80_000, 0), "the engine counted it here; its prompt's column takes it");
        assert_eq!(b0.shares(), None, "no prompt in that bucket: no shares");
        let s1 = b1.shares().unwrap();
        assert!(s1.memory >= 0.0 && s1.memory <= 100.0 && s1.ssd >= 0.0 && s1.ssd <= 100.0, "{s1:?}");
        assert!((s1.memory + s1.ssd + s1.computed - 100.0).abs() < 1e-9, "the three shares add up to 100");
        let sw = c.window(2015 * S, 900).shares().unwrap();
        assert!((sw.ssd - 100.0 * 80_000.0 / 90_000.0).abs() < 1e-9 && (sw.memory - 100.0 * 5_000.0 / 90_000.0).abs() < 1e-9, "{sw:?}");
        // more read than was not computed (a read-back of a prompt that is not in the window): the column takes what it can, never negative
        let mut d = CacheSeries::default();
        d.add_stats(3000 * S, cc(0, 0, 0, 0, 0, 0));
        d.add_stats(3002 * S, cc(0, 0, 500_000, 0, 1, 0));
        d.add_prompts(3003 * S, 1, 1000, 400);
        let s = d.window(3005 * S, 900).shares().unwrap();
        assert_eq!((s.memory, s.ssd, s.computed), (0.0, 40.0, 60.0));
    }

    #[test]
    fn counters_that_go_back_mark_the_bucket_and_a_missing_reading_is_not_a_zero() {
        let mut c = CacheSeries::default();
        c.add_stats(1000 * S, cc(1000, 900, 5000, 1, 1, 1));
        c.add_stats(1004 * S, cc(1100, 990, 5000, 1, 1, 1));
        // the engine started again: every counter is lower
        c.add_stats(1013 * S, cc(10, 9, 0, 0, 0, 0));
        // and counts on from the new start
        c.add_stats(1014 * S, cc(30, 29, 400, 0, 0, 0));
        c.add_prompts(1015 * S, 1, 1000, 600);
        let a = c.query(1015 * S, 20, 10);
        let (b0, b1) = (a.points[0].unwrap(), a.points[1].unwrap());
        assert_eq!((b0.lookups, b0.hits), (100, 90));
        assert!(b1.restarted, "the bucket where a counter went back");
        assert_eq!((b1.engram_pct(), cache_point_json(&Some(b1))["counters_read"].as_bool()), (None, Some(false)), "that bucket has no Engram point and its counters are not read");
        assert!(b1.shares().is_some(), "the memory line is the front-end's own: drawn");
        // prompts without any STATS sample in the bucket: the engine's counters were not read
        let mut d = CacheSeries::default();
        d.add_prompts(1000 * S, 1, 100, 50);
        let b = d.query(1005 * S, 10, 10).points[0].unwrap();
        assert_eq!((b.engram_pct(), b.samples), (None, 0));
        assert_eq!(cache_point_json(&Some(b))["counters_read"], false);
        assert_eq!(cache_point_json(&None), Value::Null);
        // the series is 180 buckets, the tile's window the last 90
        d.add_prompts(1930 * S, 1, 7, 7);
        assert_eq!(d.query(1935 * S, 5000, 10).points.len(), 180);
        assert_eq!(d.window(1935 * S, 900).prompt_tokens, 7, "bucket 100 left the window");
    }

    #[test]
    fn the_tile_reads_the_same_sums_as_the_chart_and_its_shares_add_up() {
        let mut c = CacheSeries::default();
        c.add_stats(1000 * S, cc(0, 0, 0, 0, 0, 0));
        let mut read_total = 0;
        for (k, (p, cached, read)) in [(1000_u64, 900_u64, 0_u64), (50_000, 40_000, 30_000), (2_000, 0, 0), (700, 700, 100)].iter().enumerate() {
            let t = (1001 + 10 * k as u64) * S;
            c.add_prompts(t, 1, *p, *cached);
            read_total += read;
            c.add_stats(t + S, cc(0, 0, read_total, 0, 0, 0));
        }
        let w = c.window(1045 * S, 900);
        let a = c.query(1045 * S, 900, 10);
        let (mut p, mut ca, mut n, mut rd) = (0, 0, 0, 0);
        for b in a.points.iter().flatten() { p += b.prompt_tokens; ca += b.cached_tokens; n += b.prompts; rd += b.read_tokens; }
        assert_eq!((w.prompt_tokens, w.cached_tokens, w.prompts, w.read_tokens), (p, ca, n, rd), "the tile's sums are the series' sums");
        let j = cache_window_json(&w, 900);
        let sh = &j["shares"];
        let (m, s, comp, hit) = (sh["memory"].as_f64().unwrap(), sh["ssd"].as_f64().unwrap(), sh["computed"].as_f64().unwrap(), sh["hit"].as_f64().unwrap());
        assert!((m + s + comp - 100.0).abs() < 0.11, "memory + SSD + computed = 100 ({m} {s} {comp})");
        assert!((hit - (m + s)).abs() < 1e-9, "the big number is the two small ones added: {hit} = {m} + {s}");
        assert_eq!((j["prompts"].as_u64(), j["window_s"].as_u64(), j["counters_read"].as_bool()), (Some(4), Some(900), Some(true)));
        // no prompt in the window: shares null, the page shows '-'
        assert_eq!(cache_window_json(&CacheSeries::default().window(1000 * S, 900), 900)["shares"], Value::Null);
    }

    #[test]
    fn the_cache_series_answer_has_the_parts_of_each_bucket_and_the_raw_sums() {
        let mut c = CacheSeries::default();
        c.add_stats(1000 * S, cc(0, 0, 0, 0, 0, 0));
        c.add_prompts(1001 * S, 1, 1000, 800);
        c.add_stats(1002 * S, cc(100, 97, 300, 5_000_000, 2, 1));
        let j = cache_answer_json(&c.query(1005 * S, 30, 10), 10);
        assert_eq!((j["step_s"].as_u64(), j["last_ms"].as_u64()), (Some(10), Some(1010 * S)));
        let pts = j["points"].as_array().unwrap();
        assert_eq!(pts.len(), 3);
        assert_eq!((pts[0].clone(), pts[1].clone()), (Value::Null, Value::Null));
        let p = &pts[2];
        // 300 of the 1,000 tokens read back from the SSDs, 500 more in memory (800 not computed), 200 computed
        assert_eq!((p["memory_tok"].as_u64(), p["ssd_tok"].as_u64(), p["computed_tok"].as_u64()), (Some(500), Some(300), Some(200)));
        assert_eq!((p["lookups"].as_u64(), p["hits"].as_u64(), p["read_tokens"].as_u64()), (Some(100), Some(97), Some(300)));
        for gone in ["engram", "memory", "ssd_pct", "ssd"] { assert!(p.get(gone).is_none(), "the shares are made by the page from the sums: no {gone}"); }
        assert_eq!((p["prompts"].as_u64(), p["prompt_tokens"].as_u64(), p["wait_ns"].as_u64(), p["pieces_read"].as_u64(), p["revives"].as_u64()), (Some(1), Some(1000), Some(5_000_000), Some(2), Some(1)));
    }

    #[test]
    fn a_request_is_counted_once_when_its_answer_is_first_known() {
        let t = Telemetry::new();
        let meta = |p| ReqMeta { key_label: "api".into(), ablit: false, prompt_tokens: p, endpoint: "/v1/chat/completions", priority: "normal" };
        // progress first: 100,000 prompt, 12,000 to compute: 88,000 not computed; a second progress and the first token add nothing
        let a = t.register(1, meta(100_000));
        t.progress(&a, 4096, 12_000);
        t.progress(&a, 12_000, 12_000);
        t.tokens(&a, 1);
        // no progress at all (the whole prompt reused): the first output token counts it, all of it not computed
        let b = t.register(2, meta(5_000));
        t.tokens(&b, 1);
        t.tokens(&b, 1);
        // a request that never got an answer (an error before either) is not counted
        let _c = t.register(3, meta(9_999));
        t.tick(Instant::now(), 1003 * S);
        let w = t.cache_window(1003 * S, 900);
        assert_eq!((w.prompts, w.prompt_tokens, w.cached_tokens), (2, 105_000, 93_000));
        // a later second adds only what is new
        let d = t.register(4, meta(1_000));
        t.progress(&d, 1_000, 1_000);
        t.tick(Instant::now(), 1004 * S);
        assert_eq!(t.cache_window(1004 * S, 900).prompts, 3);
        // STATS reach the same series
        t.set_stats_at(1004 * S, EngineStats { engram_lookups: 10, engram_hits: 5, ..Default::default() });
        t.set_stats_at(1005 * S, EngineStats { engram_lookups: 30, engram_hits: 25, restored_tokens_total: 700, ..Default::default() });
        let w = t.cache_window(1005 * S, 900);
        assert_eq!((w.lookups, w.hits, w.read_tokens, w.samples), (20, 20, 700, 1));
    }

    #[test]
    fn the_engine_is_asked_for_stats_every_ten_seconds_when_nobody_watches() {
        let t = Telemetry::new();
        let at = |s: u64| 5000 * S + s * S;
        assert!(t.stats_due(at(0)), "the first time");
        assert!(!t.stats_due(at(1)) && !t.stats_due(at(9)), "not within ten seconds");
        assert!(t.stats_due(at(10)), "ten seconds on");
        assert!(!t.stats_due(at(11)));
        // watched: every second
        t.watched_ms.store(at(11), Ordering::Relaxed);
        assert!(t.stats_due(at(12)) && t.stats_due(at(13)) && t.stats_due(at(14)), "a watched dashboard asks every second");
    }

    #[test]
    fn the_ssd_tile_carries_no_converted_token_figure() {
        // the tile shows only what the engine reports: no token figure converted from a whole chunk's bytes
        let (_, cache) = tier_json(Some(&stats()));
        assert!(cache["spill"][0].get("positions").is_none(), "no 'positions': the quarter images are not whole chunks");
        assert_eq!(cache["spill"][0]["stored_bytes"].as_u64(), Some(5 << 30));
    }

    #[test]
    fn the_cache_hits_tile_reads_shares_of_all_prompt_tokens() {
        let src = format!("const whole = new Intl.NumberFormat('en-AU', {{ maximumFractionDigits: 0 }});\n{}\n{}\n{}\n{}\nconsole.log(JSON.stringify([cacheFace({{window_s: 900, prompts: 41, shares: {{memory: 95.1, ssd: 1.3, computed: 3.6, hit: 96.4}}}}), cacheFace({{window_s: 900, prompts: 0, shares: null}}), cacheFace(null)]));",
            page_line("const one ="), page_line("const none ="), page_line("const posFmt ="), page_function("cacheFace"));
        let Some(out) = node(&src) else { eprintln!("no node here: the tile is checked where node runs"); return };
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!((v[0]["v"].as_str(), v[0]["u"].as_str(), v[0]["q"].as_str()), (Some("96.4 %"), Some("hit, 15 min"), Some("memory 95.1 · SSD 1.3")));
        assert_eq!((v[1]["v"].as_str(), v[1]["u"].as_str(), v[1]["q"].as_str()), (Some("-"), Some(""), Some("no prompts, 15 min")));
        assert_eq!(v[2]["v"].as_str(), Some("-"), "no cache block from the front-end: a dash");
    }

    // ---- the CACHE chart: running lines (ratios of sums over the trailing 15 minutes) and bars of each bucket's prompt tokens

    /// The page's pure CACHE functions and what they read, as a node script prefix.
    fn cache_page_fns(names: &[&str]) -> String {
        let mut s = format!("const whole = new Intl.NumberFormat('en-AU', {{ maximumFractionDigits: 0 }});\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n",
            page_line("const one ="), page_line("const none ="), page_line("const binFmt ="), page_line("const posFmt ="), page_line("const pl ="), page_line("const CACHE_WIN ="), page_line("const gib ="));
        for n in names { s += &page_function(n); s += "\n"; }
        s
    }

    /// A point of the series as the route sends it, for the running lines: prompt tokens split into memory / SSD / computed, and Engram's rows.
    fn pt(memory: u64, ssd: u64, computed: u64, lookups: u64, hits: u64) -> Value {
        json!({"prompt_tokens": memory + ssd + computed, "memory_tok": memory, "ssd_tok": ssd, "computed_tok": computed, "lookups": lookups, "hits": hits, "prompts": 1})
    }

    #[test]
    fn the_running_value_is_a_ratio_of_the_windows_sums_not_a_mean_of_shares() {
        // a bucket of 1 prompt token found nowhere and one of 999 found in memory: 99.9 % of the tokens, not the mean of 0 % and 100 %
        let pts = json!([pt(0, 0, 1, 10, 0), pt(999, 0, 0, 990, 990)]);
        let src = format!("{}\nconst r = cacheRunning({pts});\nconsole.log(JSON.stringify([r[1].memory, r[1].ssd, r[1].engram, r[0].memory]));", cache_page_fns(&["cacheRunning"]));
        let Some(out) = node(&src) else { eprintln!("no node here: the running values are checked where node runs"); return };
        assert_eq!(out, "[99.9,0,99,0]", "memory 999 of 1,000 tokens; Engram 990 of 1,000 rows; the first bucket's own window is itself");
    }

    #[test]
    fn a_quiet_stretch_keeps_the_running_value_and_no_record_has_none() {
        // one bucket with a prompt, then 50 recorded quiet buckets (zero tokens, zero rows), a bucket with no record, then quiet ones again
        let mut pts = vec![pt(300, 100, 100, 50, 40)];
        for _ in 0..50 { pts.push(pt(0, 0, 0, 0, 0)); }
        pts.push(Value::Null);
        for _ in 0..50 { pts.push(pt(0, 0, 0, 0, 0)); }
        let src = format!("{}\nconst r = cacheRunning({});\nconsole.log(JSON.stringify([r[0].memory, r[0].ssd, r[25].memory, r[50].ssd, r[50].engram, r[51].memory, r[52].memory, r[89].memory, r[90].memory, r[100].memory]));",
            cache_page_fns(&["cacheRunning"]), Value::Array(pts));
        let Some(out) = node(&src) else { eprintln!("no node here"); return };
        // 60 %, 20 % and Engram 80 % go on flat through the quiet stretch and on past the 90th bucket: only a request moves them. No record is null
        // (the front-end was not running); the quiet buckets after it carry the value again
        assert_eq!(out, "[60,20,60,20,80,null,60,60,60,60]");
    }

    #[test]
    fn the_running_lines_end_at_the_cache_hits_tiles_values() {
        // 140 buckets of prompts, read-backs landing in the bucket of their prompt and in the one before it, then the series as the route sends it
        let mut c = CacheSeries::default();
        c.add_stats(1000 * S, cc(0, 0, 0, 0, 0, 0));
        let (mut read, mut looked, mut found) = (0, 0, 0);
        for k in 0..140_u64 {
            // a prompt in every bucket, so the 90 buckets with traffic are the 90 of the tile's wall-clock window
            {
                let (p, cached) = (1000 + 37 * k, (1000 + 37 * k) * 3 / 4);
                c.add_prompts((1001 + 10 * k) * S, 1, p, cached);
                if k % 5 == 0 { read += cached / 3; }
            }
            looked += 100 + k;
            found += 90 + k / 2;
            // the sample falls in the prompt's bucket (offset 2 s) or in the next one (9 s)
            c.add_stats((1001 + 10 * k + if k % 2 == 0 { 9 } else { 2 }) * S, cc(looked, found, read, 0, 0, 0));
        }
        let now = (1001 + 10 * 139 + 5) * S;
        let body = cache_answer_json(&c.query(now, 1800, 10), 10);
        let tile = cache_window_json(&c.window(now, 900), 900);
        let src = format!("{}\nconst r = cacheRunning({}.points);\nconst l = r[r.length - 1], x = (v) => Math.round(v * 10) / 10;\nconsole.log(JSON.stringify([r.length, x(l.memory), x(l.ssd)]));", cache_page_fns(&["cacheRunning"]), body);
        let Some(out) = node(&src) else { eprintln!("no node here"); return };
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v[0].as_u64(), Some(180), "the series is asked for 30 minutes: a point's whole window is there");
        assert!(tile["read_tokens"].as_u64().unwrap() > 0 && tile["shares"]["ssd"].as_f64().unwrap() > 0.0, "the data reads back from the SSDs");
        assert_eq!((v[1].as_f64(), v[2].as_f64()), (tile["shares"]["memory"].as_f64(), tile["shares"]["ssd"].as_f64()), "the lines' last values are the tile's");
    }

    #[test]
    fn a_bucket_with_no_prompt_has_no_bar() {
        let src = format!("{}\nconsole.log(JSON.stringify([barStack(null), barStack({}), barStack({}), barStack({})]));",
            cache_page_fns(&["barStack"]), pt(0, 0, 0, 5, 5), pt(60, 30, 10, 0, 0), pt(0, 0, 7, 0, 0));
        let Some(out) = node(&src) else { eprintln!("no node here"); return };
        assert_eq!(out, r#"[[],[],[{"k":"memory","from":0,"to":60},{"k":"ssd","from":60,"to":90},{"k":"computed","from":90,"to":100}],[{"k":"computed","from":0,"to":7}]]"#,
            "no record and a quiet bucket: no bar; a prompt: its parts stacked from 0 in tokens");
    }

    #[test]
    fn the_token_scale_steps_from_64k_by_doubling() {
        let src = format!("{}\nconsole.log(JSON.stringify([tokTop(0), tokTop(57640), tokTop(65536), tokTop(66000), tokTop(1003622), tokTop(1048577), tokTop(5000000), [tokTop(0), tokTop(1003622)].map((t) => [posFmt(t), posFmt(t / 2)])]));",
            cache_page_fns(&["tokTop"]));
        let Some(out) = node(&src) else { eprintln!("no node here"); return };
        assert_eq!(out, r#"[65536,65536,65536,131072,1048576,2097152,8388608,[["64K","32K"],["1M","512K"]]]"#);
    }

    #[test]
    fn the_cache_series_is_asked_for_1800_seconds_and_the_plot_draws_900() {
        let src = format!("{}\nconst SERIES_S = {{ decode_tps: 900, prefill_tps: 900, cache: 1800 }};\nconsole.log(JSON.stringify([SERIES_S.cache, CACHE_WIN * 10]));", cache_page_fns(&[]));
        let Some(out) = node(&src) else { eprintln!("no node here"); return };
        assert_eq!(out, "[1800,900]");
        let page = include_str!("dashboard.html");
        assert!(page.contains("const SERIES_S = { decode_tps: 900, prefill_tps: 900, cache: 1800 };") && page.contains("seconds=${SERIES_S[n]}"), "the page asks each series for its own length");
        assert!(page_function("cacheChart").contains("body.points.slice(-CACHE_WIN)"), "the plot draws the last 15 minutes of what it was given");
        // the front-end keeps and answers 30 minutes of buckets; the tile's window is the last 15
        let mut c = CacheSeries::default();
        c.add_prompts(600 * S, 1, 7, 7);
        c.add_prompts(2500 * S, 1, 9, 9);
        assert_eq!(c.query(2505 * S, 1800, 10).points.len(), 180);
        assert_eq!(c.query(2505 * S, 5000, 10).points.len(), 180, "at most 30 minutes");
        assert_eq!(c.window(2505 * S, 900).prompt_tokens, 9);
        assert_eq!(c.query(2505 * S, 1800, 10).points.iter().flatten().map(|b| b.prompt_tokens).sum::<u64>(), 9, "bucket 60 is older than 30 minutes");
        c.add_prompts(1500 * S, 1, 1, 1);
        assert_eq!(c.query(2505 * S, 1800, 10).points.iter().flatten().map(|b| b.prompt_tokens).sum::<u64>(), 9, "an older time is refused");
    }

    #[test]
    fn no_gutter_label_takes_a_style_meant_for_another_element() {
        // the labels' classes must not be `top`, `mid` or `zero`: the page header's `.top` rule would style them
        let shell = page_function("plotShell");
        for bad in ["'g top'", "'g mid'", "'g zero'", "class: 'g top", "class: 'g mid", "class: 'g zero"] { assert!(!shell.contains(bad), "plotShell still has {bad}"); }
        let page = include_str!("dashboard.html");
        let first = |r: &str| -> String { r.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_').collect() };
        let own: Vec<String> = page.lines().filter_map(|l| l.strip_prefix('.')).map(first).collect();
        assert!(shell.contains("'data-at'"), "the label's place is an attribute, no class");
        assert!(!own.iter().any(|c| c == "top" || c == "mid" || c == "zero") || !shell.contains("class: 'g "), "no label takes `.top`");
    }

    #[test]
    fn a_read_back_lands_in_the_bucket_of_its_prompt_before_or_after_it() {
        let ssd = |c: &CacheSeries, now: u64, secs: u64| -> Vec<Value> { cache_answer_json(&c.query(now * S, secs, 10), 10)["points"].as_array().unwrap().clone() };
        // read in bucket 100, its prompt in 101: the read is in 101's column
        let mut c = CacheSeries::default();
        c.add_stats(1000 * S, cc(0, 0, 0, 0, 0, 0));
        c.add_stats(1005 * S, cc(0, 0, 80_000, 4_000_000, 3, 0));
        c.add_prompts(1012 * S, 1, 90_000, 85_000);
        let p = ssd(&c, 1015, 20);
        assert_eq!((p[0]["ssd_tok"].as_u64(), p[1]["ssd_tok"].as_u64(), p[1]["memory_tok"].as_u64(), p[1]["computed_tok"].as_u64()), (Some(0), Some(80_000), Some(5_000), Some(5_000)));
        assert_eq!(p[1]["read_unmatched"].as_u64(), Some(0));
        // its prompt in 200, the read in 201: the read is in 200's column, 201 has none
        let mut d = CacheSeries::default();
        d.add_stats(2000 * S, cc(0, 0, 0, 0, 0, 0));
        d.add_prompts(2002 * S, 1, 50_000, 40_000);
        d.add_stats(2011 * S, cc(0, 0, 30_000, 0, 1, 0));
        let p = ssd(&d, 2015, 20);
        assert_eq!((p[0]["ssd_tok"].as_u64(), p[0]["memory_tok"].as_u64(), p[1]["ssd_tok"].as_u64(), p[1]["prompt_tokens"].as_u64()), (Some(30_000), Some(10_000), Some(0), Some(0)));
        // a read with no prompt in 300 to 302: in the unmatched figure of the bucket that opens third (303), no column's split changed
        let mut e = CacheSeries::default();
        e.add_prompts(2995 * S, 1, 1000, 0);
        e.add_stats(3000 * S, cc(0, 0, 0, 0, 0, 0));
        e.add_stats(3005 * S, cc(0, 0, 5_000, 0, 1, 0));
        e.add_stats(3031 * S, cc(0, 0, 5_000, 0, 1, 0));
        let p = ssd(&e, 3035, 60);
        assert_eq!(p.iter().filter(|x| !x.is_null()).map(|x| x["ssd_tok"].as_u64().unwrap()).sum::<u64>(), 0);
        assert_eq!(p[5]["read_unmatched"].as_u64(), Some(5_000), "bucket 303");
        assert_eq!(p.iter().filter(|x| !x.is_null()).map(|x| x["read_unmatched"].as_u64().unwrap()).sum::<u64>(), 5_000);
        // only the newest bucket and the two before it can take a read: a prompt three buckets back stays as it was
        let mut f = CacheSeries::default();
        f.add_stats(4000 * S, cc(0, 0, 0, 0, 0, 0));
        f.add_prompts(4002 * S, 1, 1000, 1000);
        f.add_stats(4035 * S, cc(0, 0, 700, 0, 1, 0));
        f.add_stats(4065 * S, cc(0, 0, 700, 0, 1, 0));
        let p = ssd(&f, 4065, 70);
        assert_eq!((p[0]["ssd_tok"].as_u64(), p[0]["prompt_tokens"].as_u64(), p[6]["read_unmatched"].as_u64()), (Some(0), Some(1000), Some(700)));
    }

    #[test]
    fn every_second_the_front_end_runs_is_a_record_and_before_it_there_is_none() {
        let t = Telemetry::new();
        for s in 1001..=1025 { t.tick(Instant::now(), s * S); }
        let j = cache_answer_json(&t.cache_series(1025 * S, 60, 10), 10);
        let p = j["points"].as_array().unwrap();
        assert_eq!(p.len(), 6);
        assert!(p[..3].iter().all(Value::is_null), "before the front-end ran: no record");
        for q in &p[3..] { assert_eq!((q["prompt_tokens"].as_u64(), q["memory_tok"].as_u64(), q["prompts"].as_u64()), (Some(0), Some(0), Some(0)), "a quiet bucket is a zero bucket: {q}"); }
    }

    #[test]
    fn a_restart_drops_a_read_back_waiting_for_its_prompt() {
        let mut c = CacheSeries::default();
        c.add_stats(1000 * S, cc(10, 5, 100, 0, 1, 0));
        c.add_stats(1004 * S, cc(10, 5, 90_100, 0, 2, 0));
        // the engine started again before the prompt came: the waiting read belongs to the old run
        c.add_stats(1008 * S, cc(1, 1, 0, 0, 0, 0));
        c.add_prompts(1012 * S, 1, 100_000, 95_000);
        let p = cache_answer_json(&c.query(1015 * S, 20, 10), 10)["points"].as_array().unwrap().clone();
        assert_eq!((p[1]["ssd_tok"].as_u64(), p[1]["memory_tok"].as_u64(), p[0]["restarted"].as_bool()), (Some(0), Some(95_000), Some(true)));
    }

    #[test]
    fn the_right_scale_column_is_the_same_width_on_all_three_plots() {
        let page = include_str!("dashboard.html");
        let shell = page_function("plotShell");
        assert!(shell.contains("class: 'rgutter'") && shell.contains("h('div', { class: 'pbody' }, gutter, plot, rgutter)"), "every plot has the column; DECODE and PREFILL leave it empty");
        assert!(page.contains(".rgutter { position: relative; width: var(--gr, 0px); flex: none;"), "one width from one variable, on the box of the three plots");
        assert!(page.contains(".plots .phead { height: 13px; align-items: center; padding-right: var(--gr, 0px); }"), "the legend ends at the plot's right edge");
        assert!(page.contains(".pfoot > .rfoot { width: var(--gr, 0px); flex: none; }"));
        let fit = page_function("fitGutters");
        assert!(fit.contains("'--gl'") && fit.contains("'--gr'") && fit.contains(".rgutter .g"), "sized from the labels");
    }

    #[test]
    fn the_cache_hover_has_the_running_values_the_bar_and_the_read() {
        let full = json!({"prompt_tokens": 412_300, "memory_tok": 392_100, "ssd_tok": 5_400, "computed_tok": 14_800, "prompts": 3, "counters_read": true,
                          "read_tokens": 5_400, "pieces_read": 12, "revives": 0, "wait_ns": 180_000_000, "lookups": 10, "hits": 9});
        let quiet = json!({"prompt_tokens": 0, "memory_tok": 0, "ssd_tok": 0, "computed_tok": 0, "prompts": 0, "counters_read": true, "read_tokens": 0, "pieces_read": 0, "revives": 4, "wait_ns": 0});
        let unread = json!({"prompt_tokens": 100, "memory_tok": 80, "ssd_tok": 0, "computed_tok": 20, "prompts": 1, "counters_read": false, "read_tokens": 0, "pieces_read": 0, "revives": 0, "wait_ns": 0});
        let run = json!({"memory": 95.1, "ssd": 1.3, "engram": 97.0});
        let none = json!({"memory": null, "ssd": null, "engram": null});
        let gib = 1u64 << 30;
        let fill = json!({"pct": 41.2, "taken": (121.4 * gib as f64) as u64, "total": (295.0 * gib as f64) as u64});
        let src = format!("{}\nconsole.log(JSON.stringify([cacheRows({run}, {full}, {fill}, '23:17'), cacheRows({run}, {quiet}, {fill}, '23:17'), cacheRows({run}, null, null, '23:17'), cacheRows({none}, {unread}, null, null)]));", cache_page_fns(&["cacheRows"]));
        let Some(out) = node(&src) else { eprintln!("no node here"); return };
        let v: Value = serde_json::from_str(&out).unwrap();
        let rows = |i: usize| -> Vec<String> { v[i].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect() };
        assert_eq!(rows(0), ["in memory 95.1 %", "from SSD 1.3 %", "Engram 97.0 %", "context cache 41.2 % · 121.4 of 295.0 GiB", "each over the last 15 min of requests",
                             "last request 23:17", "412,300 prompt tok, 3 prompts", "memory 392,100 · SSD 5,400 · computed 14,800", "SSD read: 12 pieces, waited 180 ms"]);
        assert_eq!(rows(1)[6..], ["no prompt", "4 pieces taken back from memory, no read"]);
        assert_eq!(rows(2)[6..], ["no record"], "the front-end was not running");
        assert_eq!(rows(3)[..4], ["in memory -", "from SSD -", "Engram -", "context cache -"]);
        assert_eq!(rows(3)[4..6], ["each over the last 15 min of requests", "no request yet"], "before any request there is no time to say");
        assert_eq!(rows(3)[8..], ["SSD read not reported"]);
        for r in (0..4).flat_map(rows) { assert!(r.chars().count() <= 48, "a row fits the narrow plot: {r}"); }
    }

    #[test]
    fn the_cache_chart_has_lines_on_the_left_scale_and_bars_on_the_right() {
        // lines in per cent (left), bars of prompt tokens (right, the tokens' own scale); the right column is on every plot, empty unless it has labels
        let page = include_str!("dashboard.html");
        let f = page_function("cacheChart");
        assert!(f.contains("key(COLOR.memory, 'in memory %'), key(COLOR.ssd, 'from SSD %'), key(COLOR.engram, 'Engram %')"), "the three line keys, in the order of the design");
        assert!(f.contains("'prompt tok (right)'") && f.contains("right: true") && f.contains("setText(s.rg[0], posFmt(top))"), "the bars' scale is on the right, from tokTop");
        assert!(f.contains("cacheRunning(all)") && f.contains("tokTop("), "lines from the running ratios, bars from the token scale");
        assert!(!f.contains("p.ssd_pct") && !f.contains("p.engram") && !f.contains("p.memory "), "no per-bucket share is read: the lines are ratios of sums");
        assert!(!page.contains("function ssdTop") && page.contains("rgutter") && page.contains("--gr"));
    }

    #[test]
    fn the_strip_has_ten_tiles_in_the_order_the_pool_adds_up() {
        let page = include_str!("dashboard.html");
        // FIRST TOKEN, GENERATED, CACHE HITS, CONTEXT CACHE, LENT OUT, CONVERSATIONS, SHARED PREFIXES, MAX CONTEXT, SSD CACHE, ENGRAM CACHE
        assert!(page.contains("rate.els, cachehits.els, kv.by.kv, store.by.ten, kv.by.ctx, kv.by.shared, kv.by.cap, tier.by.ssd, eg.els"));
        assert!(page.contains("els: [t.el, g.el]") && page.contains("etile('generated', 'GENERATED'"), "GENERATED beside FIRST TOKEN");
        assert!(page.contains("repeat(10, minmax(0, 1fr))"));
        assert!(page.contains("live.generated_tokens") && page.contains("countFmt("), "the box reads the front-end's total, in K / M / B");
        for gone in ["Copy drafts: ", "copyLine", "st.copy", "'TOKEN IDS'", "'SSD READ-BACKS'", "store-ids", "'restores'", "tok saved · wait", "partly on SSD", "'MiB on rank 0'"] {
            assert!(!page.contains(gone), "the page still has {gone}");
        }
        for has in ["'CACHE HITS'", "'LENT OUT'", "saved to SSD", "Their token lists take", "JUMP THE QUEUE UP TO"] {
            assert!(page.contains(has), "the page lost {has}");
        }
        assert!(!page.contains("READ A 2ND PROMPT BESIDE UP TO"));
        // CONTEXT CACHE and LENT OUT say what they are of
        assert!(page.contains("`${one.format(100 * pp.taken / Math.max(1, pp.total))} % · ${posFmt(tokTaken)} of ${posFmt(Number(p.positions))} tok`"));
        assert!(page.contains("`of ${gib(SPARKS * Number(m.store_numbers) * nb)} GiB`"), "LENT OUT is of the pool the engine took at start");
        assert!(!page.contains("% full ·"), "the SSD tile's value already reads of its size: the qualifier does not repeat the per cent");
        assert!(!page.contains("Number(t.positions)"), "no converted token figure on the SSD tile");
    }

    #[test]
    fn the_three_plots_share_one_box_one_time_row_and_the_same_left_and_right_edges() {
        let page = include_str!("dashboard.html");
        // one box holds the three plots; the time row is under CACHE only
        assert!(page.contains("h('section', { class: 'box plotbox', 'data-box': 'plots' }, cd.el, cp.el, cc.el)"));
        assert_eq!(page.matches("const wrap = (name, c)").count(), 0, "no box a chart");
        assert!(page.contains(".plots > .plotbox { flex: 1; display: flex; flex-direction: column; gap: 8px; padding: 8px 10px 6px; }"));
        assert!(page.contains(".plots .pbody { flex: 1; min-height: 64px; }"));
        assert!(page.contains(".gutter { position: relative; width: var(--gl, 24px);"));
        // every plot is made by plotShell with its scale columns of one width each (fitGutters): the three plots' edges are equal by construction
        let shell = page_function("plotShell");
        assert!(shell.contains("class: 'gutter'") && shell.contains("class: 'rgutter'"));
        assert!(page.contains("@media (max-width: 1099px)") && page.contains(".plots .pbody { min-height: 110px; }"));
        // the time labels are written once, under CACHE
        assert_eq!(page.matches("xlabels(").count(), 2, "its definition and the CACHE chart's one call");
        // the hairline runs through all three plots
        assert!(page.contains("for (const c of CHARTS) c.hairAt("));
    }

    #[test]
    fn the_cache_chart_draws_its_three_lines_and_its_bars_with_the_designs_colours_and_never_a_zero_for_no_data() {
        let page = include_str!("dashboard.html");
        for c in ["#3aa888", "#3f7fc4", "#c8782c", "#5a6165"] { assert!(page.contains(c), "series colour {c}"); }
        let f = page_function("cacheChart");
        assert!(f.contains("'in memory %'") && f.contains("'from SSD %'") && f.contains("'Engram %'"));
        // a null point breaks the line; a lone point is a dot; the bars are faint and sit behind the lines
        assert!(f.contains("if (p == null) { flush(); return; }"));
        assert!(f.contains("svg('g', { 'fill-opacity': '0.35' })") && f.contains("s.plotSvg.append(bars, fillLine, fillDots, engLine"), "faint bars behind the lines");
        assert!(f.contains("if (!stack.length) return;"), "a bucket with no prompt has no bar");
        // the hover's rows are written by cacheRows: a bucket with no prompt, no record and counters not read each have their words
        let rows = page_function("cacheRows");
        for row in ["no prompt", "no record", "SSD read not reported", "nothing read from the SSDs"] { assert!(rows.contains(row), "hover form: {row}"); }
    }

    #[test]
    fn the_scale_columns_fit_their_labels_and_the_legend_ends_where_the_plot_ends() {
        // the scale column is as wide as its widest label, not a fixed width, and the legend ends at the right column
        let page = include_str!("dashboard.html");
        assert!(page.contains(".pbody { display: flex; }"), "no gap between the columns: the 4 px are inside each column");
        assert!(page.contains(".gutter .g { position: absolute; right: 4px;"));
        // the head (title, legend) spans the scale column and the plot and is padded by the right column, so the legend ends where the plot ends
        assert!(page.contains(".plots .phead { height: 13px; align-items: center; padding-right: var(--gr, 0px); }"), "the legend ends at the plot's right edge");
        assert!(page.contains(".pfoot > span { width: var(--gl, 24px); flex: none; }"), "the time row under the plot");
        assert!(page_function("plotShell").contains("h('div', { class: 'pbody' }, gutter, plot, rgutter)"));
        let fit = page_function("fitGutters");
        assert!(fit.contains("'--gl'") && fit.contains("'--gr'"), "both scale columns");
        assert!(page.contains("fitGutters(cd.el.parentNode);"), "fitted after every draw, on the box of the three plots");
        // the grid is solid hairlines; the lines 2 px
        let shell = page_function("plotShell");
        assert!(!shell.contains("stroke-dasharray"), "a dashed grid reads as a threshold");
        assert!(!page_function("chart").contains("'stroke-width': '1.25'") && !page_function("cacheChart").contains("'1.25'"));
        // an idle plot's scale is whole numbers, and a scale that reaches 10,000 is compact
        assert!(page_function("chart").contains("Math.max(CHART_FLOOR[name] || 1, niceTop("));
        let src = format!("{}\n{}\n{}\n{}\nconsole.log(JSON.stringify([CHART_FLOOR.decode_tps, CHART_FLOOR.prefill_tps, tickFmt(10, 10), tickFmt(5, 10), tickFmt(0.5, 1), tickFmt(8000, 8000), tickFmt(4000, 8000), tickFmt(10000, 10000), tickFmt(5000, 10000), tickFmt(15000, 15000)]));",
            page_line("const loose ="), page_line("const CHART_FLOOR ="), page_line("const tickCompact ="), page_line("const tickFmt ="));
        let Some(out) = node(&src) else { eprintln!("no node here"); return };
        assert_eq!(out, r#"[10,100,"10","5","0.5","8,000","4,000","10K","5K","15K"]"#);
    }

    #[test]
    fn the_hover_box_stays_inside_its_plot_beside_the_hairline() {
        // the hover box is placed by its own width beside the hairline, never centred on the pointer or clamped to a fixed anchor (a wide box would pass the plot's edge)
        let tip = page_line(".plot .tip {");
        assert!(!tip.contains("translateX(-50%)"), "the box is placed by its own width, not centred on the pointer");
        for f in ["chart", "cacheChart"] {
            let body = page_function(f);
            assert!(!body.contains("tipBox.style.left = Math.min(88"), "{f}: no clamp to 88 %");
            let fill = body.find("clear(s.tipBox).append(").unwrap_or_else(|| panic!("{f} fills its box"));
            let place = body.find("placeTip(s.plot, s.tipBox, ").unwrap_or_else(|| panic!("{f} places its box"));
            assert!(place > fill, "{f}: placed after it is filled (its width is known then)");
        }
        // right of the hairline when it fits, else left of it, never outside the plot (600 px plot; 8 px from the hairline)
        let src = format!("{}\nconst at = (w, tw, f) => {{ const t = {{ offsetWidth: tw, style: {{}} }}; placeTip({{ clientWidth: w }}, t, f); return t.style.left; }};\nconsole.log(JSON.stringify([at(600, 200, 0.1), at(600, 200, 0.95), at(600, 200, 1), at(600, 300, 0.5), at(600, 700, 0.5)]));", page_function("placeTip"));
        let Some(out) = node(&src) else { eprintln!("no node here"); return };
        assert_eq!(out, r#"["68px","362px","392px","0px","0px"]"#);
    }

    #[test]
    fn a_legend_swatch_takes_no_style_meant_for_another_element() {
        // a swatch class that is also a rule of its own takes that rule's padding and margin (the banner's `.bar` would draw a key as a block)
        let page = include_str!("dashboard.html");
        let first = |r: &str| -> String { r.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_').collect() };
        let own: Vec<String> = page.lines().filter_map(|l| l.strip_prefix('.')).map(first).collect();
        let keys: Vec<String> = page.lines().filter_map(|l| l.strip_prefix(".legend i.")).map(first).collect();
        for k in &keys { assert!(!own.contains(k), "the swatch class `{k}` is also a rule of its own"); }
        // the three keys are line keys of the base rule (three lines on one scale), with no class of their own
        let f = page_function("cacheChart");
        assert!(f.contains("const key = (c, text) => h('span', null, h('i', { style: `background: ${c}` }), text);"));
    }

    #[test]
    fn the_conversations_tile_says_saved_and_the_tokens_lists_are_in_its_hover() {
        let page = include_str!("dashboard.html");
        assert!(page.contains("`${whole.format(Number(cv.on_disk))} saved to SSD`"));
        assert!(page.contains("each Spark keeps its quarter"));
        assert!(page.contains("a returning conversation takes it back at once, with no read"));
        assert!(page.contains("A conversation counts as saved from its first saved piece."));
        assert!(page.contains("Their token lists take ${one.format(Number(m.ids_host_bytes) / 2 ** 20)} MiB in the server on Spark 1"));
        assert!(page.contains("conversations' token lists are in files"), "the files are named only when there are some");
    }

    #[test]
    fn the_queue_setting_is_named_for_what_it_does() {
        let page = include_str!("dashboard.html");
        assert!(page.contains("'push_rows', 'JUMP THE QUEUE UP TO', 'While a long prompt is being processed, a prompt with at most this many tokens to compute does not wait behind it: it is processed first, then the long one continues. One at a time. A larger prompt waits its turn.'"));
    }

    #[test]
    fn each_request_reports_its_own_progress_and_cached_part() {
        let t = Telemetry::new();
        let meta = |p| ReqMeta { key_label: "api".into(), ablit: false, prompt_tokens: p, endpoint: "/v1/chat/completions", priority: "normal" };
        let a = t.register(1, meta(9000)); // 1,000 positions reused, 8,000 to compute
        let b = t.register(2, meta(5000)); // prefilling beside it (two long prompts at once)
        let c = t.register(3, meta(3000)); // the whole prompt reused: no PROGRESS, straight to its tokens
        let _d = t.register(4, meta(700)); // still queued
        t.progress(&a, 4096, 8000);
        t.progress(&b, 2048, 5000);
        t.tokens(&c, 1);
        let j = requests_json(&t);
        let row = |id: u64| j["in_flight"].as_array().unwrap().iter().find(|x| x["id"] == id).unwrap().clone();
        let fields = |r: Value| (r["state"].as_str().map(String::from), r["computed"].as_u64(), r["to_compute"].as_u64(), r["cached_tokens"].as_u64());
        assert_eq!(fields(row(1)), (Some("prefill".into()), Some(4096), Some(8000), Some(1000)));
        assert_eq!(fields(row(2)), (Some("prefill".into()), Some(2048), Some(5000), Some(0)), "two prompts read at once are both prefill");
        assert_eq!(fields(row(3)), (Some("decode".into()), Some(0), Some(0), Some(3000)), "a whole prompt reused sends no PROGRESS");
        assert!(row(4)["state"] == "queued" && row(4)["cached_tokens"].is_null(), "unknown while queued");
    }

    #[test]
    fn requests_move_through_their_states() {
        let t = Telemetry::new();
        let meta = |l: &str, ablit| ReqMeta { key_label: l.into(), ablit, prompt_tokens: 9000, endpoint: "/v1/chat/completions", priority: "normal" };
        let a = t.register(1, meta("api", false));
        let b = t.register(2, meta("alice", true));
        let j = requests_json(&t);
        assert_eq!(j["in_flight"].as_array().unwrap().len(), 2);
        assert_eq!(j["in_flight"][0]["state"], "queued");
        t.progress(&a, 4096, 9000);
        t.progress(&a, 8192, 9000);
        assert_eq!(t.prefill_tokens.load(Ordering::Relaxed), 8192, "progress counts the change");
        assert_eq!((a.state(), a.kv_tokens(), b.state()), ("prefill", 8192, "queued"));
        t.tokens(&a, 3);
        t.tokens(&a, 2);
        assert_eq!((a.state(), a.kv_tokens()), ("decode", 9005));
        assert_eq!(t.decode_ids.load(Ordering::Relaxed), 5);
        assert_eq!(t.generated_tokens(), 5, "the GENERATED box's total: every output token since the front-end started");
        let j = requests_json(&t);
        let first = &j["in_flight"][0];
        assert_eq!((first["id"].as_u64(), first["generated_tokens"].as_u64(), first["key_label"].as_str(), first["ablit"].as_bool()), (Some(1), Some(5), Some("api"), Some(false)));
        let done = |id, reason, queue_us, prefill_us| Done {
            req_id: id,
            reason,
            prompt_tokens: 9000,
            cached_tokens: 100,
            completion_tokens: 5,
            queue_us,
            prefill_us,
            decode_us: 7,
            error: String::new(),
            tail: None,
        };
        t.finish(&done(1, DoneReason::Stop, 500_000, 1_250_000));
        t.finish(&done(2, DoneReason::Error, 1, 1));
        t.finish(&done(3, DoneReason::PrefixGone, 1, 1));
        let j = requests_json(&t);
        assert_eq!(j["in_flight"].as_array().unwrap().len(), 0);
        let r = j["recent"].as_array().unwrap();
        assert_eq!(r.len(), 1, "an error is not a finished request");
        assert_eq!((r[0]["ttft_s"].as_f64(), r[0]["cached_tokens"].as_u64(), r[0]["ablit"].as_bool()), (Some(1.75), Some(100), Some(false)));
        for i in 0..300 {
            t.finish(&done(100 + i, DoneReason::Length, 0, 1));
        }
        assert_eq!(requests_json(&t)["recent"].as_array().unwrap().len(), RECENT);
    }

    #[test]
    fn samples_become_node_health_and_the_ledger() {
        let t = Telemetry::new();
        *t.node_addrs.lock().unwrap() = vec!["a".into(), "b".into(), "c".into()];
        t.store_node(0, mock_node_json(0, 1.0));
        t.store_node(2, mock_node_json(2, 1.0));
        let h = node_health_json(&t);
        let ranks = h["ranks"].as_array().unwrap();
        assert_eq!(ranks.len(), 3);
        assert!(ranks[0]["age_ms"].is_u64() && ranks[1]["age_ms"].is_null() && ranks[2]["age_ms"].is_u64());
        // a reading keeps its own time on every call, so a page polling twice per reading sees it once
        assert!(ranks[0]["at_ms"].is_u64() && ranks[1]["at_ms"].is_null());
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(node_health_json(&t)["ranks"][0]["at_ms"], ranks[0]["at_ms"]);
        t.store_node(0, mock_node_json(0, 1.0));
        assert!(node_health_json(&t)["ranks"][0]["at_ms"].as_u64() > ranks[0]["at_ms"].as_u64());
        assert_eq!(ranks[2]["rank"], 2);
        assert_eq!(h["zone_names"][0], "cpu-thermal");
        let l = ledger_json(&t);
        assert_eq!(l["ranks"].as_array().unwrap().len(), 2);
        let e = l["entries"].as_array().unwrap();
        assert_eq!(e.len(), 6);
        let gib = 1u64 << 30;
        assert_eq!(e[0], json!({"rank": 0, "feature_id": "weights", "device_bytes": (74.2 * gib as f64) as u64}));
        assert_eq!(e[1]["feature_id"], "store");
        assert_eq!(e[2]["feature_id"], "work");
    }

    #[tokio::test]
    async fn a_sampler_is_asked_with_get() {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = l.local_addr().unwrap().port();
        tokio::spawn(async move {
            use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
            let (s, _) = l.accept().await.unwrap();
            let (rd, mut wr) = s.into_split();
            let mut cmd = String::new();
            BufReader::new(rd).read_line(&mut cmd).await.unwrap();
            assert_eq!(cmd, "GET\n");
            wr.write_all(format!("{}\n", mock_node_json(1, 0.0)).as_bytes()).await.unwrap();
        });
        let v = sample(&format!("127.0.0.1:{port}")).await.unwrap();
        assert_eq!(v["rank"], 1);
        assert!(v["gpu"]["temp_c"].is_number() && v["nvme"]["temp_c"].is_number() && v["rails"][1]["tx_bytes"].is_u64());
    }

    // ---- the two qualifiers, the running lines' idle hold and the context cache's fill (a level beside the rates)

    #[test]
    fn the_ssd_and_lent_out_qualifiers_fit_one_line_at_their_longest_values() {
        let src = format!("const whole = new Intl.NumberFormat('en-AU', {{ maximumFractionDigits: 0 }});\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
            page_line("const one ="), page_line("const none ="), page_line("const gib ="), page_line("const gibQ ="), page_function("lentQual"), page_line("const ssdQual ="),
            "const G = 2 ** 30, gb = 3 * G, gq = G / 4;\nconsole.log(JSON.stringify([ssdQual(4, 512 * G), ssdQual(4, 0), ssdQual(4, 2000 * G), lentQual(4, gb, [0, 1, 0, 1]), lentQual(4, gb, [0, 1, 0, 0.3]), lentQual(4, gb, [1, 1, 0, 1]), lentQual(4, gq, [18, 0.7, 0, 1.4]), lentQual(4, gb, [0, 0.83, 0, 0.8]), [gibQ(9.94), gibQ(9.96), gibQ(10), gibQ(123.6)], lentQual(4, gq, [99, 9.9, 0, 9.9]), lentQual(4, gq, [99, 0.9, 0, 0.4]), lentQual(4, gq, [18, 0.7, 0, 0])]));");
        let Some(out) = node(&src) else { eprintln!("no node here: the qualifiers are checked where node runs"); return };
        let v: Value = serde_json::from_str(&out).unwrap();
        let q = |i: usize| v[i].as_str().unwrap().to_string();
        // SSD CACHE: no per cent (its value already reads of its size), the GiB the Sparks hold together
        assert_eq!((q(0).as_str(), q(1).as_str(), q(2).as_str()), ("2,048.0 GiB on all 4", "0.0 GiB on all 4", "8,000.0 GiB on all 4"));
        // LENT OUT: whole GiB from 10 up, one decimal below, while the line fits the tile (31 characters fit it at 1920 and
        // 1100 px, 33 wrapped at 1920 px; a phone's two tiles a row may take two lines); then whole GiB from 1 up; then '<1' for a loan under 1 GiB
        assert_eq!(q(3), "Engram 12 · images 12");
        assert_eq!(q(4), "Engram 12 · images 3.6");
        assert_eq!(q(5), "Engram 12 · images 12 · long 12");
        assert_eq!(q(6), "Engram 0.7 · images 1 · long 18", "a served run's loans: one decimal would take 33 characters");
        assert_eq!(q(7), "Engram 10 · images 9.6", "what would round to 10.0 is shown whole");
        assert_eq!(v[8], json!(["9.9", "10", "10", "124"]));
        assert_eq!(q(9), "Engram 10 · images 10 · long 99", "the longest the pool can lend");
        assert_eq!(q(10), "Engram <1 · images <1 · long 99");
        assert_eq!(q(11), "Engram 0.7 · images 0 · long 18", "nothing lent is 0");
        for i in (0..8).chain(9..12) { assert!(q(i).chars().count() <= 31, "one line: {}", q(i)); assert!(!q(i).contains('%'), "no per cent: {}", q(i)); }
    }

    /// Runs `cacheRunning` over the points and prints the three values (and the last request's index) of the points asked for.
    fn running_at(points: Vec<Value>, at: &[usize]) -> Option<Value> {
        let src = format!("{}\nconst r = cacheRunning({});\nconsole.log(JSON.stringify({}.map((i) => [r[i].memory, r[i].ssd, r[i].engram, r[i].lastReq])));", cache_page_fns(&["cacheRunning"]), Value::Array(points), json!(at));
        node(&src).map(|o| serde_json::from_str(&o).unwrap())
    }

    #[test]
    fn an_idle_engine_holds_the_running_lines_still_and_they_never_vanish() {
        // 20 buckets of traffic (the share in memory alternating 90 % and 50 %, Engram 80 % and 60 %), then 300 quiet recorded buckets
        let mut pts = Vec::new();
        for k in 0..20 { pts.push(if k % 2 == 0 { pt(90, 0, 10, 100, 80) } else { pt(50, 25, 25, 100, 60) }); }
        for _ in 0..300 { pts.push(pt(0, 0, 0, 0, 0)); }
        let at: Vec<usize> = (19..320).collect();
        let Some(v) = running_at(pts, &at) else { eprintln!("no node here"); return };
        let rows = v.as_array().unwrap();
        assert!(rows[0][0].is_number() && rows[0][1].is_number() && rows[0][2].is_number(), "the last bucket with traffic has all three lines");
        for (i, r) in rows.iter().enumerate() {
            assert_eq!(r.as_array().unwrap()[..3], rows[0].as_array().unwrap()[..3], "bucket {}: flat through the quiet time, not drifting, not gone", 19 + i);
            assert_eq!(r[3], json!(19), "the last request stays the last bucket with a prompt");
        }
    }

    #[test]
    fn before_any_traffic_there_is_no_line_and_the_first_traffic_starts_it() {
        // quiet buckets, a bucket with no record, then the first prompt: no line before it (a quiet bucket is not 0 %), the line from it on
        let mut pts = vec![pt(0, 0, 0, 0, 0), pt(0, 0, 0, 0, 0), Value::Null, pt(0, 0, 0, 40, 30), pt(80, 0, 20, 0, 0), pt(0, 0, 0, 0, 0)];
        pts.push(Value::Null);
        pts.push(pt(0, 0, 0, 0, 0));
        let Some(v) = running_at(pts, &[0, 1, 2, 3, 4, 5, 6, 7]) else { eprintln!("no node here"); return };
        assert_eq!(v, json!([[null, null, null, null], [null, null, null, null], [null, null, null, null],
                             [null, null, 75, null], [80, 0, 75, 4], [80, 0, 75, 4], [null, null, null, 4], [80, 0, 75, 4]]),
            "Engram starts with its first lookup and the others with their first prompt; no record is null and the values come back after it");
    }

    #[test]
    fn the_running_window_is_the_last_90_buckets_with_traffic_however_long_ago() {
        // 100 buckets of prompts (the first 10 found nowhere, the other 90 in memory), then 200 quiet ones; and 100 buckets of lookups (the first 10 missed)
        let mut pts = Vec::new();
        for k in 0..100 { pts.push(if k < 10 { pt(0, 0, 100, 10, 0) } else { pt(100, 0, 0, 10, 10) }); }
        for _ in 0..200 { pts.push(pt(0, 0, 0, 0, 0)); }
        let Some(v) = running_at(pts, &[9, 99, 299]) else { eprintln!("no node here"); return };
        assert_eq!(v, json!([[0, 0, 0, 9], [100, 0, 100, 99], [100, 0, 100, 99]]),
            "the first 10 left the window by traffic, not by the clock: 100 % from bucket 99 on, through 200 quiet buckets");
    }

    #[test]
    fn the_page_asks_for_the_buckets_with_traffic_from_before_the_series_and_says_when_the_last_request_was() {
        let f = page_function("cacheChart");
        assert!(f.contains("const before = body.before || [], all = before.concat(body.points);") && f.contains("cacheRunning(all).slice(-CACHE_WIN)"), "the window of traffic reaches back past the series");
        assert!(f.contains("hm.format(new Date(endOf(r.lastReq) - body.step_s * 1000))") && f.contains("state.lastAt[i]"), "the hover has the time of the last request");
    }

    #[test]
    fn traffic_that_left_the_series_is_still_in_the_answer_before_its_first_point() {
        let mut c = CacheSeries::default();
        // a prompt in bucket 100 and Engram lookups in bucket 101, then the front-end runs on for two hours: both are long out of the 180 buckets
        c.add_prompts(1002 * S, 1, 50_000, 40_000);
        c.add_stats(1011 * S, cc(0, 0, 0, 0, 0, 0));
        c.add_stats(1015 * S, cc(100, 90, 0, 0, 0, 0));
        for k in 0..720_u64 { c.touch((1100 + 10 * k) * S); }
        let now = (1100 + 10 * 719 + 5) * S;
        let j = cache_answer_json(&c.query(now, 1800, 10), 10);
        let before = j["before"].as_array().expect("the answer has the buckets with traffic from before its points");
        assert_eq!(before.len(), 2);
        assert_eq!((before[0]["prompt_tokens"].as_u64(), before[0]["end_ms"].as_u64()), (Some(50_000), Some(1010 * S)), "the prompt's bucket, with the end of its 10 seconds");
        assert_eq!((before[1]["lookups"].as_u64(), before[1]["hits"].as_u64(), before[1]["end_ms"].as_u64()), (Some(100), Some(90), Some(1020 * S)));
        assert!(j["points"].as_array().unwrap().iter().all(|p| p["prompt_tokens"].as_u64() == Some(0) && p["lookups"].as_u64() == Some(0)), "the points themselves are quiet");
        // traffic inside the series is not in `before`; and only a window of each kind is kept
        let mut d = CacheSeries::default();
        for k in 0..300_u64 { d.add_prompts((1000 + 10 * k) * S, 1, 1000, 500); }
        let j = cache_answer_json(&d.query((1000 + 10 * 299 + 5) * S, 1800, 10), 10);
        assert_eq!(j["points"].as_array().unwrap().len(), 180);
        assert_eq!(j["before"].as_array().unwrap().len(), 90, "the newest 90 buckets with prompts from before the points, and no more");
        let short = cache_answer_json(&d.query((1000 + 10 * 299 + 5) * S, 900, 10), 10);
        assert_eq!((short["points"].as_array().unwrap().len(), short["before"].as_array().unwrap().len()), (90, 90), "a shorter series has its window before it too");
    }

    #[test]
    fn the_context_cache_fill_is_each_buckets_last_reading_of_the_pool() {
        let t = Telemetry::new();
        let pool = |held: u64, free: u64| EngineStats { pool_held: held, pool_free: free, number_bytes: 2_916_352, ranks: 4, ..Default::default() };
        // bucket 100: two readings, the last counts; bucket 101: none; bucket 102: one; the engine reported no pool in bucket 103
        t.set_stats_at(1001 * S, pool(500, 8000));
        t.set_stats_at(1008 * S, pool(913, 7600));
        t.cache.lock().unwrap().touch(1011 * S);
        t.set_stats_at(1021 * S, pool(2000, 6500));
        t.set_stats_at(1031 * S, EngineStats::default());
        let j = cache_answer_json(&t.cache_series(1035 * S, 40, 10), 10);
        let p = j["points"].as_array().unwrap();
        assert_eq!(p[0]["pool"], json!({"held": 913, "total": 8513, "chunk_bytes": 4 * 2_916_352}), "held and free are the tile's numbers: the total is both; a chunk is the Sparks' together");
        assert_eq!(p[1]["pool"], Value::Null, "no reading in that bucket: no field, the page keeps the level before");
        assert_eq!(p[2]["pool"]["held"], 2000);
        assert_eq!(p[3]["pool"], Value::Null, "no pool reported is no reading, never an empty pool");
        // the page's figures from it are the CONTEXT CACHE tile's: per cent of the chunks taken, and the GiB of the pool's GiB
        let src = format!("{}\nconst f = cacheFill({});\nconst x = (v) => Math.round(v * 100) / 100;\nconsole.log(JSON.stringify(f.map((q) => q && [x(q.pct), gib(q.taken), gib(q.total)])));", cache_page_fns(&["cacheFill"]), j["points"]);
        let Some(out) = node(&src) else { eprintln!("no node here"); return };
        assert_eq!(out, r#"[[10.72,"9.9","92.5"],[10.72,"9.9","92.5"],[23.53,"21.7","92.3"],[23.53,"21.7","92.3"]]"#);
    }

    #[test]
    fn the_fill_is_a_level_it_holds_between_readings_and_breaks_where_there_is_no_record() {
        let q = |held: u64| json!({"held": held, "total": 1000, "chunk_bytes": 1 << 30});
        let pts = json!([{"pool": q(100)}, {"pool": null}, null, {"pool": null}, {"pool": q(250)}, {"pool": {"held": 5, "total": 0, "chunk_bytes": 1}}]);
        let src = format!("{}\n{}\nconsole.log(JSON.stringify([cacheFill({pts}).map((f) => f && f.pct), stepPath([[0, 50], [100, 50], [200, 20], [300, 20]]), stepPath([[0, 10]])]));", cache_page_fns(&["cacheFill"]), page_function("stepPath"));
        let Some(out) = node(&src) else { eprintln!("no node here"); return };
        assert_eq!(out, r#"[[10,10,null,null,25,25],"M0.0,50.0 H100.0 V50.0 H200.0 V20.0 H300.0 V20.0","M0.0,10.0"]"#,
            "a gap in the reading keeps the level; no record is null; an empty pool total is no reading; a level is stepped (horizontal, then vertical)");
    }

    #[test]
    fn the_cache_chart_draws_the_fill_stepped_on_the_left_scale_with_its_own_key_and_hover() {
        let page = include_str!("dashboard.html");
        let f = page_function("cacheChart");
        // its own colour (the next hue of the page's order, validated) and its own look: a dashed step, never one of the running lines' smooth strokes
        assert!(f.contains("fill: '#c4508f'") && page.contains("#c4508f"));
        assert!(f.contains("fillLine.setAttribute('stroke-dasharray', '5 3')") && f.contains("stepPath(run)"), "dashed and stepped");
        // the same left scale as the lines: a fill of 100 % is the plot's top, never the bars' token scale
        assert!(f.contains("vals.forEach((p, i) => { if (p == null) { flush(); return; } run.push([frac(i, vals.length) * 600, plotY(p / 100)]); });"));
        assert!(f.contains("const lev = steps(fill.map((f) => (f ? f.pct : null)));") && f.contains("fillLine.setAttribute('d', lev.line)"));
        // its legend entry between the Engram key and the bars' key
        assert!(f.contains("'context cache full %'") && f.contains("key(COLOR.engram, 'Engram %'), fillKey, barKey"));
        // the hover: the swatch of the fourth row is the fill's, and the row is made by cacheRows from the bucket's level
        assert!(f.contains("[COLOR.memory, COLOR.ssd, COLOR.engram, COLOR.fill]") && f.contains("cacheRows(state.run[i], state.pts[i], state.fill[i], state.lastAt[i])"));
        assert!(f.contains("const fill = cacheFill(body.points).slice(-CACHE_WIN);"), "the level of each point drawn, from the series' points");
        assert!(page_function("cacheRows").contains("`context cache ${pc(fill.pct)} · ${gib(fill.taken)} of ${gib(fill.total)} GiB`"));
        // the colour is checked with the data-viz validator in the report (dark, the plot's surface); here: it is not one of the other series' colours
        for other in ["#3aa888", "#3f7fc4", "#c8782c", "#5a6165"] { assert_ne!(other, "#c4508f"); }
    }
}
