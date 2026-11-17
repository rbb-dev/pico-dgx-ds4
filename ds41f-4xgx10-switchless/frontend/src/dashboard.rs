//! The dashboard's data: what the front-end sees of the engine (requests, token and prefill counters, the STATS frame)
//! and what the samplers report, served under /dashboard (docs/dashboard.md). The token path only bumps atomics or
//! takes a lock for a few instructions; rates, series and JSON are made by the 1 Hz task or a handler.

use crate::protocol::{ConfigFrame, Done, DoneReason, EngineStats};
use crate::server::App;
use axum::extract::State;
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
/// The finished requests the page's TTFT figures are taken from.
const RECENT: usize = 200;
/// The dashboard counts as watched for this long after a dashboard request; only then is the engine
/// asked for STATS and are the samplers polled.
const WATCH_S: u64 = 10;
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

    /// `n` ids of a request arrived. Atomics only.
    pub fn tokens(&self, live: &ReqLive, n: usize) {
        self.decode_ids.fetch_add(n as u64, Ordering::Relaxed);
        live.generated.fetch_add(n as u32, Ordering::Relaxed);
        live.decoding.store(true, Ordering::Relaxed);
    }

    /// A prefill chunk of a request finished: `computed` of `total` prompt tokens. Atomics only.
    pub fn progress(&self, live: &ReqLive, computed: u32, total: u32) {
        let before = live.computed.swap(computed, Ordering::Relaxed);
        live.total.store(total, Ordering::Relaxed);
        live.progressed.store(true, Ordering::Relaxed);
        self.prefill_tokens.fetch_add(computed.saturating_sub(before) as u64, Ordering::Relaxed);
    }

    /// A request ended (an error is not a finished request for the TTFT figures).
    pub fn finish(&self, d: &Done) {
        let live = self.live.lock().unwrap().remove(&d.req_id);
        if d.reason == DoneReason::Error {
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
        *self.stats.lock().unwrap() = Some((Instant::now(), s));
    }

    pub fn stats(&self) -> Option<EngineStats> {
        self.stats.lock().unwrap().map(|(_, s)| s)
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

/// The engine's sessions as the page's card reads them: its fixed slots (a request holds one for its whole context;
/// `slots_big` of them take up to `ctx_big` positions, the rest up to `ctx_small`), and the session store a finished
/// request's state moves to (its own memory on each rank, never counted against the slots).
pub fn sessions_json(s: &EngineStats, usable_window: u32) -> Value {
    let positions = s.slots_big.saturating_mul(s.slot_ctx_big).saturating_add(s.slots_total.saturating_sub(s.slots_big).saturating_mul(s.slot_ctx_small));
    json!({"slots": s.slots_total, "slots_big": s.slots_big, "ctx_big": s.slot_ctx_big, "ctx_small": s.slot_ctx_small,
           "positions": positions, "usable_window": usable_window,
           "store_used_bytes": s.store_used_bytes, "store_cap_bytes": s.store_cap_bytes, "sessions_resident": s.sessions_resident})
}

/// The SSD tier as the page's card reads it: `spill` (its state) and `prompt_cache` (its counters).
pub fn tier_json(s: Option<&EngineStats>) -> (Value, Value) {
    let Some(s) = s else { return (json!({"state": "unknown"}), Value::Null) };
    let state = if s.disk_cap_bytes > 0 { "on" } else { "off" };
    let cache = json!({
        "spill": [{"stored_bytes": s.disk_used_bytes, "cap_bytes": s.disk_cap_bytes, "points": s.sessions_on_disk,
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
                 "prefill_tokens_per_second": r1(pre), "prompt_pass": pass},
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
           "engine": {"connected": info.is_some(), "name": info.as_ref().map(|i| i.name.clone()), "uptime_s": t.started.elapsed().as_secs()},
           "health": health, "sessions": kv, "node_health": node_health_json(t),
           "interleave": settings_json(stats.as_ref())})
}

/// Interleaved prefill's settings as the engine runs them (its last STATS), and the long prompts being processed and waiting
pub fn settings_json(s: Option<&EngineStats>) -> Value {
    match s {
        Some(s) if s.il_share > 0 => json!({"on": s.il_on == 1, "whole_rows": s.il_whole_rows, "push_rows": s.il_push_rows,
                                            "gap_ms": s.il_gap_ms, "share": s.il_share, "burst_max_ms": s.il_burst_max_ms,
                                            "reading": s.il_reading, "queued": s.il_queued}),
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
            _ => return Err(format!("{k}: not a setting (on, share, gap_ms, burst_max_ms, whole_rows, push_rows)")),
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
    match t.series(name, unix_ms(), seconds, step) {
        Some(a) => {
            let step_s = (step.max(BUCKET_S) / BUCKET_S).min(BUCKETS) * BUCKET_S;
            json_response(StatusCode::OK, json!({"points": a.points, "peaks": a.peaks, "step_s": step_s, "last_ms": a.last_ms}))
        }
        None => failure(StatusCode::BAD_REQUEST, "name is decode_tps or prefill_tps"),
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
        for bad in [json!({"share": 95}), json!({"share": 5}), json!({"gap_ms": 50}), json!({"whole_rows": 4096}),
                    json!({"whole_rows": 2_000_000}), json!({"share": "40"}), json!({"on": 1}), json!({"speed": 1}),
                    json!({}), json!([1])] {
            assert!(parse_settings(&bad, 1 << 20).is_err(), "{bad} must be refused");
        }
    }

    #[test]
    fn settings_json_reads_the_stats() {
        assert_eq!(settings_json(None), Value::Null);
        assert_eq!(settings_json(Some(&EngineStats::default())), Value::Null, "an engine without interleave");
        let s = EngineStats { il_on: 1, il_share: 40, il_gap_ms: 600, il_reading: 1, il_queued: 2, ..Default::default() };
        let v = settings_json(Some(&s));
        assert_eq!((v["on"].as_bool(), v["share"].as_u64(), v["queued"].as_u64()), (Some(true), Some(40), Some(2)));
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

    fn stats() -> EngineStats {
        EngineStats {
            store_cap_bytes: 890 * 1_000_000,
            disk_cap_bytes: 256 << 30,
            disk_used_bytes: 5 << 30,
            sessions_on_disk: 41,
            spills_total: 400,
            spill_bytes_total: 9 << 30,
            spills_pending: 2,
            restores_total: 10,
            restored_tokens_total: 123456,
            restore_wait_ns_total: 50_000_000,
            restore_wait_max_ns: 9_000_000,
            slots_total: 16,
            slots_big: 2,
            slot_ctx_big: 400_000,
            slot_ctx_small: 14_285,
            ..Default::default()
        }
    }

    #[test]
    fn stats_map_to_the_sessions_and_the_ssd_tier() {
        let s = EngineStats { store_used_bytes: 300 * 1_000_000, sessions_resident: 7, ..stats() };
        let ss = sessions_json(&s, 1_048_576);
        let positions = 2 * 400_000 + 14 * 14_285;
        assert_eq!((ss["slots"].as_u64(), ss["slots_big"].as_u64()), (Some(16), Some(2)));
        assert_eq!((ss["ctx_big"].as_u64(), ss["ctx_small"].as_u64()), (Some(400_000), Some(14_285)));
        assert_eq!(ss["positions"], positions);
        assert_eq!(ss["usable_window"], 1_048_576);
        // the session store is its own memory: its bytes are never spread over the slots' positions
        assert_eq!((ss["store_used_bytes"].as_u64(), ss["store_cap_bytes"].as_u64(), ss["sessions_resident"].as_u64()), (Some(300_000_000), Some(890_000_000), Some(7)));
        assert!(ss.get("bytes_per_token").is_none());
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
        assert_eq!(sessions_json(&EngineStats::default(), 5)["positions"], 0);
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
}
