//! Frames of the front-end ↔ rank-0 protocol (docs/protocol.md, engine/common/frontend_protocol.h).
//! An 8-byte header `{u32 size; u16 type; u16 reserved}`, then `size` payload bytes: a fixed
//! little-endian struct followed by arrays.

use tokio::io::{AsyncRead, AsyncReadExt};

pub const VERSION: u32 = 4;
pub const DEFAULT_SOCKET: &str = "/tmp/dsgb10-engine.sock";
/// Up to 100 image spans, plus room for a 1M-token prompt and framing.
pub const MAX_FRAME: usize = crate::vision::MAX_IMAGES * crate::vision::MAX_IMAGE_ROWS * crate::vision::IMAGE_DIM * 2 + (8 << 20);

/// A SUBMIT's fields before stop_ids[] (docs/protocol.md).
const SUBMIT_FIXED: usize = 72;

pub const T_HELLO: u16 = 0x01;
pub const T_SUBMIT: u16 = 0x02;
pub const T_CANCEL: u16 = 0x03;
pub const T_RELEASE: u16 = 0x04;
pub const T_STATS_REQ: u16 = 0x05;
pub const T_CONFIG: u16 = 0x06;
pub const T_HELLO_ENGINE: u16 = 0x81;
pub const T_TOKENS: u16 = 0x82;
pub const T_DONE: u16 = 0x83;
pub const T_PROGRESS: u16 = 0x84;
pub const T_STATS: u16 = 0x85;

pub const FLAG_SEED: u32 = 1;
pub const FLAG_KEEP: u32 = 2;
pub const FLAG_LOGPROBS: u32 = 4;
/// vision: the image part (vision::payload) follows prompt[]
pub const FLAG_IMAGES: u32 = 8;
/// the ablated weights path: the request runs with the ablation direction applied
pub const FLAG_ABLIT: u32 = 16;

pub const TOK_LOGPROBS: u32 = 1;
pub const TOK_TOP: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoneReason {
    Stop = 0,
    Length = 1,
    Cancelled = 2,
    Context = 3,
    Error = 4,
}

impl DoneReason {
    pub fn from_u32(v: u32) -> DoneReason {
        match v {
            0 => DoneReason::Stop,
            1 => DoneReason::Length,
            2 => DoneReason::Cancelled,
            3 => DoneReason::Context,
            _ => DoneReason::Error,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Submit {
    pub req_id: u64,
    pub session_id: u64,
    pub base_session_id: u64,
    pub seed: u64,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub max_new_tokens: u32,
    pub flags: u32,
    pub n_top_logprobs: u32,
    pub prefix_hint: u32,
    pub stop_ids: Vec<u32>,
    pub prompt: Vec<u32>,
    /// vision: the image part after prompt[] (with FLAG_IMAGES); None for text
    pub images: Option<Vec<u8>>,
}

impl Submit {
    /// The image part's spans, (start, rows, hash) each, checked against the prompt: every span lies
    /// within it, on image placeholders, in order and without overlap. Empty for a text request.
    pub fn image_spans(&self) -> Result<Vec<(u32, u32, u64)>, String> {
        let Some(part) = &self.images else { return Ok(Vec::new()) };
        let n = u32::from_le_bytes(part.get(..4).ok_or("image part too short")?.try_into().unwrap()) as usize;
        let table = part.get(4..4 + 16 * n).ok_or("image part too short")?;
        let mut spans = Vec::with_capacity(n);
        let mut end = 0usize;
        for t in table.chunks_exact(16) {
            let start = u32::from_le_bytes(t[..4].try_into().unwrap());
            let rows = u32::from_le_bytes(t[4..8].try_into().unwrap());
            let hash = u64::from_le_bytes(t[8..].try_into().unwrap());
            let (a, b) = (start as usize, start as usize + rows as usize);
            if a < end || b > self.prompt.len() || self.prompt[a..b].iter().any(|&id| id != crate::vision::IMAGE_TOKEN_ID) {
                return Err(format!("image span {start}+{rows} is not on image placeholders of the prompt"));
            }
            end = b;
            spans.push((start, rows, hash));
        }
        Ok(spans)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HelloEngine {
    pub version: u32,
    pub max_context: u32,
    pub vocab_size: u32,
    pub eos_id: u32,
    pub max_sessions: u32,
    /// positions of conversation the engine can keep (its pool and disk tier): the session table keeps as much
    pub keep_positions: u64,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tokens {
    pub req_id: u64,
    pub ids: Vec<u32>,
    pub logprobs: Option<Vec<f32>>,
    /// n_top (id, logprob) pairs per token
    pub top: Option<Vec<(u32, f32)>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Done {
    pub req_id: u64,
    pub reason: DoneReason,
    pub prompt_tokens: u32,
    pub cached_tokens: u32,
    pub completion_tokens: u32,
    pub queue_us: u64,
    pub prefill_us: u64,
    pub decode_us: u64,
    pub error: String,
}

/// The fields of a STATS frame, in wire order (docs/protocol.md, "The dashboard's frames"); a frame of another version is refused.
pub const STATS_VERSION: u32 = 2;
pub const STATS_FIELDS: usize = 31;

/// The engine's own counters for the dashboard: the chunk pool and its conversations (rank 0's; every rank's are the same),
/// the disk tier, interleaved prefill. Counters count from the engine's start.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EngineStats {
    /// the chunk pool: its numbers (each one slice of every cache array on each rank)
    pub pool_numbers: u64,
    /// of them no conversation names
    pub pool_free: u64,
    /// named by a conversation's caches or bookmarks
    pub pool_held: u64,
    /// of those, named by more than one (shared beginnings)
    pub pool_shared: u64,
    /// kept for the running requests' horizons, not yet held
    pub pool_reserved: u64,
    /// one number's bytes on each rank
    pub number_bytes: u64,
    /// positions of conversation the whole pool holds
    pub pool_positions: u64,
    /// conversations held by a line
    pub conv_on_lines: u64,
    /// idle, their chunks in the pool
    pub conv_in_memory: u64,
    /// in the disk tier
    pub conv_on_disk: u64,
    /// naming at least one shared number
    pub conv_sharing: u64,
    /// sequences the engine decodes at once
    pub lines: u64,
    pub disk_cap_bytes: u64,
    pub disk_used_bytes: u64,
    pub spills_total: u64,
    pub spill_bytes_total: u64,
    pub spills_pending: u64,
    pub restores_total: u64,
    pub restore_bytes_total: u64,
    pub restored_tokens_total: u64,
    pub restore_wait_ns_total: u64,
    pub restore_wait_max_ns: u64,
    /// interleaved prefill: the settings the engine runs now (ConfigFrame), and the long prompts being processed and waiting
    pub il_on: u64,
    pub il_whole_rows: u64,
    pub il_push_rows: u64,
    pub il_gap_ms: u64,
    pub il_share: u64,
    pub il_burst_max_ms: u64,
    pub il_reading: u64,
    pub il_queued: u64,
    /// a conversation unused this many days is dropped (CONFIG keep_days)
    pub keep_days: u64,
}

impl EngineStats {
    fn fields(&self) -> [u64; STATS_FIELDS] {
        [
            self.pool_numbers,
            self.pool_free,
            self.pool_held,
            self.pool_shared,
            self.pool_reserved,
            self.number_bytes,
            self.pool_positions,
            self.conv_on_lines,
            self.conv_in_memory,
            self.conv_on_disk,
            self.conv_sharing,
            self.lines,
            self.disk_cap_bytes,
            self.disk_used_bytes,
            self.spills_total,
            self.spill_bytes_total,
            self.spills_pending,
            self.restores_total,
            self.restore_bytes_total,
            self.restored_tokens_total,
            self.restore_wait_ns_total,
            self.restore_wait_max_ns,
            self.il_on,
            self.il_whole_rows,
            self.il_push_rows,
            self.il_gap_ms,
            self.il_share,
            self.il_burst_max_ms,
            self.il_reading,
            self.il_queued,
            self.keep_days,
            self.copy_adapt,
            self.copy_proposals_total,
            self.copy_scored_total,
            self.copy_first_right_total,
            self.copy_rows_verified_total,
            self.copy_rows_accepted_total,
            self.copy_chance_pm[0],
            self.copy_chance_pm[1],
            self.copy_chance_pm[2],
            self.copy_chance_pm[3],
            self.copy_failing_requests_total,
        ]
    }

    /// Fields beyond those given are 0; extra ones are ignored.
    fn from_fields(v: &[u64]) -> EngineStats {
        let f = |i: usize| v.get(i).copied().unwrap_or(0);
        EngineStats {
            pool_numbers: f(0),
            pool_free: f(1),
            pool_held: f(2),
            pool_shared: f(3),
            pool_reserved: f(4),
            number_bytes: f(5),
            pool_positions: f(6),
            conv_on_lines: f(7),
            conv_in_memory: f(8),
            conv_on_disk: f(9),
            conv_sharing: f(10),
            lines: f(11),
            disk_cap_bytes: f(12),
            disk_used_bytes: f(13),
            spills_total: f(14),
            spill_bytes_total: f(15),
            spills_pending: f(16),
            restores_total: f(17),
            restore_bytes_total: f(18),
            restored_tokens_total: f(19),
            restore_wait_ns_total: f(20),
            restore_wait_max_ns: f(21),
            il_on: f(22),
            il_whole_rows: f(23),
            il_push_rows: f(24),
            il_gap_ms: f(25),
            il_share: f(26),
            il_burst_max_ms: f(27),
            il_reading: f(28),
            il_queued: f(29),
            keep_days: f(30),
        }
    }
}

/// Interleaved prefill's settings for rank 0 (docs/protocol.md, CONFIG): a field < 0 keeps the engine's value; the engine
/// takes each one in its flag's range and ignores the others; its next STATS shows what it runs.
pub const CONFIG_VERSION: u32 = 2;
pub const CONFIG_FIELDS: u32 = 8;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConfigFrame {
    pub on: i64,
    pub whole_rows: i64,
    pub push_rows: i64,
    pub gap_ms: i64,
    pub share: i64,
    pub burst_max_ms: i64,
    /// the disk tier's size in GiB (at most the file the engine made at its start; smaller evicts to fit)
    pub ssd_gib: i64,
    /// a conversation unused this many days is dropped
    pub keep_days: i64,
}

impl Default for ConfigFrame {
    fn default() -> Self {
        ConfigFrame { on: -1, whole_rows: -1, push_rows: -1, gap_ms: -1, share: -1, burst_max_ms: -1, ssd_gib: -1, keep_days: -1 }
    }
}

impl ConfigFrame {
    fn fields(&self) -> [i64; CONFIG_FIELDS as usize] {
        [self.on, self.whole_rows, self.push_rows, self.gap_ms, self.share, self.burst_max_ms, self.ssd_gib, self.keep_days]
    }
}

/// A decoded frame, either direction.
#[derive(Clone, Debug, PartialEq)]
pub enum Frame {
    Hello { version: u32 },
    Submit(Submit),
    Cancel { req_id: u64 },
    Release { session_id: u64 },
    HelloEngine(HelloEngine),
    Tokens(Tokens),
    Done(Done),
    /// Prefill progress of one request: prompt tokens computed so far, of those it has to compute.
    Progress { req_id: u64, computed: u32, total: u32 },
    /// Front-end to rank 0: send one STATS frame.
    StatsReq,
    /// Front-end to rank 0: interleaved prefill's settings, at once (the dashboard's live settings).
    Config(ConfigFrame),
    Stats(EngineStats),
}

// ------------------------------------------------------------------ encoding

struct W(Vec<u8>);

impl W {
    fn new(ty: u16) -> W {
        let mut v = Vec::with_capacity(64);
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&ty.to_le_bytes());
        v.extend_from_slice(&0u16.to_le_bytes());
        W(v)
    }
    fn u32(&mut self, x: u32) -> &mut Self {
        self.0.extend_from_slice(&x.to_le_bytes());
        self
    }
    fn u64(&mut self, x: u64) -> &mut Self {
        self.0.extend_from_slice(&x.to_le_bytes());
        self
    }
    fn f32(&mut self, x: f32) -> &mut Self {
        self.0.extend_from_slice(&x.to_le_bytes());
        self
    }
    fn u32s(&mut self, xs: &[u32]) -> &mut Self {
        self.0.reserve(xs.len() * 4);
        for x in xs {
            self.0.extend_from_slice(&x.to_le_bytes());
        }
        self
    }
    fn finish(mut self) -> Vec<u8> {
        let size = (self.0.len() - 8) as u32;
        self.0[..4].copy_from_slice(&size.to_le_bytes());
        self.0
    }
}

impl Frame {
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Frame::Hello { version } => {
                let mut w = W::new(T_HELLO);
                w.u32(*version).u32(0);
                w.finish()
            }
            Frame::Submit(s) => {
                let mut w = W::new(T_SUBMIT);
                // the whole frame at once: growing by doubling would hold up to twice an image part
                w.0.reserve_exact(SUBMIT_FIXED + 4 * (s.stop_ids.len() + s.prompt.len()) + s.images.as_ref().map_or(0, Vec::len));
                w.u64(s.req_id).u64(s.session_id).u64(s.base_session_id).u64(s.seed);
                w.f32(s.temperature).f32(s.top_p).u32(s.top_k).u32(s.max_new_tokens);
                w.u32(s.flags).u32(s.n_top_logprobs).u32(s.prefix_hint);
                w.u32(s.stop_ids.len() as u32).u32(s.prompt.len() as u32).u32(0);
                w.u32s(&s.stop_ids).u32s(&s.prompt);
                if let Some(img) = &s.images {
                    w.0.extend_from_slice(img);
                }
                w.finish()
            }
            Frame::Cancel { req_id } => {
                let mut w = W::new(T_CANCEL);
                w.u64(*req_id);
                w.finish()
            }
            Frame::Release { session_id } => {
                let mut w = W::new(T_RELEASE);
                w.u64(*session_id);
                w.finish()
            }
            Frame::Progress { req_id, computed, total } => {
                let mut w = W::new(T_PROGRESS);
                w.u64(*req_id).u32(*computed).u32(*total);
                w.finish()
            }
            Frame::StatsReq => {
                let mut w = W::new(T_STATS_REQ);
                w.u32(0).u32(0);
                w.finish()
            }
            Frame::Config(c) => {
                let mut w = W::new(T_CONFIG);
                w.u32(CONFIG_VERSION).u32(CONFIG_FIELDS);
                for x in c.fields() {
                    w.u64(x as u64);
                }
                w.finish()
            }
            Frame::Stats(st) => {
                let mut w = W::new(T_STATS);
                w.u32(STATS_VERSION).u32(STATS_FIELDS as u32);
                for x in st.fields() {
                    w.u64(x);
                }
                w.finish()
            }
            Frame::HelloEngine(h) => {
                let mut w = W::new(T_HELLO_ENGINE);
                w.u32(h.version).u32(h.max_context).u32(h.vocab_size).u32(h.eos_id).u32(h.max_sessions);
                w.u32(h.name.len() as u32).u64(h.keep_positions);
                w.0.extend_from_slice(h.name.as_bytes());
                w.finish()
            }
            Frame::Tokens(t) => {
                let mut w = W::new(T_TOKENS);
                let flags = if t.logprobs.is_some() { TOK_LOGPROBS } else { 0 } | if t.top.is_some() { TOK_TOP } else { 0 };
                w.u64(t.req_id).u32(t.ids.len() as u32).u32(flags).u32s(&t.ids);
                if let Some(lp) = &t.logprobs {
                    for x in lp {
                        w.f32(*x);
                    }
                }
                if let Some(top) = &t.top {
                    for (id, lp) in top {
                        w.u32(*id).f32(*lp);
                    }
                }
                w.finish()
            }
            Frame::Done(d) => {
                let mut w = W::new(T_DONE);
                w.u64(d.req_id).u32(d.reason as u32).u32(d.prompt_tokens).u32(d.cached_tokens).u32(d.completion_tokens);
                w.u64(d.queue_us).u64(d.prefill_us).u64(d.decode_us).u32(d.error.len() as u32).u32(0);
                w.0.extend_from_slice(d.error.as_bytes());
                w.finish()
            }
        }
    }

    /// Decode one frame's payload. `n_top` is the request's n_top_logprobs, needed to size the
    /// top-logprob array of a TOKENS frame (pass 0 if it is not known; such arrays are then rejected).
    pub fn decode(ty: u16, p: &[u8], n_top: impl Fn(u64) -> u32) -> Result<Frame, String> {
        let mut r = R { p, i: 0 };
        let f = match ty {
            T_HELLO => {
                let version = r.u32()?;
                r.u32()?;
                Frame::Hello { version }
            }
            T_SUBMIT => {
                let req_id = r.u64()?;
                let session_id = r.u64()?;
                let base_session_id = r.u64()?;
                let seed = r.u64()?;
                let temperature = r.f32()?;
                let top_p = r.f32()?;
                let top_k = r.u32()?;
                let max_new_tokens = r.u32()?;
                let flags = r.u32()?;
                let n_top_logprobs = r.u32()?;
                let prefix_hint = r.u32()?;
                let n_stop = r.u32()? as usize;
                let n_prompt = r.u32()? as usize;
                r.u32()?;
                let stop_ids = r.u32s(n_stop)?;
                let prompt = r.u32s(n_prompt)?;
                // vision: the image part, taken only when it is laid out as rank 0 requires it
                // (engine/server/serve.cpp): u32 n, n spans, then exactly the spans' rows
                let images = if flags & FLAG_IMAGES != 0 {
                    let part = &r.p[r.i..];
                    let n = r.u32()? as usize;
                    if !(1..=crate::vision::MAX_IMAGES).contains(&n) {
                        return Err(format!("SUBMIT with {n} images"));
                    }
                    let mut rows = 0usize;
                    for _ in 0..n {
                        r.u32()?; // start
                        let span = r.u32()? as usize;
                        r.u64()?; // hash
                        if !(3..=crate::vision::MAX_IMAGE_ROWS).contains(&span) {
                            return Err(format!("SUBMIT with an image span of {span} rows"));
                        }
                        rows += span;
                    }
                    r.bytes(rows * crate::vision::IMAGE_DIM * 2).map_err(|_| "SUBMIT: the image rows do not match the spans".to_string())?;
                    Some(part[..part.len() - (r.p.len() - r.i)].to_vec())
                } else {
                    None
                };
                Frame::Submit(Submit {
                    req_id,
                    session_id,
                    base_session_id,
                    seed,
                    temperature,
                    top_p,
                    top_k,
                    max_new_tokens,
                    flags,
                    n_top_logprobs,
                    prefix_hint,
                    stop_ids,
                    prompt,
                    images,
                })
            }
            T_CANCEL => Frame::Cancel { req_id: r.u64()? },
            T_RELEASE => Frame::Release { session_id: r.u64()? },
            T_HELLO_ENGINE => {
                let version = r.u32()?;
                let max_context = r.u32()?;
                let vocab_size = r.u32()?;
                let eos_id = r.u32()?;
                let max_sessions = r.u32()?;
                let n = r.u32()? as usize;
                let keep_positions = r.u64()?;
                let name = String::from_utf8_lossy(r.bytes(n)?).into_owned();
                Frame::HelloEngine(HelloEngine { version, max_context, vocab_size, eos_id, max_sessions, keep_positions, name })
            }
            T_TOKENS => {
                let req_id = r.u64()?;
                let n = r.u32()? as usize;
                let flags = r.u32()?;
                let ids = r.u32s(n)?;
                let logprobs = if flags & TOK_LOGPROBS != 0 { Some((0..n).map(|_| r.f32()).collect::<Result<Vec<_>, _>>()?) } else { None };
                let top = if flags & TOK_TOP != 0 {
                    let k = n_top(req_id) as usize;
                    Some((0..n * k).map(|_| Ok((r.u32()?, r.f32()?))).collect::<Result<Vec<_>, String>>()?)
                } else {
                    None
                };
                Frame::Tokens(Tokens { req_id, ids, logprobs, top })
            }
            T_DONE => {
                let req_id = r.u64()?;
                let reason = DoneReason::from_u32(r.u32()?);
                let prompt_tokens = r.u32()?;
                let cached_tokens = r.u32()?;
                let completion_tokens = r.u32()?;
                let queue_us = r.u64()?;
                let prefill_us = r.u64()?;
                let decode_us = r.u64()?;
                let n = r.u32()? as usize;
                r.u32()?;
                let error = String::from_utf8_lossy(r.bytes(n)?).into_owned();
                Frame::Done(Done { req_id, reason, prompt_tokens, cached_tokens, completion_tokens, queue_us, prefill_us, decode_us, error })
            }
            T_PROGRESS => Frame::Progress { req_id: r.u64()?, computed: r.u32()?, total: r.u32()? },
            T_STATS_REQ => {
                r.u32()?;
                r.u32()?;
                Frame::StatsReq
            }
            T_CONFIG => {
                let version = r.u32()?;
                let n = r.u32()? as usize;
                if version != CONFIG_VERSION || n > 64 {
                    return Err(format!("CONFIG version {version}, {n} fields"));
                }
                let v = (0..n).map(|_| r.u64().map(|x| x as i64)).collect::<Result<Vec<_>, _>>()?;
                let f = |i: usize| v.get(i).copied().unwrap_or(-1);
                Frame::Config(ConfigFrame { on: f(0), whole_rows: f(1), push_rows: f(2), gap_ms: f(3), share: f(4), burst_max_ms: f(5), ssd_gib: f(6), keep_days: f(7) })
            }
            T_STATS => {
                let version = r.u32()?;
                let n = r.u32()? as usize;
                if version != STATS_VERSION {
                    return Err(format!("STATS version {version}, expected {STATS_VERSION}"));
                }
                if n > 4096 {
                    return Err(format!("STATS with {n} fields"));
                }
                let v = (0..n).map(|_| r.u64()).collect::<Result<Vec<_>, _>>()?;
                Frame::Stats(EngineStats::from_fields(&v))
            }
            _ => return Err(format!("unknown frame type {ty:#x}")),
        };
        if r.i != p.len() {
            return Err(format!("frame type {ty:#x}: {} trailing bytes", p.len() - r.i));
        }
        Ok(f)
    }
}

struct R<'a> {
    p: &'a [u8],
    i: usize,
}

impl<'a> R<'a> {
    fn bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.p.len() - self.i < n {
            return Err("truncated frame".into());
        }
        let b = &self.p[self.i..self.i + n];
        self.i += n;
        Ok(b)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn u32s(&mut self, n: usize) -> Result<Vec<u32>, String> {
        let b = self.bytes(n.checked_mul(4).ok_or("bad length")?)?;
        Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
    }
}

/// Read one frame: (type, payload). Ok(None) on a clean end of stream between frames.
pub async fn read_frame<Rd: AsyncRead + Unpin>(rd: &mut Rd) -> std::io::Result<Option<(u16, Vec<u8>)>> {
    let mut h = [0u8; 8];
    match rd.read_exact(&mut h).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let size = u32::from_le_bytes(h[..4].try_into().unwrap()) as usize;
    let ty = u16::from_le_bytes(h[4..6].try_into().unwrap());
    if size > MAX_FRAME {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("frame of {size} bytes")));
    }
    let mut p = vec![0u8; size];
    rd.read_exact(&mut p).await?;
    Ok(Some((ty, p)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_submit_with_images_is_one_allocation() {
        let img = vec![7u8; 4 + 16 + 3 * crate::vision::IMAGE_DIM * 2];
        let s = Submit {
            req_id: 1,
            session_id: 2,
            base_session_id: 0,
            seed: 0,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            max_new_tokens: 8,
            flags: FLAG_IMAGES,
            n_top_logprobs: 0,
            prefix_hint: 0,
            stop_ids: vec![1, 2],
            prompt: vec![5; 1000],
            images: Some(img.clone()),
        };
        let b = Frame::Submit(s).encode();
        let want = 8 + SUBMIT_FIXED + 4 * 2 + 4 * 1000 + img.len();
        assert_eq!((b.len(), b.capacity()), (want, want), "sized once, not grown by doubling");
        assert_eq!(u32::from_le_bytes(b[..4].try_into().unwrap()) as usize, want - 8);
        assert!(b[want - img.len()..] == img[..], "the image part follows prompt[] unchanged");
    }

    /// The mock engine takes a SUBMIT's image part under the checks rank 0 applies (serve.cpp).
    #[test]
    fn image_part_roundtrip_and_checks() {
        use crate::vision::{Encoded, IMAGE_DIM, IMAGE_TOKEN_ID};
        let rows = vec![3u32, 4];
        let n = 7 * IMAGE_DIM * 2;
        let mut data = vec![0u8; 4 + 16 * 2 + n];
        data[4 + 16 * 2..].iter_mut().enumerate().for_each(|(i, b)| *b = (i % 251) as u8);
        let part = crate::vision::payload(&[1, 5], Encoded { rows, hashes: vec![11, 22], data, ms: 0.0 });
        let mut prompt = vec![0u32; 10];
        prompt[1..4].fill(IMAGE_TOKEN_ID);
        prompt[5..9].fill(IMAGE_TOKEN_ID);
        let s = Submit {
            req_id: 1,
            session_id: 2,
            base_session_id: 0,
            seed: 0,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            max_new_tokens: 8,
            flags: FLAG_IMAGES,
            n_top_logprobs: 0,
            prefix_hint: 0,
            stop_ids: vec![1],
            prompt,
            images: Some(part.clone()),
        };
        let b = Frame::Submit(s.clone()).encode();
        assert_eq!(Frame::decode(T_SUBMIT, &b[8..], |_| 0).unwrap(), Frame::Submit(s.clone()));
        assert_eq!(s.image_spans().unwrap(), vec![(1, 3, 11), (5, 4, 22)]);
        // what rank 0 refuses, the mock refuses: a row short, a row over, no images, a span out of range
        let with = |part: Vec<u8>| Frame::decode(T_SUBMIT, &Frame::Submit(Submit { images: Some(part), ..s.clone() }).encode()[8..], |_| 0);
        assert!(with(part[..part.len() - 2].to_vec()).is_err());
        assert!(with([part.clone(), vec![0, 0]].concat()).is_err());
        let mut none = part.clone();
        none[..4].copy_from_slice(&0u32.to_le_bytes());
        assert!(with(none).is_err());
        let mut short = part.clone();
        short[8..12].copy_from_slice(&2u32.to_le_bytes());
        assert!(with(short).is_err());
        // a text SUBMIT must not carry extra bytes, as before
        let mut text = Frame::Submit(Submit { flags: 0, images: None, ..s.clone() }).encode();
        text.extend_from_slice(&[0; 4]);
        assert!(Frame::decode(T_SUBMIT, &text[8..], |_| 0).is_err());
        // spans must sit on image placeholders of the prompt
        let mut off = s.clone();
        off.prompt[1] = 0;
        assert!(off.image_spans().is_err());
    }

    #[test]
    fn roundtrip_and_layout() {
        let s = Submit {
            req_id: 7,
            session_id: 9,
            base_session_id: 3,
            seed: 42,
            temperature: 0.5,
            top_p: 0.95,
            top_k: 20,
            max_new_tokens: 100,
            flags: FLAG_SEED | FLAG_KEEP | FLAG_ABLIT,
            n_top_logprobs: 0,
            prefix_hint: 5,
            stop_ids: vec![1],
            prompt: vec![0, 10, 20],
            images: None,
        };
        let b = Frame::Submit(s.clone()).encode();
        assert_eq!(b.len(), 8 + 72 + 4 + 12); // header, SubmitHeader, stop id, prompt
        assert_eq!(u32::from_le_bytes(b[8 + 64..8 + 68].try_into().unwrap()), 3); // n_prompt at offset 64
        assert_eq!(Frame::decode(T_SUBMIT, &b[8..], |_| 0).unwrap(), Frame::Submit(s));
        let d = Done {
            req_id: 1,
            reason: DoneReason::Length,
            prompt_tokens: 3,
            cached_tokens: 2,
            completion_tokens: 4,
            queue_us: 5,
            prefill_us: 6,
            decode_us: 7,
            error: "x".into(),
        };
        let b = Frame::Done(d.clone()).encode();
        assert_eq!(b.len(), 8 + 56 + 1);
        assert_eq!(Frame::decode(T_DONE, &b[8..], |_| 0).unwrap(), Frame::Done(d));
        // the dashboard's frames
        let b = Frame::Progress { req_id: 9, computed: 4096, total: 20000 }.encode();
        assert_eq!(b.len(), 8 + 16);
        assert_eq!(u16::from_le_bytes(b[4..6].try_into().unwrap()), 0x84);
        assert_eq!(Frame::decode(T_PROGRESS, &b[8..], |_| 0).unwrap(), Frame::Progress { req_id: 9, computed: 4096, total: 20000 });
        let b = Frame::StatsReq.encode();
        assert_eq!((b.len(), u16::from_le_bytes(b[4..6].try_into().unwrap())), (8 + 8, 0x05));
        assert_eq!(Frame::decode(T_STATS_REQ, &b[8..], |_| 0).unwrap(), Frame::StatsReq);
        let st = EngineStats { pool_numbers: 8913, conv_on_disk: 3, restore_wait_max_ns: 7, conv_sharing: 2, lines: 32, ..Default::default() };
        let b = Frame::Stats(st).encode();
        assert_eq!(b.len(), 8 + 8 + 8 * 31);
        assert_eq!(u32::from_le_bytes(b[8..12].try_into().unwrap()), 2, "version 2");
        assert_eq!(u32::from_le_bytes(b[12..16].try_into().unwrap()), 31, "n_fields after the version");
        assert_eq!(u64::from_le_bytes(b[16..24].try_into().unwrap()), 8913, "pool_numbers is the first field");
        assert_eq!(u64::from_le_bytes(b[16 + 8 * 11..16 + 8 * 12].try_into().unwrap()), 32, "lines is the 12th");
        let il = EngineStats { il_share: 40, il_queued: 2, keep_days: 30, ..Default::default() };
        let bi = Frame::Stats(il).encode();
        assert_eq!(u64::from_le_bytes(bi[bi.len() - 8..].try_into().unwrap()), 30, "keep_days is the last");
        assert_eq!(u64::from_le_bytes(bi[bi.len() - 16..bi.len() - 8].try_into().unwrap()), 2, "il_queued before it");
        assert_eq!(u64::from_le_bytes(bi[16 + 8 * 26..16 + 8 * 27].try_into().unwrap()), 40, "il_share is the 27th");
        assert_eq!(Frame::decode(T_STATS, &bi[8..], |_| 0).unwrap(), Frame::Stats(il));
        assert_eq!(Frame::decode(T_STATS, &b[8..], |_| 0).unwrap(), Frame::Stats(st));
        // a shorter or a longer field list: missing fields are 0, extra ones ignored
        let mut short = Frame::Stats(st).encode();
        short.truncate(8 + 8 + 8 * 4);
        short[12..16].copy_from_slice(&4u32.to_le_bytes());
        let Frame::Stats(got) = Frame::decode(T_STATS, &short[8..], |_| 0).unwrap() else { panic!() };
        assert_eq!((got.pool_numbers, got.conv_on_disk, got.lines), (8913, 0, 0));
        let mut long = Frame::Stats(st).encode();
        long.extend(5u64.to_le_bytes());
        long[12..16].copy_from_slice(&32u32.to_le_bytes());
        assert_eq!(Frame::decode(T_STATS, &long[8..], |_| 0).unwrap(), Frame::Stats(st));
        long[8..12].copy_from_slice(&1u32.to_le_bytes());
        assert!(Frame::decode(T_STATS, &long[8..], |_| 0).is_err(), "another version");
        // HELLO from the engine (version 4): the C struct's 32 bytes, keep_positions after name_len, then the name
        let he = HelloEngine { version: VERSION, max_context: 1 << 20, vocab_size: 129280, eos_id: 1, max_sessions: 16,
                               keep_positions: 21_004_288, name: "pico".into() };
        let b = Frame::HelloEngine(he.clone()).encode();
        assert_eq!((VERSION, b.len()), (4, 8 + 32 + 4));
        assert_eq!(u32::from_le_bytes(b[8 + 20..8 + 24].try_into().unwrap()), 4, "name_len");
        assert_eq!(u64::from_le_bytes(b[8 + 24..8 + 32].try_into().unwrap()), 21_004_288, "keep_positions at byte 24");
        assert_eq!(&b[8 + 32..], b"pico");
        assert_eq!(Frame::decode(T_HELLO_ENGINE, &b[8..], |_| 0).unwrap(), Frame::HelloEngine(he));
        // CONFIG: version, n, then i64 fields; a field not sent is -1 (kept)
        let c = ConfigFrame { share: 40, gap_ms: 600, ssd_gib: 128, keep_days: 14, ..Default::default() };
        let b = Frame::Config(c).encode();
        assert_eq!((b.len(), u16::from_le_bytes(b[4..6].try_into().unwrap())), (8 + 8 + 8 * 8, 0x06));
        assert_eq!(u32::from_le_bytes(b[8..12].try_into().unwrap()), 2, "CONFIG version 2");
        assert_eq!(i64::from_le_bytes(b[16..24].try_into().unwrap()), -1, "on: kept");
        assert_eq!(i64::from_le_bytes(b[16 + 8 * 4..16 + 8 * 5].try_into().unwrap()), 40, "share is the 5th");
        assert_eq!(i64::from_le_bytes(b[16 + 8 * 6..16 + 8 * 7].try_into().unwrap()), 128, "ssd_gib is the 7th");
        assert_eq!(i64::from_le_bytes(b[16 + 8 * 7..16 + 8 * 8].try_into().unwrap()), 14, "keep_days is the 8th");
        assert_eq!(Frame::decode(T_CONFIG, &b[8..], |_| 0).unwrap(), Frame::Config(c));
        let mut short = Frame::Config(c).encode();
        short.truncate(8 + 8 + 8 * 2);
        short[12..16].copy_from_slice(&2u32.to_le_bytes());
        assert_eq!(Frame::decode(T_CONFIG, &short[8..], |_| 0).unwrap(), Frame::Config(ConfigFrame::default()));
        let t = Tokens { req_id: 2, ids: vec![5, 6], logprobs: Some(vec![-0.5, -1.0]), top: Some(vec![(5, -0.5), (6, -1.0)]) };
        let b = Frame::Tokens(t.clone()).encode();
        assert_eq!(Frame::decode(T_TOKENS, &b[8..], |_| 1).unwrap(), Frame::Tokens(t));
    }
}
