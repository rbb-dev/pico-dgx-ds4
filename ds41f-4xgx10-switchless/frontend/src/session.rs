//! Session matching for exact prefix reuse (docs/protocol.md, "Sessions and exact prefix reuse").
//!
//! A session keeps cut points (the places where tokenisation restarts), each as (bytes of text before it, ids before it, a hash of
//! that text), and the epoch and history size the engine last reported. A request is matched by its text to the session that shares its
//! latest cut, of its own weights path (`ablit`), YaRN factor (`factor_for`) and images (`ImageKey`); the ids before that cut are the
//! engine's own, so only the text after it is tokenised and sent, with the epoch that vouches for the cut.
//!
//! The table outlives a stop (`save`, `load`; docs/engine.md, "Kept sessions"): the engine keeps its
//! sessions across a planned stop, and the table is what finds them again. It holds hashes and numbers, never a conversation's
//! text, and its file is ours alone all the same: mode 0600 in a directory of mode 0700, written whole or not at all.

use crate::protocol::DoneTail;
use crate::tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// Bytes of shared text below which a match is not worth a session (a BOS and a few tokens).
const MIN_SHARED: usize = 64;
/// A request continues a session in place when the shared text reaches this close to the end of
/// the session's prompt; the encoding may rewrite the tail of the prompt (`<think>` → `</think>`).
const PROMPT_SLACK: usize = 16;

/// The window (in tokens) one YaRN factor covers per step: factor 16 holds 1M, 32 holds 2M, 64 holds 4M.
const FACTOR_STEP: usize = 1 << 20;
/// The factor a prompt of `n_tokens` ids is served at: the smallest of 16 / 32 / 64 whose window holds it with room
/// for the next token (the engine's `factor_for`: below 2^20 ids is 16, below 2^21 is 32, else 64).
pub fn factor_for(n_tokens: usize) -> u32 {
    if n_tokens < FACTOR_STEP {
        16
    } else if n_tokens < 2 * FACTOR_STEP {
        32
    } else {
        64
    }
}

/// One image of a prompt: where its run of placeholders starts in the prompt's text (a byte offset), how
/// many rows it spans, and a hash of the image's bytes (the encoder's).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageKey {
    pub at: usize,
    pub rows: u32,
    pub hash: u64,
}

/// Whether two prompts' images agree inside their first `n` bytes: the same images, in the same places.
fn images_agree(a: &[ImageKey], b: &[ImageKey], n: usize) -> bool {
    a.iter().filter(|k| k.at < n).eq(b.iter().filter(|k| k.at < n))
}

/// A place in a conversation where tokenisation restarts: the bytes of text before it, the ids the engine holds for that text, and
/// the hash of the text: the first 128 bits of its SHA-256. A session is continued on a match of this hash alone, so it must be one whose
/// collisions nobody can compute. Two texts that have a cut of the same bytes and hash share everything before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cut {
    pub bytes: usize,
    pub ids: u32,
    pub hash: u128,
}

impl Cut {
    /// Before everything: no text, no ids (the hash is never compared: no edge sits at byte 0).
    pub const START: Cut = Cut { bytes: 0, ids: 0, hash: 0 };
}

/// The SHA-256 state after the text up to a cut: what hashing the text after it goes on from.
#[derive(Clone)]
pub struct Resume(Sha256);

impl Resume {
    /// Before any text.
    pub fn start() -> Resume {
        Resume(Sha256::new())
    }
}

impl std::fmt::Debug for Resume {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Resume")
    }
}

/// The first 128 bits of the digest of what `h` has been fed.
fn digest128(h: &Sha256) -> u128 {
    let d = h.clone().finalize();
    u128::from_le_bytes(d[..16].try_into().unwrap())
}

/// Whether byte `p` of a prompt lies strictly inside one image's run of placeholders: no cut sits there.
fn in_image_run(images: &[ImageKey], p: usize) -> bool {
    images.iter().any(|k| p > k.at && p < k.at + k.rows as usize * crate::encoding::IMAGE_PLACEHOLDER.len())
}

/// A cut of a prompt that has no ids yet: where an added token starts or ends, and the hash of the text up to there.
#[derive(Clone)]
struct Edge {
    bytes: usize,
    hash: u128,
    /// the hash's state there, to go on from it at the request's DONE
    state: Sha256,
}

#[derive(Clone)]
struct Session {
    /// Ascending in bytes and in ids; the last is where the history the engine reported ends, or before it.
    cuts: Vec<Cut>,
    /// The bytes of the prompt of its last turn (where the output began).
    prompt_len: usize,
    busy: bool,
    last_used: u64,
    /// Unix time of its last request: a saved table leaves out what is older than its limit.
    t_used: u64,
    /// The ablated weights path: its state was computed with the ablation direction applied.
    ablit: bool,
    /// The YaRN factor its state was computed at (`factor_for` its prompt's ids when it was created).
    factor: u32,
    /// The images of its prompt, in order.
    images: Vec<ImageKey>,
    /// Its text's prefix keys (`scan`), as the index holds them.
    keys: Vec<u128>,
    /// The epoch the engine's last DONE gave for it (0: not known, a request cannot name it) and how many ids its history held
    /// then (0: no finished turn).
    epoch: u64,
    n_hist: u32,
}

impl Session {
    /// The memory the session holds: its buffers by capacity, not length, and its place in the table and the index.
    fn bytes(&self) -> usize {
        std::mem::size_of::<Session>()
            + SESSION_FIXED
            + std::mem::size_of::<Cut>() * self.cuts.capacity()
            + std::mem::size_of::<ImageKey>() * self.images.capacity()
            + 8 * self.keys.capacity()
            + INDEX_ENTRY * self.keys.len()
    }

    /// A record with no history, for a session that starts (a new conversation or a fork).
    fn empty(ablit: bool, factor: u32, clock: u64) -> Session {
        Session { cuts: Vec::new(), prompt_len: 0, busy: false, last_used: clock, t_used: 0, ablit, factor, images: Vec::new(), keys: Vec::new(), epoch: 0, n_hist: 0 }
    }

    /// Whether a request can continue it: its last turn finished and the engine reported its history.
    fn ready(&self) -> bool {
        self.n_hist > 0
    }
}

/// A prompt as the table needs it, made outside the table's lock: one pass over its text finds the hash at every edge of an added
/// token and at every index level.
pub struct Prompt<'a> {
    text: &'a str,
    ablit: bool,
    images: &'a [ImageKey],
    edges: Vec<Edge>,
    keys: Vec<u128>,
}

impl<'a> Prompt<'a> {
    pub fn new(text: &'a str, tok: &Tokenizer, ablit: bool, images: &'a [ImageKey]) -> Prompt<'a> {
        let (mut edges, keys) = scan(text, &Resume::start(), 0, tok);
        edges.retain(|e| !in_image_run(images, e.bytes));
        Prompt { text, ablit, images, edges, keys }
    }
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub session_id: u64,
    /// The session a new session may fork from (0 = none).
    pub base_session_id: u64,
    /// Where the request continues the entry the engine holds: the ids before it are the entry's own; `Cut::START` for a cold start.
    pub from: Cut,
    /// The epoch of that entry (0: none, or not known; a request then names no prefix and carries its whole prompt).
    pub epoch: u64,
    /// The hash state at `from`.
    pub state: Resume,
    /// The ids of the text after `from`.
    pub suffix: Vec<u32>,
    /// For the log, counts only: the session that shares the most with the prompt (0: none shares any) and the bytes of text it shares,
    /// at least (`SessionTable::pick`).
    pub matched: u64,
    pub shared: usize,
}

impl Plan {
    /// Whether the request carries only its suffix and names the entry it continues: the engine's protocol is the one that has the
    /// frame (`engine_version`), the prompt has no images (their spans are positions in the whole prompt) and the entry's epoch is
    /// known and there are ids before the cut to name.
    pub fn prefix_mode(&self, engine_version: u32, has_images: bool) -> bool {
        engine_version >= crate::protocol::VERSION_PREFIX && !has_images && self.epoch != 0 && self.from.ids > 0
    }

    /// The prompt's ids: the entry's before `from` and the suffix.
    pub fn total(&self) -> usize {
        self.from.ids as usize + self.suffix.len()
    }

    /// Where the request starts, for the log (no text, no ids): a continuation of a session, a
    /// fork of one (the engine starts from it if it holds it), or a new session that names none.
    pub fn describe(&self, prompt_bytes: usize) -> String {
        let of = format!("{} of {prompt_bytes} bytes", self.shared);
        if self.base_session_id != 0 {
            format!("session {}, a fork of {}: it shares {of}", self.session_id, self.base_session_id)
        } else if self.matched == self.session_id {
            format!("session {} in place: it shares {of}", self.session_id)
        } else if self.matched != 0 {
            format!("session {}, new: the best match, session {}, shares {} bytes (under {MIN_SHARED})", self.session_id, self.matched, self.shared)
        } else {
            format!("session {}, new: no session shares {MIN_SHARED} bytes", self.session_id)
        }
    }
}

/// What a finished (or dropped) request tells the table of its prompt.
pub struct Turn<'a> {
    /// The prompt's whole text.
    pub text: &'a str,
    /// The cut the request was planned from; `Cut::START` when it carried its whole prompt.
    pub from: Cut,
    /// The hash state at `from`.
    pub state: &'a Resume,
    /// The ids the request carried: the text after `from`, or the whole prompt's.
    pub suffix: &'a [u32],
    pub images: &'a [ImageKey],
}

pub struct SessionTable {
    sessions: HashMap<u64, Session>,
    /// The sessions by the hashes of their text's prefixes (`prefix_keys`): a prompt finds its best match among those
    /// sharing its longest indexed prefix, not by reading every session's text.
    index: HashMap<u128, Vec<u64>>,
    next_id: u64,
    clock: u64,
    /// Bumped by every change to the sessions held (a request planned or finished, a session
    /// dropped, a table loaded): what tells a periodic save that there is something new.
    changes: u64,
    max_sessions: usize,
    /// False until the engine's limits are applied (`keep_for`) on a table that `defer_limits` marked: the session and id counts
    /// bound nothing meanwhile, so a saved table loaded before the engine's HELLO keeps every session.
    limits_known: bool,
    max_tokens: usize,
    /// The most memory the sessions may hold (`Session::bytes`, with `image_bytes`).
    max_bytes: usize,
    /// The most memory the image rows in flight count for against `max_bytes`.
    image_cap: usize,
    /// What the image rows in flight count for now (at most `image_cap`).
    image_bytes: usize,
    /// Test control: `finish` appends the new cuts without first dropping those above the cut the request continued from.
    #[cfg(test)]
    append_without_truncating: bool,
}

/// The prefix lengths a text is indexed at: MIN_SHARED bytes, then four times longer each (64, 256, 1 KiB, ... 4 MiB at
/// level 8). A prompt and a session that share at least a level's length are in one bucket at that level.
fn level_len(level: usize) -> usize {
    MIN_SHARED << (2 * level)
}

/// The prefix key of index level `level` for a text whose first `level_len(level)` bytes hash to `h`.
fn level_key(level: usize, h: u128) -> u128 {
    h ^ (level as u128 + 1).wrapping_mul(0x9e3779b97f4a7c15f39cc0605cedc835)
}

/// One pass over `region`, the text that follows a cut of `start_bytes` bytes whose hash state is `start`: the edges of its added
/// tokens, as (bytes from the text's beginning, hash of the text up to there, the state there), and the prefix keys of the index levels
/// it reaches beyond the cut, in level order.
fn scan(region: &str, start: &Resume, start_bytes: usize, tok: &Tokenizer) -> (Vec<Edge>, Vec<u128>) {
    let mut at_edges: Vec<usize> = tok.added_spans(region).into_iter().flat_map(|(a, e)| [a, e]).filter(|&p| p > 0).collect();
    at_edges.sort_unstable();
    at_edges.dedup();
    let mut levels: Vec<(usize, usize)> = Vec::new(); // (bytes into the region, level)
    let mut level = (0..).find(|&l| level_len(l) > start_bytes).unwrap();
    while level_len(level) <= start_bytes + region.len() {
        levels.push((level_len(level) - start_bytes, level));
        level += 1;
    }
    let bytes = region.as_bytes();
    let (mut edges, mut keys) = (Vec::with_capacity(at_edges.len()), Vec::with_capacity(levels.len()));
    let (mut h, mut at, mut i, mut j) = (start.0.clone(), 0usize, 0usize, 0usize);
    loop {
        let next = match (at_edges.get(i), levels.get(j)) {
            (Some(&a), Some(&(b, _))) => a.min(b),
            (Some(&a), None) => a,
            (None, Some(&(b, _))) => b,
            (None, None) => break,
        };
        h.update(&bytes[at..next]);
        at = next;
        let d = digest128(&h);
        while at_edges.get(i) == Some(&next) {
            edges.push(Edge { bytes: start_bytes + next, hash: d, state: h.clone() });
            i += 1;
        }
        while j < levels.len() && levels[j].0 == next {
            keys.push(level_key(levels[j].1, d));
            j += 1;
        }
    }
    (edges, keys)
}

fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Layout `DSFT0004`: each session has its cuts, keys, epoch and history size, and no text and no ids (`write_session`).
const TABLE_MAGIC: &[u8; 8] = b"DSFT0004";
/// `DSFT0003`: each session has its text and ids, a flags word and then a factor word (u64 each); read once, the text and ids become cuts.
const TABLE_MAGIC_V3: &[u8; 8] = b"DSFT0003";
/// `DSFT0002`: a flags word, no factor word; read as factor 16.
const TABLE_MAGIC_V2: &[u8; 8] = b"DSFT0002";
/// `DSFT0001`: no flags and no factor word; read as the plain path at factor 16.
const TABLE_MAGIC_V1: &[u8; 8] = b"DSFT0001";
const TABLE_FLAG_ABLIT: u64 = 1;
/// The session's record ends with its images: u64 count, then (at, rows, hash) as three u64 each.
const TABLE_FLAG_IMAGES: u64 = 2;

const FNV_START: u64 = 1469598103934665603;

fn fnv_update(h: u64, b: &[u8]) -> u64 {
    b.iter().fold(h, |h, &x| (h ^ x as u64).wrapping_mul(1099511628211))
}

fn fnv(b: &[u8]) -> u64 {
    fnv_update(FNV_START, b)
}

/// Tokens between two marks (`Session::marks`).
const MARK_EVERY: usize = 32;

/// The marks of `ids` (`len_of` is a token's byte length) and the byte length of the whole text. A text longer than
/// u32::MAX bytes has no marks.
fn mark_ids(ids: &[u32], len_of: impl Fn(u32) -> usize) -> (Vec<u32>, usize) {
    let mut marks = Vec::with_capacity(ids.len() / MARK_EVERY);
    let (mut at, mut fits) = (0usize, true);
    for (i, &id) in ids.iter().enumerate() {
        at += len_of(id);
        if i % MARK_EVERY == MARK_EVERY - 1 {
            match u32::try_from(at) {
                Ok(m) => marks.push(m),
                Err(_) => fits = false,
            }
        }
    }
    if !fits {
        marks = Vec::new();
    }
    (marks, at)
}

/// The number of leading ids whose text is exactly `p` bytes long, when `p` is the end of a token; None otherwise (and
/// always None for a text of more than u32::MAX bytes, which has no marks).
fn token_end_in(marks: &[u32], ids: &[u32], text_len: usize, p: usize, len_of: impl Fn(u32) -> usize) -> Option<usize> {
    if text_len > u32::MAX as usize {
        return None;
    }
    let c = marks.partition_point(|&m| m as usize <= p);
    if c > 0 && marks[c - 1] as usize == p {
        return Some(c * MARK_EVERY);
    }
    let mut at = if c > 0 { marks[c - 1] as usize } else { 0 };
    for (i, &id) in ids.iter().enumerate().take((c + 1) * MARK_EVERY).skip(c * MARK_EVERY) {
        at += len_of(id);
        if at == p {
            return Some(i + 1);
        }
        if at > p {
            return None;
        }
    }
    None
}

/// Capacity a vector may keep over its length before it is given back: a quarter and a page.
fn slack_over(len: usize, cap: usize) -> bool {
    cap > len + len / 4 + 4096
}

fn shrink_slack<T>(v: &mut Vec<T>) {
    if slack_over(v.len(), v.capacity()) {
        v.shrink_to_fit();
    }
}

/// What a session costs besides its buffers: its struct, its slot in the table, the allocator's header on each of its
/// three buffers.
const SESSION_FIXED: usize = 64 + 16 * 3;
/// What each of a session's index keys costs: the index slot and the bucket's list.
const INDEX_ENTRY: usize = 112;

/// The directory is ours and nobody else's (made 0700 if it was ours but open to others).
fn private_dir(dir: &Path) -> Result<(), String> {
    let me = std::fs::metadata("/proc/self").map_err(|e| format!("/proc/self: {e}"))?.uid();
    let m = std::fs::symlink_metadata(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    if !m.is_dir() || m.uid() != me {
        return Err(format!("{} is not a directory of ours", dir.display()));
    }
    let abs = std::fs::canonicalize(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    if abs.ancestors().any(|a| a.join(".git").exists()) {
        return Err(format!("{} is in a git tree, where no conversation goes", dir.display()));
    }
    if m.mode() & 0o077 != 0 {
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700)).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    Ok(())
}

struct Bytes<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Bytes<'a> {
    fn u128(&mut self) -> Result<u128, String> {
        Ok(u128::from_le_bytes(self.take(16)?.try_into().unwrap()))
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.b.len() - self.at < n {
            return Err("cut short".into());
        }
        self.at += n;
        Ok(&self.b[self.at - n..self.at])
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

/// `bytes` (a table's `snapshot`) at `path`, whole or not at all: written to `path`.new (mode 0600), synced,
/// renamed over `path`, the directory synced. A crash in between leaves the previous file and at most a
/// `.new`, which a load ignores. The directory must be ours alone. One write at a time in the process.
pub fn write_table(path: &Path, bytes: &[u8]) -> Result<(), String> {
    static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    private_dir(path.parent().unwrap_or(Path::new(".")))?;
    let tmp = PathBuf::from(format!("{}.new", path.display()));
    let _ = std::fs::remove_file(&tmp);
    let write = || -> std::io::Result<()> {
        let mut o = std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&tmp)?;
        o.write_all(bytes)?;
        o.sync_all()?;
        drop(o);
        std::fs::rename(&tmp, path)?;
        if let Some(d) = path.parent() {
            std::fs::File::open(d)?.sync_all()?;
        }
        Ok(())
    };
    write().map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("{}: {e}", path.display())
    })
}

/// Whether a session is in a saved table: a session in a request is saved as its last finished turn left it (its cuts, epoch and
/// history size change only at `finish`); one with no finished turn, or unused for longer than `max_age_s`, is left out.
fn saved_in(s: &Session, now: u64, max_age_s: u64) -> bool {
    s.ready() && now.saturating_sub(s.t_used) <= max_age_s
}

/// One session's record in a saved table: u64 id, last_used, t_used, prompt_len, flags, factor, epoch, n_hist, n_keys, n_cuts; the
/// keys (16 bytes each); the cuts as (bytes u64, ids u64, hash 16 bytes); with `TABLE_FLAG_IMAGES` the images.
fn write_session(body: &mut Vec<u8>, id: u64, s: &Session) {
    let flags = (if s.ablit { TABLE_FLAG_ABLIT } else { 0 }) | (if s.images.is_empty() { 0 } else { TABLE_FLAG_IMAGES });
    for v in [id, s.last_used, s.t_used, s.prompt_len as u64, flags, s.factor as u64, s.epoch, s.n_hist as u64, s.keys.len() as u64, s.cuts.len() as u64] {
        body.extend(v.to_le_bytes());
    }
    for &k in &s.keys {
        body.extend(k.to_le_bytes());
    }
    for c in &s.cuts {
        body.extend((c.bytes as u64).to_le_bytes());
        body.extend((c.ids as u64).to_le_bytes());
        body.extend(c.hash.to_le_bytes());
    }
    if !s.images.is_empty() {
        body.extend((s.images.len() as u64).to_le_bytes());
        for k in &s.images {
            for v in [k.at as u64, k.rows as u64, k.hash] {
                body.extend(v.to_le_bytes());
            }
        }
    }
}

/// The images of a record: u64 count, then (at, rows, hash) as three u64 each.
fn read_images(r: &mut Bytes) -> Result<Vec<ImageKey>, String> {
    let n = r.u64()? as usize;
    if n.checked_mul(24).is_none_or(|b| b > r.b.len() - r.at) {
        return Err("a bad count".into());
    }
    let mut images = Vec::with_capacity(n);
    for _ in 0..n {
        let (at, rows, hash) = (r.u64()? as usize, r.u64()?, r.u64()?);
        images.push(ImageKey { at, rows: rows as u32, hash });
    }
    Ok(images)
}

/// One `DSFT0004` record (`write_session`); None when the session is not taken (too old, no history, a bad shape).
fn read_session(r: &mut Bytes, now: u64, max_age_s: u64) -> Result<Option<(u64, Session)>, String> {
    let (id, last_used, t_used, prompt_len, flags) = (r.u64()?, r.u64()?, r.u64()?, r.u64()? as usize, r.u64()?);
    let (factor, epoch, n_hist, n_keys, n_cuts) = (r.u64()?, r.u64()?, r.u64()?, r.u64()? as usize, r.u64()? as usize);
    if !matches!(factor, 16 | 32 | 64) {
        return Err("a bad factor".into());
    }
    if n_keys > 64 || n_hist > u32::MAX as u64 || n_cuts.checked_mul(32).is_none_or(|b| b > r.b.len() - r.at) {
        return Err("a bad count".into());
    }
    let keys = (0..n_keys).map(|_| r.u128()).collect::<Result<Vec<_>, _>>()?;
    let mut cuts = Vec::with_capacity(n_cuts);
    for _ in 0..n_cuts {
        let (bytes, ids, hash) = (r.u64()? as usize, r.u64()?, r.u128()?);
        cuts.push(Cut { bytes, ids: u32::try_from(ids).map_err(|_| "a bad cut")?, hash });
    }
    let images = if flags & TABLE_FLAG_IMAGES != 0 { read_images(r)? } else { Vec::new() };
    let shaped = id != 0 && n_hist > 0 && cuts.windows(2).all(|w| w[0].bytes < w[1].bytes && w[0].ids <= w[1].ids) && cuts.iter().all(|c| c.bytes > 0 && c.ids as u64 <= n_hist);
    if !shaped || now.saturating_sub(t_used) > max_age_s || images.iter().any(|k| k.rows < 3) {
        return Ok(None);
    }
    let ablit = flags & TABLE_FLAG_ABLIT != 0;
    Ok(Some((id, Session { cuts, prompt_len, busy: false, last_used, t_used, ablit, factor: factor as u32, images, keys, epoch, n_hist: n_hist as u32 })))
}

/// One record of a layout before `DSFT0004` (text and ids): its conversation's cuts are found in the text, which is then let go. The
/// epoch is not known (0): the session's first request carries its whole prompt, and the DONE teaches the epoch.
fn read_old_session(r: &mut Bytes, with_flags: bool, with_factor: bool, tok: &Tokenizer, now: u64, max_age_s: u64) -> Result<Option<(u64, Session)>, String> {
    let (id, last_used, t_used, prompt_len, text_len) = (r.u64()?, r.u64()?, r.u64()?, r.u64()? as usize, r.u64()? as usize);
    let flags = if with_flags { r.u64()? } else { 0 };
    let factor = if with_factor { r.u64()? } else { 16 };
    if !matches!(factor, 16 | 32 | 64) {
        return Err("a bad factor".into());
    }
    let text = r.take(text_len)?;
    let n_ids = r.u64()? as usize;
    let raw = r.take(n_ids.checked_mul(4).ok_or("a bad count")?)?;
    let ids: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let images = if flags & TABLE_FLAG_IMAGES != 0 { read_images(r)? } else { Vec::new() };
    let Ok(text) = std::str::from_utf8(text) else { return Ok(None) };
    let vocab = tok.vocab_size() as u32;
    if id == 0 || ids.is_empty() || now.saturating_sub(t_used) > max_age_s || prompt_len > text.len() || ids.iter().any(|&x| x >= vocab) {
        return Ok(None);
    }
    let (marks, at) = mark_ids(&ids, |x| tok.token_bytes(x).len());
    if at != text.len() || tok.decode_bytes(&ids) != text.as_bytes() {
        return Ok(None); // another tokenizer's ids
    }
    if images.iter().any(|k| k.at >= text.len() || k.rows < 3) {
        return Ok(None);
    }
    let (edges, keys) = scan(text, &Resume::start(), 0, tok);
    let cuts = edges
        .iter()
        .filter_map(|e| token_end_in(&marks, &ids, text.len(), e.bytes, |x| tok.token_bytes(x).len()).map(|n| Cut { bytes: e.bytes, ids: n as u32, hash: e.hash }))
        .collect();
    let ablit = flags & TABLE_FLAG_ABLIT != 0;
    Ok(Some((id, Session { cuts, prompt_len, busy: false, last_used, t_used, ablit, factor: factor as u32, images, keys, epoch: 0, n_hist: ids.len() as u32 })))
}

/// The 56 bytes that open a saved table.
fn table_header(written: u64, next_id: u64, clock: u64, n: u64, body_len: u64, hash: u64) -> [u8; 56] {
    let mut f = [0u8; 56];
    f[..8].copy_from_slice(TABLE_MAGIC);
    for (i, v) in [written, next_id, clock, n, body_len, hash].into_iter().enumerate() {
        f[8 + 8 * i..16 + 8 * i].copy_from_slice(&v.to_le_bytes());
    }
    f
}

/// Sessions are copied out of the table until their records reach this many bytes, which are then written with the lock
/// let go.
const SAVE_BATCH: usize = 2 << 20;

/// The sessions of `table` used within `max_age_s`, saved to `path` like `write_table` (same bytes as `snapshot`, same crash behaviour)
/// without the lock held across a disk write or the file in memory: the lock is taken for one batch of records at a time (`SAVE_BATCH`,
/// or one session when it is larger), which are written after it is let go, the header (its length and hash) last. A session changed
/// between batches is written as it is when its batch is taken, a session removed meanwhile is left out, and the table's change count then
/// differs from the one returned, so the next save follows. Returns the sessions written and the change count the save started from.
pub fn save_table(table: &std::sync::Mutex<SessionTable>, path: &Path, max_age_s: u64) -> Result<(usize, u64), String> {
    save_table_with(table, path, max_age_s, &mut || {})
}

fn save_table_with(table: &std::sync::Mutex<SessionTable>, path: &Path, max_age_s: u64, before_write: &mut dyn FnMut()) -> Result<(usize, u64), String> {
    use std::os::unix::fs::FileExt;
    static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    private_dir(path.parent().unwrap_or(Path::new(".")))?;
    let tmp = PathBuf::from(format!("{}.new", path.display()));
    let _ = std::fs::remove_file(&tmp);
    let now = unix_now();
    let (ids, at) = {
        let t = table.lock().unwrap();
        (t.sessions.keys().copied().collect::<Vec<u64>>(), t.changes)
    };
    let mut write = || -> std::io::Result<usize> {
        let mut o = std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&tmp)?;
        o.write_all(&[0u8; 56])?;
        let (mut hash, mut body_len, mut n) = (FNV_START, 0u64, 0usize);
        let mut buf: Vec<u8> = Vec::new();
        let mut next = 0;
        while next < ids.len() {
            buf.clear();
            {
                let t = table.lock().unwrap();
                while next < ids.len() && buf.len() < SAVE_BATCH {
                    if let Some(s) = t.sessions.get(&ids[next]).filter(|s| saved_in(s, now, max_age_s)) {
                        write_session(&mut buf, ids[next], s);
                        n += 1;
                    }
                    next += 1;
                }
            }
            before_write();
            o.write_all(&buf)?;
            hash = fnv_update(hash, &buf);
            body_len += buf.len() as u64;
        }
        drop(buf);
        let (next_id, clock) = {
            let t = table.lock().unwrap();
            (t.next_id, t.clock)
        };
        o.write_all_at(&table_header(now, next_id, clock, n as u64, body_len, hash), 0)?;
        o.sync_all()?;
        drop(o);
        std::fs::rename(&tmp, path)?;
        if let Some(d) = path.parent() {
            std::fs::File::open(d)?.sync_all()?;
        }
        Ok(n)
    };
    match write() {
        Ok(n) => Ok((n, at)),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(format!("{}: {e}", path.display()))
        }
    }
}

/// One periodic save: if `table` changed since the count in `saved`, save it (`save_table`) and set `saved` to the count
/// the save started from, then give the allocator's free memory back. None when nothing changed; a failed write leaves
/// `saved`, so the next call tries again.
pub fn save_if_changed(table: &std::sync::Mutex<SessionTable>, saved: &mut u64, path: &Path, max_age_s: u64) -> Option<Result<usize, String>> {
    if table.lock().unwrap().changes() == *saved {
        return None;
    }
    let r = save_table(table, path, max_age_s).map(|(n, at)| {
        *saved = at;
        n
    });
    trim_heap();
    Some(r)
}

/// Memory the process holds, as the kernel counts it (VmRSS), in bytes; None where /proc has no such line.
pub fn vm_rss_bytes() -> Option<usize> {
    parse_vm_rss(&std::fs::read_to_string("/proc/self/status").ok()?)
}

/// The most memory the process has held (VmHWM), in bytes.
pub fn vm_hwm_bytes() -> Option<usize> {
    parse_status_kb(&std::fs::read_to_string("/proc/self/status").ok()?, "VmHWM:")
}

fn parse_vm_rss(status: &str) -> Option<usize> {
    parse_status_kb(status, "VmRSS:")
}

fn parse_status_kb(status: &str, key: &str) -> Option<usize> {
    let line = status.lines().find_map(|l| l.strip_prefix(key))?;
    let kb: usize = line.trim().strip_suffix("kB")?.trim().parse().ok()?;
    kb.checked_mul(1024)
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod heap {
    extern "C" {
        fn mallopt(param: i32, value: i32) -> i32;
        fn malloc_trim(pad: usize) -> i32;
    }
    const M_ARENA_MAX: i32 = -8;
    pub fn cap_arenas(n: i32) {
        unsafe {
            mallopt(M_ARENA_MAX, n);
        }
    }
    pub fn trim() {
        unsafe {
            malloc_trim(0);
        }
    }
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
mod heap {
    pub fn cap_arenas(_n: i32) {}
    pub fn trim() {}
}

/// At most `n` malloc arenas for the process (call before threads allocate): each thread's own arena keeps what it
/// freed, and a table of long texts built and dropped on many threads would hold many times its size.
pub fn cap_heap_arenas(n: i32) {
    heap::cap_arenas(n);
}

/// Gives the allocator's free memory back to the kernel.
pub fn trim_heap() {
    heap::trim();
}

/// What one reading of the process's memory led to (`Governor::step`).
#[derive(Debug, Default, PartialEq)]
pub struct Governed {
    /// The process is over its limit: no image request is taken.
    pub over: bool,
    /// Sessions dropped, to release on the engine.
    pub released: Vec<u64>,
    /// A line for the log, once per episode: the memory is held by something other than the table.
    pub held_elsewhere: Option<String>,
}

/// Keeps the process under a memory limit by dropping idle sessions, without emptying the table when the memory is not
/// the table's. A round that freed bytes is judged by the next reading, taken after the allocator gave memory back:
/// if the reading has not fallen by at least a quarter of what was freed, nothing more is dropped until the process has
/// been under the limit once. Within one stretch over the limit the table never goes below half of what it held when the
/// stretch began.
#[derive(Debug, Default)]
pub struct Governor {
    over: bool,
    stalled: bool,
    floor: usize,
    /// What the last round freed and the reading it was made at.
    last: Option<(usize, usize)>,
}

impl Governor {
    pub fn step(&mut self, table: &mut SessionTable, rss: usize, limit: usize) -> Governed {
        if rss <= limit {
            *self = Governor::default();
            return Governed::default();
        }
        if !self.over {
            self.over = true;
            self.floor = table.bytes() / 2;
        }
        let mut held_elsewhere = None;
        if let Some((freed, at)) = self.last.take() {
            if at.saturating_sub(rss) < freed / 4 && !self.stalled {
                self.stalled = true;
                held_elsewhere = Some(format!(
                    "front-end memory is held by something other than the session table: RSS {} MiB, the table counts {} MiB, the limit is {} MiB; no more sessions are dropped until RSS has been under the limit",
                    rss >> 20,
                    table.bytes() >> 20,
                    limit >> 20
                ));
            }
        }
        if self.stalled {
            return Governed { over: true, released: Vec::new(), held_elsewhere };
        }
        let (freed, released) = table.shed(rss - limit, self.floor);
        if freed > 0 {
            self.last = Some((freed, rss));
        }
        Governed { over: true, released, held_elsewhere }
    }
}

/// The line that says what the table and the process hold, to choose the limits from.
pub fn front_status(bytes: usize, sessions: usize, tokens: usize, rss: Option<usize>, hwm: Option<usize>) -> String {
    let mib = |b: Option<usize>| b.map_or("?".to_string(), |b| (b >> 20).to_string());
    let per = if tokens == 0 { 0.0 } else { bytes as f64 / tokens as f64 };
    format!("session table: {} MiB counted, {sessions} sessions, {tokens} tokens, {per:.1} B/token; RSS {} MiB, high-water {} MiB", bytes >> 20, mib(rss), mib(hwm))
}

/// The latest cut that session `s` and the prompt `p` have in common (the same bytes and the same hash) with the same images
/// inside it. Cuts are a chain: once the hashes differ at one place, no later cut is shared.
fn shared_cut(s: &Session, p: &Prompt) -> Option<Cut> {
    let (mut i, mut j, mut best) = (0, 0, None);
    while i < s.cuts.len() && j < p.edges.len() {
        let (c, e) = (s.cuts[i], &p.edges[j]);
        match c.bytes.cmp(&e.bytes) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                if c.hash != e.hash || !images_agree(&s.images, p.images, c.bytes) {
                    break;
                }
                best = Some(c);
                i += 1;
                j += 1;
            }
        }
    }
    best
}

/// The session a prompt continues or forks, as `SessionTable::pick` found it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Best {
    sid: u64,
    /// The latest cut it shares with the prompt (`Cut::START`: none).
    cut: Cut,
    /// The bytes of text it shares with the prompt, at least.
    shared: usize,
    in_place: bool,
    epoch: u64,
    /// The factor of that session.
    factor: u32,
}

/// What `SessionTable::pick` found: the session that shares the most with the prompt, whatever that is (for the log), and the one
/// worth continuing or forking (it shares at least `MIN_SHARED` bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Found {
    top: Option<(u64, usize)>,
    best: Option<Best>,
}

/// `SessionTable::plan` for the table behind a lock, which is held only while the table is read and written, never while text is
/// tokenised: the prompt is scanned first, the table picks a session under the lock, the text after its cut is tokenised outside it,
/// and the choice is made again under the lock to commit. A choice the table has since changed is made over (after a few times,
/// under the lock). Returns the plan and the sessions to RELEASE on the engine.
pub fn plan_shared(table: &Mutex<SessionTable>, text: &str, tok: &Tokenizer, threads: usize, ablit: bool, images: &[ImageKey]) -> (Plan, Vec<u64>) {
    plan_shared_with(table, text, tok, threads, ablit, images, &mut || {})
}

/// `plan_shared`, with `between` called after each choice is tokenised and before the table is locked again to commit it: the moment
/// another request may take the session (a test's hook).
fn plan_shared_with(table: &Mutex<SessionTable>, text: &str, tok: &Tokenizer, threads: usize, ablit: bool, images: &[ImageKey], between: &mut dyn FnMut()) -> (Plan, Vec<u64>) {
    let p = Prompt::new(text, tok, ablit, images);
    let (mut only, mut tries) = (None, 0);
    let mut tokenised: Option<(usize, Vec<u32>)> = None;
    for _ in 0..4 {
        let found = table.lock().unwrap().pick(&p, only);
        let from = found.best.map_or(Cut::START, |b| b.cut);
        if tokenised.as_ref().is_none_or(|(at, _)| *at != from.bytes) {
            tokenised = Some((from.bytes, tok.encode_parallel(&p.text[from.bytes..], threads)));
        }
        let f = factor_for(from.ids as usize + tokenised.as_ref().map_or(0, |t| t.1.len()));
        if !(found.best.is_none_or(|b| b.factor == f) || only == Some(0)) {
            tries += 1;
            only = Some(if tries >= 2 { 0 } else { f });
            continue;
        }
        between();
        let mut t = table.lock().unwrap();
        if t.pick(&p, only) == found {
            let suffix = tokenised.take().map(|t| t.1).unwrap_or_default();
            return t.commit(&found, &p, suffix, f);
        }
    }
    table.lock().unwrap().plan_locked(&p, tok, threads)
}

impl SessionTable {
    /// Holds up to `max_sessions` sessions and `max_tokens` ids in total. Ids start from the clock,
    /// so that a table begun afresh never reuses an earlier one's ids, whose sessions the engine may
    /// still keep.
    pub fn new(max_sessions: usize, max_tokens: usize) -> Self {
        SessionTable {
            sessions: HashMap::new(),
            index: HashMap::new(),
            next_id: (unix_now() << 20).max(1),
            clock: 0,
            changes: 0,
            max_sessions,
            limits_known: true,
            max_tokens,
            max_bytes: usize::MAX,
            image_cap: 0,
            image_bytes: 0,
            #[cfg(test)]
            append_without_truncating: false,
        }
    }

    /// How many changes the sessions held have seen (see `changes`); equal counts mean equal tables.
    pub fn changes(&self) -> u64 {
        self.changes
    }

    /// The table for the next start: the sessions used within `max_age_s` (one in a request as its
    /// last finished turn left it), written to `path` (see `write_table`). Returns how many sessions it holds.
    pub fn save(&self, path: &Path, max_age_s: u64) -> Result<usize, String> {
        let (bytes, n) = self.snapshot(max_age_s);
        write_table(path, &bytes)?;
        Ok(n)
    }

    /// The saved form of the sessions used within `max_age_s` (one in a request as its last finished turn left it;
    /// one with no finished turn is left out), and how many they are, whole in memory. A caller that writes it to a
    /// file while requests are served uses `save_table`, which holds a few megabytes at a time and no lock across
    /// a disk write.
    pub fn snapshot(&self, max_age_s: u64) -> (Vec<u8>, usize) {
        let now = unix_now();
        let (mut body, mut n) = (Vec::new(), 0u64);
        for (&id, s) in &self.sessions {
            if saved_in(s, now, max_age_s) {
                n += 1;
                write_session(&mut body, id, s);
            }
        }
        let mut f = table_header(now, self.next_id, self.clock, n, body.len() as u64, fnv(&body)).to_vec();
        f.extend(body);
        (f, n as usize)
    }

    /// The table an earlier run saved at `path`, when the file is whole, ours alone (mode 0600, in a
    /// directory of ours) and at most `max_age_s` old. The file stays where it is, whole or not, until a
    /// save replaces it: a crash at any moment from the start on finds the last saved table. Sessions not used within `max_age_s`, or whose ids do not spell their text with
    /// this tokenizer, are left out. Err holds the reason ("" when there is no file).
    pub fn load(&mut self, path: &Path, tok: &Tokenizer, max_age_s: u64) -> Result<usize, String> {
        if let Err(e) = std::fs::symlink_metadata(path) {
            return Err(if e.kind() == std::io::ErrorKind::NotFound { String::new() } else { format!("{}: {e}", path.display()) });
        }
        let r = self.load_file(path, tok, max_age_s);
        let _ = std::fs::remove_file(format!("{}.new", path.display()));
        r
    }

    fn load_file(&mut self, path: &Path, tok: &Tokenizer, max_age_s: u64) -> Result<usize, String> {
        private_dir(path.parent().unwrap_or(Path::new(".")))?;
        let me = std::fs::metadata("/proc/self").map_err(|e| e.to_string())?.uid();
        let mut file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let m = file.metadata().map_err(|e| e.to_string())?;
        if !m.is_file() || m.uid() != me || m.mode() & 0o777 != 0o600 {
            return Err(format!("{} is not a file of ours of mode 0600", path.display()));
        }
        let mut f = Vec::new();
        file.read_to_end(&mut f).map_err(|e| e.to_string())?;
        let known = [TABLE_MAGIC, TABLE_MAGIC_V3, TABLE_MAGIC_V2, TABLE_MAGIC_V1];
        if f.len() < 56 || !known.iter().any(|m| f[..8] == m[..]) {
            return Err("not a saved table".into());
        }
        // Layouts before `DSFT0004` hold each conversation's text and ids: read once, they become cuts.
        let old = &f[..8] != TABLE_MAGIC;
        let with_flags = &f[..8] != TABLE_MAGIC_V1;
        let with_factor = &f[..8] == TABLE_MAGIC_V3;
        let mut h = Bytes { b: &f[8..56], at: 0 };
        let (written, next_id, clock, n, body_len, hash) = (h.u64()?, h.u64()?, h.u64()?, h.u64()?, h.u64()?, h.u64()?);
        let now = unix_now();
        if now.saturating_sub(written) > max_age_s {
            return Err(format!("saved {} s ago", now.saturating_sub(written)));
        }
        let body = &f[56..];
        if body.len() as u64 != body_len || fnv(body) != hash {
            return Err("incomplete or damaged".into());
        }
        let mut r = Bytes { b: body, at: 0 };
        let mut got = HashMap::new();
        for _ in 0..n {
            let one = if old { read_old_session(&mut r, with_flags, with_factor, tok, now, max_age_s)? } else { read_session(&mut r, now, max_age_s)? };
            if let Some((id, s)) = one {
                got.insert(id, s);
            }
        }
        if r.at != body.len() {
            return Err("a bad body".into());
        }
        let k = got.len();
        self.next_id = self.next_id.max(next_id).max(got.keys().max().map_or(0, |m| m + 1));
        self.clock = self.clock.max(clock);
        self.sessions = got;
        self.changes += 1;
        self.index.clear();
        let ids: Vec<u64> = self.sessions.keys().copied().collect();
        for id in ids {
            self.enter(id);
        }
        let _ = self.evict(0);
        Ok(k)
    }

    fn new_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// The session of the prompt's weights path (and factor `only`, when given) that shares the most with the prompt, found from the
    /// deepest index level the prompt shares with some session: the sessions of that bucket are compared by their cuts, never by reading
    /// text. What a session shares is at least the length of the deepest level its prefix hash agrees at, and at least its latest cut
    /// shared with the prompt; a session that shares a long text with no added token in it is still worth naming as a base (the engine
    /// finds the ids it has in common itself), though no cut of it can be named.
    fn pick(&self, p: &Prompt, only: Option<u32>) -> Found {
        // (session, its latest shared cut or `Cut::START`, the bytes it shares at least)
        let mut top: Option<(u64, Cut, usize)> = None;
        for level in (0..p.keys.len()).rev() {
            for &id in self.index.get(&p.keys[level]).map_or(&[][..], |v| &v[..]) {
                let s = &self.sessions[&id];
                if s.ablit != p.ablit || only.is_some_and(|f| s.factor != f) || !s.ready() {
                    continue;
                }
                let cut = shared_cut(s, p);
                let by_level = if images_agree(&s.images, p.images, level_len(level)) { level_len(level) } else { 0 };
                let shared = by_level.max(cut.map_or(0, |c| c.bytes));
                let cut_bytes = cut.map_or(0, |c| c.bytes);
                if shared > 0 && top.is_none_or(|(_, c, sh)| (shared, cut_bytes) > (sh, c.bytes)) {
                    top = Some((id, cut.unwrap_or(Cut::START), shared));
                }
            }
            if top.is_some_and(|(_, _, sh)| sh >= level_len(level)) {
                break;
            }
        }
        let best = top.filter(|&(_, _, sh)| sh >= MIN_SHARED).map(|(sid, cut, shared)| {
            let s = &self.sessions[&sid];
            Best { sid, cut, shared, in_place: !s.busy && cut.bytes + PROMPT_SLACK >= s.prompt_len, epoch: s.epoch, factor: s.factor }
        });
        Found { top: top.map(|(id, _, sh)| (id, sh)), best }
    }

    /// Takes the plan `found` made: the session (a new one for a fork or a start) is busy until `finish`, `give_up` or `abandon`.
    /// Returns the plan and the sessions to RELEASE on the engine.
    fn commit(&mut self, found: &Found, p: &Prompt, suffix: Vec<u32>, factor: u32) -> (Plan, Vec<u64>) {
        self.clock += 1;
        self.changes += 1;
        let (session_id, base_session_id, from, epoch) = match found.best {
            None => (self.new_id(), 0, Cut::START, 0),
            Some(b) if b.in_place => (b.sid, 0, b.cut, b.epoch),
            Some(b) => (self.new_id(), b.sid, b.cut, b.epoch),
        };
        let ablit = p.ablit;
        let state = match p.edges.binary_search_by_key(&from.bytes, |e| e.bytes) {
            Ok(i) if from.bytes > 0 => Resume(p.edges[i].state.clone()),
            _ => Resume::start(),
        };
        let clock = self.clock;
        if base_session_id != 0 && !self.sessions.contains_key(&session_id) {
            // the fork's own record starts with the cuts it continues; it is no candidate until its first turn finishes
            let src = &self.sessions[&base_session_id];
            let cuts: Vec<Cut> = src.cuts.iter().copied().filter(|c| c.bytes <= from.bytes).collect();
            let levels = (0..).take_while(|&l| level_len(l) <= from.bytes).count();
            let keys = src.keys[..levels.min(src.keys.len())].to_vec();
            self.sessions.insert(session_id, Session { cuts, keys, ..Session::empty(ablit, factor, clock) });
        }
        let s = self.sessions.entry(session_id).or_insert_with(|| Session::empty(ablit, factor, clock));
        s.busy = true;
        s.last_used = clock;
        s.t_used = unix_now();
        let released = self.evict(session_id);
        let plan = Plan { session_id, base_session_id, from, epoch, state, suffix, matched: found.top.map_or(0, |t| t.0), shared: found.top.map_or(0, |t| t.1) };
        (plan, released)
    }

    /// `plan_shared` with the table held throughout (tokenising under the caller's lock).
    fn plan_locked(&mut self, p: &Prompt, tok: &Tokenizer, threads: usize) -> (Plan, Vec<u64>) {
        // The request's factor follows from its ids, which follow from the session it continues: try the best
        // match at any factor, then the best of the factor the ids came to; if that moves the factor again,
        // start fresh (a prompt tokenised whole has one length and so one factor).
        let mut only = None;
        let mut tries = 0;
        loop {
            let found = self.pick(p, only);
            let from = found.best.map_or(Cut::START, |b| b.cut);
            let suffix = tok.encode_parallel(&p.text[from.bytes..], threads);
            let f = factor_for(from.ids as usize + suffix.len());
            if found.best.is_none_or(|b| b.factor == f) || only == Some(0) {
                return self.commit(&found, p, suffix, f);
            }
            tries += 1;
            only = Some(if tries >= 2 { 0 } else { f });
        }
    }

    /// Choose the session and the ids for a prompt of the weights path `ablit`: only sessions of
    /// that path are candidates. The chosen session is busy until `finish`, `give_up` or `abandon`. Returns the
    /// plan and the sessions to RELEASE on the engine.
    pub fn plan(&mut self, text: &str, tok: &Tokenizer, threads: usize, ablit: bool) -> (Plan, Vec<u64>) {
        self.plan_images(text, tok, threads, ablit, &[])
    }

    /// `plan` for a prompt with images: `text` has each image's placeholder written once per row, and
    /// `images` says where each run starts in it, how long it is and what the image is. A session whose
    /// images inside the shared text are not these (see the module comment) is no candidate.
    pub fn plan_images(&mut self, text: &str, tok: &Tokenizer, threads: usize, ablit: bool, images: &[ImageKey]) -> (Plan, Vec<u64>) {
        let p = Prompt::new(text, tok, ablit, images);
        self.plan_locked(&p, tok, threads)
    }

    /// Record what the session holds after a request that ended in a DONE: the cuts of the prompt's new text and of the output, and
    /// the epoch and size of the history the engine reports (`tail`; none from an engine of the older protocol: the session then
    /// names no epoch, and its next request carries its whole prompt).
    pub fn finish(&mut self, session_id: u64, turn: &Turn, generated: &[u32], tail: Option<DoneTail>, tok: &Tokenizer) {
        self.changes += 1;
        if !self.sessions.contains_key(&session_id) {
            return;
        }
        let gen_bytes = tok.decode_bytes(generated);
        let mut region = turn.text[turn.from.bytes..].to_string();
        let mut ids = turn.suffix.to_vec();
        // Generated text that is not valid UTF-8 cannot come back verbatim; keep the prompt only.
        if let Ok(g) = std::str::from_utf8(&gen_bytes) {
            region.push_str(g);
            ids.extend_from_slice(generated);
        }
        let (marks, at) = mark_ids(&ids, |id| tok.token_bytes(id).len());
        debug_assert_eq!(at, region.len());
        let total = turn.from.ids as u64 + ids.len() as u64;
        let n_hist = tail.map_or(total, |t| t.n_hist as u64);
        let (mut edges, scanned) = scan(&region, turn.state, turn.from.bytes, tok);
        edges.retain(|e| !in_image_run(turn.images, e.bytes));
        self.leave(session_id);
        let s = self.sessions.get_mut(&session_id).unwrap();
        // What the engine rewrote below the cut this request continued from is no longer what the cuts above it describe: they go
        // before the new ones come (a cut kept would vouch for ids that are not there).
        #[cfg(test)]
        let truncate = !self.append_without_truncating;
        #[cfg(not(test))]
        let truncate = true;
        if truncate {
            s.cuts.retain(|c| c.bytes <= turn.from.bytes);
        }
        let before = s.cuts.len();
        for e in &edges {
            if let Some(n) = token_end_in(&marks, &ids, region.len(), e.bytes - turn.from.bytes, |id| tok.token_bytes(id).len()) {
                let upto = turn.from.ids as u64 + n as u64;
                if upto <= n_hist {
                    s.cuts.push(Cut { bytes: e.bytes, ids: upto as u32, hash: e.hash });
                }
            }
        }
        // the cuts stay in order (the test control that skips the truncation breaks it on purpose, and its error is the point)
        debug_assert!(!truncate || s.cuts[before.saturating_sub(1)..].windows(2).all(|w| w[0].bytes < w[1].bytes && w[0].ids <= w[1].ids));
        shrink_slack(&mut s.cuts);
        let levels = (0..).take_while(|&l| level_len(l) <= turn.from.bytes).count();
        s.keys.truncate(levels);
        s.keys.extend(scanned);
        shrink_slack(&mut s.keys);
        s.epoch = tail.map_or(0, |t| t.epoch);
        s.n_hist = n_hist.min(u32::MAX as u64) as u32;
        // The session's factor is the larger of its own and the one its whole history needs, so the next turn matches it at the factor the engine used.
        s.factor = s.factor.max(factor_for(n_hist as usize));
        s.prompt_len = turn.text.len();
        s.images = turn.images.to_vec();
        s.busy = false;
        self.enter(session_id);
    }

    /// A request that ended without a DONE (its client left, or it was cancelled in the queue) changes nothing in its session but
    /// that it is no longer in a request; a session that has never finished a turn is dropped, and returned for the caller to
    /// release on the engine. If the engine cut the history meanwhile, the next request is answered `PrefixGone` and goes whole.
    pub fn give_up(&mut self, session_id: u64) -> bool {
        self.changes += 1;
        let Some(s) = self.sessions.get_mut(&session_id) else { return false };
        s.busy = false;
        if s.ready() {
            return false;
        }
        self.remove(session_id);
        true
    }

    /// The prefix keys of session `id` into the index.
    fn enter(&mut self, id: u64) {
        let Some(s) = self.sessions.get(&id) else { return };
        for &k in &s.keys {
            self.index.entry(k).or_default().push(id);
        }
    }

    /// Session `id` out of the index (before its keys change or it goes).
    fn leave(&mut self, id: u64) {
        let Some(s) = self.sessions.get(&id) else { return };
        for k in &s.keys {
            if let Some(v) = self.index.get_mut(k) {
                v.retain(|&x| x != id);
                if v.is_empty() {
                    self.index.remove(k);
                }
            }
        }
    }

    /// Remove session `id` from the table and the index.
    fn remove(&mut self, id: u64) {
        self.changes += 1;
        self.leave(id);
        self.sessions.remove(&id);
    }

    /// The request failed: forget the session (the engine's copy is in an unknown state; the
    /// caller releases it).
    pub fn abandon(&mut self, session_id: u64) {
        self.remove(session_id);
    }

    /// Drop least recently used idle sessions (never `keep`, never one in a request) while the table holds more than
    /// `max_sessions` sessions, `max_tokens` ids or `max_bytes` bytes; returns their ids.
    fn evict(&mut self, keep: u64) -> Vec<u64> {
        let mut out = Vec::new();
        let mut tokens: usize = self.sessions.values().map(|s| s.n_hist as usize).sum();
        let mut bytes: usize = self.bytes();
        loop {
            let counts_ok = !self.limits_known || (self.sessions.len() <= self.max_sessions && tokens <= self.max_tokens);
            if counts_ok && bytes <= self.max_bytes {
                return out;
            }
            let Some(id) = self.lru_idle(keep) else { return out };
            let s = &self.sessions[&id];
            tokens -= s.n_hist as usize;
            bytes -= s.bytes();
            self.remove(id);
            out.push(id);
        }
    }

    /// The least recently used session that is in no request and is not `keep`.
    fn lru_idle(&self, keep: u64) -> Option<u64> {
        self.sessions.iter().filter(|(&id, s)| id != keep && !s.busy).min_by_key(|(_, s)| s.last_used).map(|(&id, _)| id)
    }

    /// The memory the table holds: every session's buffers by capacity (`Session::bytes`) and the image rows in flight
    /// (`set_image_bytes`).
    pub fn bytes(&self) -> usize {
        self.sessions.values().map(|s| s.bytes()).sum::<usize>() + self.image_bytes
    }

    /// The most memory the sessions may hold (0: no limit), and the most the image rows in flight count for against it.
    pub fn set_byte_limits(&mut self, max_bytes: usize, image_cap: usize) {
        self.max_bytes = if max_bytes == 0 { usize::MAX } else { max_bytes };
        self.image_cap = image_cap;
    }

    /// The image rows in flight take `bytes` of the process (counted up to the image limit): the table gives that room up.
    pub fn set_image_bytes(&mut self, bytes: usize) {
        self.image_bytes = bytes.min(self.image_cap);
    }

    /// Drop idle sessions, least recently used first (never one in a request), until their memory is at least `need`
    /// bytes, but not past the point where the table would hold less than `floor` bytes. Returns the memory freed and
    /// the ids dropped, which the caller releases on the engine.
    pub fn shed(&mut self, need: usize, floor: usize) -> (usize, Vec<u64>) {
        let (mut freed, mut out) = (0usize, Vec::new());
        let mut held = self.bytes();
        while freed < need {
            let Some(id) = self.lru_idle(0) else { break };
            let b = self.sessions[&id].bytes();
            if held.saturating_sub(b) < floor {
                break;
            }
            held -= b;
            freed += b;
            self.remove(id);
            out.push(id);
        }
        (freed, out)
    }

    /// (memory held by the sessions and image rows by `bytes`, sessions, ids).
    pub fn stats(&self) -> (usize, usize, usize) {
        (self.bytes(), self.sessions.len(), self.sessions.values().map(|s| s.n_hist as usize).sum())
    }

    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    /// Until `keep_for` runs, the session and id counts bound nothing (the memory limit still does): the engine's limits are not
    /// known yet, and the table must not trim below them.
    pub fn defer_limits(&mut self) {
        self.limits_known = false;
    }

    /// Drops idle sessions beyond the limits now set; returns their ids, which the caller releases on the engine.
    pub fn trim_to_limits(&mut self) -> Vec<u64> {
        self.evict(0)
    }

    /// The table's limits from what the engine can keep (HELLO keep_positions), never under 256 sessions and 16M ids; the id
    /// limit also holds four conversations of the whole window plus a sibling of at most 1M ids beside each (the session
    /// left behind at a factor boundary).
    pub fn keep_for(&mut self, keep_positions: u64, max_context: u32) {
        self.limits_known = true;
        let conversations = 4 * (max_context as usize + max_context.min(FACTOR_STEP as u32) as usize);
        self.max_tokens = (keep_positions as usize).max(16 << 20).max(conversations);
        self.max_sessions = (keep_positions as usize / 4096).max(256);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::{encode_messages, EncodeOptions, ThinkingMode};
    use serde_json::{json, Value};
    use std::time::Instant;

    const TOKENIZER: &str = "/data/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/tokenizer.json";

    fn syn() -> Tokenizer {
        crate::tokenizer::synthetic::tokenizer()
    }

    /// The longest common prefix of two byte strings.
    fn common_prefix(a: &[u8], b: &[u8]) -> usize {
        a.iter().zip(b).take_while(|(x, y)| x == y).count()
    }

    /// A finished session of `text` straight into the table and its index: the cuts, keys and counts a request that ran would have left.
    fn put(t: &mut SessionTable, id: u64, text: &str, ablit: bool) {
        let tok = syn();
        let ids = tok.encode(text);
        let (marks, _) = mark_ids(&ids, |x| tok.token_bytes(x).len());
        let (edges, keys) = scan(text, &Resume::start(), 0, &tok);
        let cuts = edges
            .iter()
            .filter_map(|e| token_end_in(&marks, &ids, text.len(), e.bytes, |x| tok.token_bytes(x).len()).map(|n| Cut { bytes: e.bytes, ids: n as u32, hash: e.hash }))
            .collect();
        t.sessions.insert(id, Session { cuts, keys, prompt_len: text.len(), n_hist: ids.len() as u32, epoch: 5, ..Session::empty(ablit, 16, 0) });
        t.enter(id);
    }

    fn table_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("dsfront-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir(&dir).unwrap();
        dir
    }

    /// `put`, used just now (a session older than the save's age limit is not saved).
    fn put_fresh(t: &mut SessionTable, id: u64, text: &str, ablit: bool) {
        put(t, id, text, ablit);
        t.sessions.get_mut(&id).unwrap().t_used = unix_now();
    }

    /// The tail an engine reports for a request that ran: its history is the prompt and the output but the output's last id.
    fn tail_of(total: usize, generated: &[u32]) -> DoneTail {
        DoneTail { epoch: 5, n_hist: (total + generated.len().saturating_sub(1)) as u32 }
    }

    /// What a request carries to the engine: its suffix after `plan.from` when it names the entry (`prefix`), else the whole prompt.
    fn carried(plan: &Plan, tok: &Tokenizer, text: &str, prefix: bool) -> (Cut, Resume, Vec<u32>) {
        if prefix || plan.from == Cut::START {
            (plan.from, plan.state.clone(), plan.suffix.clone())
        } else {
            (Cut::START, Resume::start(), tok.encode(text))
        }
    }

    /// The table's finish for a planned request whose output was `reply`, with the tail an engine would report.
    fn finish_plan(t: &mut SessionTable, tok: &Tokenizer, plan: &Plan, text: &str, reply: &[u32], images: &[ImageKey]) {
        let (from, state, suffix) = carried(plan, tok, text, images.is_empty());
        let total = from.ids as usize + suffix.len();
        t.finish(plan.session_id, &Turn { text, from, state: &state, suffix: &suffix, images }, reply, Some(tail_of(total, reply)), tok);
    }

    /// One turn through the table's own steps: plan, then finish.
    fn serve(t: &mut SessionTable, tok: &Tokenizer, text: &str, reply: &[u32], ablit: bool) -> Plan {
        serve_images(t, tok, text, reply, ablit, &[])
    }

    fn serve_images(t: &mut SessionTable, tok: &Tokenizer, text: &str, reply: &[u32], ablit: bool, images: &[ImageKey]) -> Plan {
        let (plan, _) = t.plan_images(text, tok, 1, ablit, images);
        finish_plan(t, tok, &plan, text, reply, images);
        plan
    }

    /// The best match of `text` as a session id (None: none shares enough).
    fn best_of(t: &SessionTable, tok: &Tokenizer, text: &str, ablit: bool, only: Option<u32>, images: &[ImageKey]) -> Option<u64> {
        t.pick(&Prompt::new(text, tok, ablit, images), only).best.map(|b| b.sid)
    }

    // ---------------------------------------------------------------- the engine's side, as a model

    /// What rank 0 keeps per session (engine/server/prefix.h): the history, the epoch a request that continues it must name, and how many
    /// ids the last DONE reported. The epoch is new when a run starts below what the last DONE reported.
    #[derive(Clone)]
    struct EEntry {
        hist: Vec<u32>,
        epoch: u64,
        reported: usize,
    }

    #[derive(Default)]
    struct Eng {
        entries: HashMap<u64, EEntry>,
        epochs: u64,
    }

    impl Eng {
        /// The prompt rank 0 builds from a request: None is `PrefixGone`. The entry is the base's when the request names one.
        fn build(&self, session: u64, base: u64, prefix: Option<(u64, u32)>, ids: &[u32]) -> Option<Vec<u32>> {
            let Some((epoch, n)) = prefix else { return Some(ids.to_vec()) };
            let e = self.entries.get(&if base != 0 { base } else { session })?;
            if e.epoch != epoch || n as usize > e.reported || n as usize > e.hist.len() {
                return None;
            }
            let mut p = e.hist[..n as usize].to_vec();
            p.extend_from_slice(ids);
            Some(p)
        }

        /// The run: the history becomes the prompt and the output but its last id. `back` is how far below the shared ids the run restarts
        /// (the engine may restart at a snapshot).
        fn run(&mut self, session: u64, prompt: &[u32], generated: &[u32], back: usize) -> DoneTail {
            let q = self.entries.get(&session).map(|e| common_prefix_ids(&e.hist, prompt).saturating_sub(back));
            let epoch = match (self.entries.get(&session), q) {
                (Some(e), Some(q)) if q >= e.reported => e.epoch,
                _ => {
                    self.epochs += 1;
                    self.epochs
                }
            };
            let mut hist = prompt.to_vec();
            hist.extend_from_slice(&generated[..generated.len().saturating_sub(1)]);
            let reported = hist.len();
            self.entries.insert(session, EEntry { hist, epoch, reported });
            DoneTail { epoch, n_hist: reported as u32 }
        }
    }

    fn common_prefix_ids(a: &[u32], b: &[u32]) -> usize {
        a.iter().zip(b).take_while(|(x, y)| x == y).count()
    }

    /// The table and the engine's model, driven as the server drives them.
    struct Rig<'a> {
        t: SessionTable,
        tok: &'a Tokenizer,
        e: Eng,
        /// The protocol version the connection says: 4 never carries a prefix.
        version: u32,
        /// The engine restarts below the shared ids now and then and drops a session now and then.
        wild: bool,
        rng: u64,
    }

    /// One request as it went to the engine.
    struct Sent {
        plan: Plan,
        built: Vec<u32>,
        prefix: bool,
        gone: bool,
    }

    impl<'a> Rig<'a> {
        fn new(tok: &'a Tokenizer, version: u32, wild: bool, seed: u64) -> Rig<'a> {
            Rig { t: SessionTable::new(1 << 20, 1 << 40), tok, e: Eng::default(), version, wild, rng: seed }
        }

        fn turn(&mut self, text: &str, ablit: bool, images: &[ImageKey], reply: &[u32]) -> Sent {
            let (plan, released) = self.t.plan_images(text, self.tok, 1, ablit, images);
            for r in released {
                self.e.entries.remove(&r);
            }
            // never across the weights path or the factor
            let src = if plan.base_session_id != 0 { plan.base_session_id } else { plan.session_id };
            if let Some(s) = self.t.sessions.get(&src).filter(|s| s.ready()) {
                assert_eq!(s.ablit, ablit, "a session of the other weights path was continued");
                assert_eq!(s.factor, factor_for(plan.total()), "a session of another factor was continued");
            }
            let prefix = plan.prefix_mode(self.version, !images.is_empty());
            let (from, state, ids) = carried(&plan, self.tok, text, prefix);
            if self.wild && lcg(&mut self.rng).is_multiple_of(9) {
                self.e.entries.remove(&src);
            }
            let mut built = self.e.build(plan.session_id, plan.base_session_id, prefix.then_some((plan.epoch, from.ids)), &ids);
            let (mut from, mut state, mut ids, mut gone) = (from, state, ids, false);
            if built.is_none() {
                // PrefixGone: tokenise everything, send whole
                (from, state, ids, gone) = (Cut::START, Resume::start(), self.tok.encode(text), true);
                built = Some(ids.clone());
            }
            let built = built.unwrap();
            let back = if self.wild && lcg(&mut self.rng).is_multiple_of(4) { (lcg(&mut self.rng) % 40) as usize } else { 0 };
            let tail = self.e.run(plan.session_id, &built, reply, back);
            let sent_prefix = prefix && !gone;
            // a connection of version 4 has no tail
            let tail = (self.version >= 5).then_some(tail);
            self.t.finish(plan.session_id, &Turn { text, from, state: &state, suffix: &ids, images }, reply, tail, self.tok);
            Sent { plan, built, prefix: sent_prefix, gone }
        }
    }

    // ---------------------------------------------------------------- the plan that reads every session's text and ids, kept as the reference

    struct RefSession {
        text: String,
        ids: Vec<u32>,
        prompt_len: usize,
        ablit: bool,
        factor: u32,
        images: Vec<ImageKey>,
    }

    struct RefPlan {
        session: u64,
        base: u64,
        reused: usize,
        ids: Vec<u32>,
        factor: u32,
    }

    /// A table that keeps every session's text and ids, read whole by brute force.
    #[derive(Default)]
    struct RefTable {
        s: Vec<(u64, RefSession)>,
        next: u64,
    }

    impl RefTable {
        fn best(&self, text: &str, ablit: bool, only: Option<u32>, images: &[ImageKey]) -> Option<(usize, usize)> {
            let mut top: Option<(usize, usize)> = None;
            for (i, (_, s)) in self.s.iter().enumerate() {
                if s.ablit != ablit || only.is_some_and(|f| s.factor != f) {
                    continue;
                }
                let n = common_prefix(text.as_bytes(), s.text.as_bytes());
                if !images_agree(&s.images, images, n) {
                    continue;
                }
                if top.is_none_or(|(_, bn)| n > bn) {
                    top = Some((i, n));
                }
            }
            top
        }

        fn build(&mut self, text: &str, tok: &Tokenizer, ablit: bool, only: Option<u32>, images: &[ImageKey]) -> (RefPlan, Option<u32>) {
            let top = self.best(text, ablit, only, images);
            let best = top.filter(|&(_, n)| n >= MIN_SHARED);
            let (mut plan, cut) = match best {
                None => {
                    self.next += 1;
                    (RefPlan { session: self.next, base: 0, reused: 0, ids: Vec::new(), factor: 16 }, 0)
                }
                Some((i, mut shared)) => {
                    while !text.is_char_boundary(shared) {
                        shared -= 1;
                    }
                    let (sid, s) = &self.s[i];
                    let (marks, _) = mark_ids(&s.ids, |x| tok.token_bytes(x).len());
                    let mut cut = (0, 0);
                    for (a, e) in tok.added_spans(text) {
                        if a > shared {
                            break;
                        }
                        for p in [a, e] {
                            if p <= shared && p > cut.0 {
                                if let Some(n) = token_end_in(&marks, &s.ids, s.text.len(), p, |x| tok.token_bytes(x).len()) {
                                    cut = (p, n);
                                }
                            }
                        }
                    }
                    let ids = s.ids[..cut.1].to_vec();
                    let sid = *sid;
                    let in_place = shared + PROMPT_SLACK >= s.prompt_len;
                    if in_place {
                        (RefPlan { session: sid, base: 0, reused: cut.1, ids, factor: 16 }, cut.0)
                    } else {
                        self.next += 1;
                        (RefPlan { session: self.next, base: sid, reused: cut.1, ids, factor: 16 }, cut.0)
                    }
                }
            };
            plan.ids.extend(tok.encode(&text[cut..]));
            (plan, best.map(|(i, _)| self.s[i].1.factor))
        }

        fn plan(&mut self, text: &str, tok: &Tokenizer, ablit: bool, images: &[ImageKey]) -> RefPlan {
            let (mut only, mut tries) = (None, 0);
            loop {
                let (mut plan, used) = self.build(text, tok, ablit, only, images);
                let f = factor_for(plan.ids.len());
                if used.is_none_or(|u| u == f) || only == Some(0) {
                    plan.factor = f;
                    return plan;
                }
                tries += 1;
                only = Some(if tries >= 2 { 0 } else { f });
            }
        }

        fn finish(&mut self, p: &RefPlan, text: &str, generated: &[u32], tok: &Tokenizer, ablit: bool, images: &[ImageKey]) {
            let gen_bytes = tok.decode_bytes(generated);
            let (mut t, mut ids) = (text.to_string(), p.ids.clone());
            if let Ok(g) = std::str::from_utf8(&gen_bytes) {
                t.push_str(g);
                ids.extend_from_slice(generated);
            }
            let factor = factor_for(ids.len()).max(p.factor);
            let rec = RefSession { text: t, ids, prompt_len: text.len(), ablit, factor, images: images.to_vec() };
            match self.s.iter_mut().find(|(id, _)| *id == p.session) {
                Some((_, s)) => *s = RefSession { factor: s.factor.max(rec.factor), ..rec },
                None => self.s.push((p.session, rec)),
            }
        }

        fn has(&self, id: u64) -> bool {
            self.s.iter().any(|(i, _)| *i == id)
        }
    }

    // ---------------------------------------------------------------- edit histories from the real chat template

    /// A deterministic pseudo-random stream.
    fn lcg(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *state >> 33
    }

    const WORDS: [&str; 16] = ["build", "tests", "the", "file", "was", "changed", "héllo", "run", "✓", "again", "cargo", "error", "line", "fix", "next", "ok"];

    /// `base` words and up to `spread` more.
    fn words_r(rng: &mut u64, base: usize, spread: usize) -> String {
        let n = base + lcg(rng) as usize % spread;
        words(rng, n)
    }

    fn words(rng: &mut u64, n: usize) -> String {
        (0..n).map(|_| WORDS[lcg(rng) as usize % WORDS.len()]).collect::<Vec<_>>().join(" ")
    }

    /// A conversation as a client holds it, rendered through the real chat encoding.
    #[derive(Clone)]
    struct Conv {
        thinking: bool,
        msgs: Vec<Value>,
    }

    impl Conv {
        fn text(&self) -> Option<String> {
            let o = EncodeOptions::new(if self.thinking { ThinkingMode::Thinking } else { ThinkingMode::Chat });
            encode_messages(&self.msgs, &o).ok().map(|x| x.0)
        }

        /// An edit of one message's text that keeps its length (`same_len`) or not.
        fn edit(&mut self, rng: &mut u64, same_len: bool) -> Option<(usize, Value)> {
            let at: Vec<usize> = (0..self.msgs.len().saturating_sub(1)).filter(|&i| self.msgs[i]["content"].as_str().is_some_and(|c| !c.is_empty())).collect();
            if at.is_empty() {
                return None;
            }
            let i = at[lcg(rng) as usize % at.len()];
            let old = self.msgs[i]["content"].clone();
            let s = old.as_str().unwrap();
            let new = if same_len {
                let mut b: Vec<u8> = s.bytes().collect();
                let k = b.iter().position(|c| c.is_ascii_lowercase()).unwrap_or(0);
                if b[k].is_ascii_lowercase() {
                    b[k] = if b[k] == b'z' { b'a' } else { b[k] + 1 };
                }
                String::from_utf8(b).unwrap_or_else(|_| s.to_string())
            } else {
                format!("{s} {}", words(rng, 3))
            };
            self.msgs[i]["content"] = json!(new);
            Some((i, old))
        }
    }

    /// The tallies of a run over many scenarios.
    #[derive(Default, Debug)]
    struct Tally {
        requests: usize,
        compared: usize,
        same: usize,
        /// counted differences: another session wins (the latest shared cut, not the longest byte match) and a fork where the reference's slack rule
        /// says in place; and ids equal though the session choice differs
        other_session: usize,
        fork_for_place: usize,
        same_ids_other_choice: usize,
        /// of those: cold where the reference plan names a base, a base where the reference plan starts cold
        cold_for_base: usize,
        base_for_cold: usize,
        gone: usize,
        prefix_sent: usize,
        skipped: usize,
    }

    /// One scenario: a conversation and the edits a client makes to its history, each request planned by the table (against the
    /// engine's model) and by the reference plan. Err names the first thing that is wrong.
    fn scenario(seed: u64, tok: &Tokenizer, tally: &mut Tally, version: u32, wild: bool, control: bool) -> Result<(), String> {
        let mut rng = seed.wrapping_mul(0x9e3779b97f4a7c15) | 1;
        let thinking = lcg(&mut rng).is_multiple_of(2);
        let mut msgs: Vec<Value> = Vec::new();
        if !lcg(&mut rng).is_multiple_of(4) {
            msgs.push(json!({"role": "system", "content": words_r(&mut rng, 20, 60)}));
        }
        msgs.push(json!({"role": "user", "content": words_r(&mut rng, 4, 30)}));
        let mut conv = Conv { thinking, msgs };
        let mut rig = Rig::new(tok, version, wild, seed ^ 0x5555);
        rig.t.append_without_truncating = control;
        let mut reft = RefTable::default();
        let mut n2r: HashMap<u64, u64> = HashMap::new();
        let mut lockstep = true;
        let mut ablit = lcg(&mut rng).is_multiple_of(7);
        let mut undo: Option<(usize, Value)> = None;
        for k in 0..(2 + lcg(&mut rng) as usize % 6) {
            let Some(text) = conv.text() else {
                tally.skipped += 1;
                return Ok(());
            };
            let reasoning = words_r(&mut rng, 3, 10);
            let answer = words_r(&mut rng, 2, 25);
            let reply_text = if conv.thinking { format!("{reasoning}</think>{answer}") } else { answer.clone() };
            let mut reply = tok.encode(&reply_text);
            reply.push(1);
            let sent = rig.turn(&text, ablit, &[], &reply);
            tally.requests += 1;
            tally.gone += sent.gone as usize;
            tally.prefix_sent += sent.prefix as usize;
            if version < 5 && sent.prefix {
                return Err(format!("seed {seed} request {k}: a prefix on a connection of version {version}"));
            }
            if tok.decode(&sent.built) != text {
                return Err(format!(
                    "seed {seed} request {k}: the engine built a prompt that is not the request's text (prefix {}, from {} ids, session {} base {})",
                    sent.prefix, sent.plan.from.ids, sent.plan.session_id, sent.plan.base_session_id
                ));
            }
            let rp = reft.plan(&text, tok, ablit, &[]);
            if tok.decode(&rp.ids) != text {
                return Err(format!("seed {seed} request {k}: the reference plan is not the text (the reference is wrong)"));
            }
            if lockstep {
                tally.compared += 1;
                let new_src = if sent.plan.base_session_id != 0 { sent.plan.base_session_id } else if n2r.contains_key(&sent.plan.session_id) { sent.plan.session_id } else { 0 };
                let ref_src = if rp.base != 0 { rp.base } else if reft.has(rp.session) { rp.session } else { 0 };
                let same_src = n2r.get(&new_src).copied().unwrap_or(0) == ref_src;
                let (new_fork, ref_fork) = (sent.plan.base_session_id != 0, rp.base != 0);
                if sent.built == rp.ids {
                    tally.same += 1;
                    if !same_src {
                        tally.same_ids_other_choice += 1;
                        tally.cold_for_base += (new_src == 0 && ref_src != 0) as usize;
                        tally.base_for_cold += (new_src != 0 && ref_src == 0) as usize;
                        lockstep = false;
                    } else if new_fork != ref_fork {
                        tally.fork_for_place += 1;
                        lockstep = false;
                    }
                } else if !same_src {
                    tally.other_session += 1;
                    lockstep = false;
                } else if new_fork && !ref_fork {
                    tally.fork_for_place += 1;
                    lockstep = false;
                } else {
                    return Err(format!(
                        "seed {seed} request {k}: the same session, and the prompts differ: new reuses {} ids ({} carried), the reference {}",
                        sent.plan.from.ids,
                        sent.plan.suffix.len(),
                        rp.reused
                    ));
                }
                if lockstep && (new_src == 0 || new_fork) {
                    n2r.insert(sent.plan.session_id, rp.session);
                }
            }
            reft.finish(&rp, &text, &reply, tok, ablit, &[]);
            // the next request
            if let Some((i, old)) = undo.take() {
                conv.msgs[i]["content"] = old; // the edit is undone
            }
            if lcg(&mut rng).is_multiple_of(12) {
                continue; // regenerate: the same prompt again
            }
            if conv.msgs.last().is_some_and(|m| m["role"] == "user" || m["role"] == "tool") {
                if lcg(&mut rng).is_multiple_of(6) {
                    let mut m = json!({"role": "assistant", "content": "", "tool_calls": [{"id": "a", "function": {"name": "f", "arguments": "{}"}}]});
                    if conv.thinking {
                        m["reasoning_content"] = json!(reasoning);
                    }
                    conv.msgs.push(m);
                    conv.msgs.push(json!({"role": "tool", "tool_call_id": "a", "content": words_r(&mut rng, 10, 40)}));
                } else {
                    let mut m = json!({"role": "assistant", "content": answer});
                    if conv.thinking {
                        m["reasoning_content"] = json!(reasoning);
                    }
                    conv.msgs.push(m);
                    conv.msgs.push(json!({"role": "user", "content": words_r(&mut rng, 3, 30)}));
                }
            }
            match lcg(&mut rng) % 8 {
                0 => {
                    // a system (or first) message changed
                    let _ = conv.edit(&mut rng, false);
                }
                1 => {
                    // a message in the middle edited, then restored at the next request
                    let same_len = lcg(&mut rng).is_multiple_of(2);
                    undo = conv.edit(&mut rng, same_len);
                }
                2 if conv.msgs.len() > 3 => {
                    // a fork: the conversation goes back to an earlier user message and continues another way
                    let users: Vec<usize> = (0..conv.msgs.len()).filter(|&i| conv.msgs[i]["role"] == "user").collect();
                    let j = users[lcg(&mut rng) as usize % users.len()];
                    conv.msgs.truncate(j);
                    conv.msgs.push(json!({"role": "user", "content": words(&mut rng, 6)}));
                }
                3 => ablit = !ablit,
                _ => {}
            }
        }
        Ok(())
    }

    /// The edit-then-undo history, in place: the client edits the output the engine just gave (same length) in its next request, which
    /// continues the session in place and rewrites the engine's history below what it reported; then it restores the output. The cut at
    /// the end of that output, made by the first request, describes ids the engine no longer holds.
    fn edit_then_undo(tok: &Tokenizer, control: bool) -> Result<(), String> {
        let mut msgs = vec![
            json!({"role": "system", "content": "You are a careful assistant who answers briefly and well, and who never gives a long answer where a short one will do."}),
            json!({"role": "user", "content": "please check the build and the tests of the project and tell me what is wrong"}),
        ];
        let mut rig = Rig::new(tok, 5, false, 1);
        rig.t.append_without_truncating = control;
        let reply = {
            let mut r = tok.encode("the build is fine");
            r.push(1);
            r
        };
        let render = |msgs: &[Value]| encode_messages(msgs, &EncodeOptions::new(ThinkingMode::Chat)).unwrap().0;
        for step in 0..4 {
            match step {
                1 => {
                    msgs.push(json!({"role": "assistant", "content": "uhe build is fine"}));
                    msgs.push(json!({"role": "user", "content": "and the tests"}));
                }
                2 => msgs[2]["content"] = json!("the build is fine"),
                3 => {
                    msgs.push(json!({"role": "assistant", "content": "all pass"}));
                    msgs.push(json!({"role": "user", "content": "thanks"}));
                }
                _ => {}
            }
            let text = render(&msgs);
            let sent = rig.turn(&text, false, &[], &reply);
            if tok.decode(&sent.built) != text {
                return Err(format!("request {step} (prefix {}, from {} ids): the engine built a prompt that is not the request's text", sent.prefix, sent.plan.from.ids));
            }
            if step == 1 {
                assert_eq!(sent.plan.base_session_id, 0, "the edit of the last reply continues the session in place");
            }
        }
        Ok(())
    }

    #[test]
    fn an_edit_then_its_undo_never_reuses_a_rewritten_prefix() {
        let tok = syn();
        assert_eq!(edit_then_undo(&tok, false), Ok(()), "cuts above the cut a request continued from are dropped at its DONE");
        // control: appending the new cuts without dropping the old ones leaves a cut whose hash describes ids the engine has since rewritten
        let wrong = edit_then_undo(&tok, true).expect_err("the control reuses a rewritten prefix");
        eprintln!("control: {wrong}");
        assert!(wrong.contains("not the request's text"), "{wrong}");
    }

    /// 10,000 edit histories from the real chat template (thinking and chat, tools, a system change, a message edited, an edit and its
    /// undo, a regenerate, a fork, the other weights path): every request's prompt the engine builds spells the request's text, the same
    /// session is never given a different prompt than the reference plan gives it, and what differs is one of the two counted differences.
    #[test]
    fn the_plan_from_cuts_is_the_reference_plan_or_one_of_two_counted_differences() {
        let tok = syn();
        let mut tally = Tally::default();
        for seed in 0..10_000u64 {
            if let Err(e) = scenario(seed, &tok, &mut tally, 5, seed % 5 == 0, false) {
                panic!("{e}\n{tally:?}");
            }
        }
        eprintln!("{tally:?}");
        assert!(tally.skipped < 100, "{tally:?}: the encoder refused too many histories");
        assert!(tally.requests > 25_000, "{tally:?}");
        assert!(tally.same * 10 >= tally.compared * 9, "{tally:?}: the plans must mostly be the reference's");
        assert!(tally.other_session + tally.fork_for_place + tally.same_ids_other_choice > 0, "the counted differences never happen: the test says nothing");
        assert!(tally.prefix_sent * 2 > tally.requests, "{tally:?}: most requests carry a prefix");
        assert!(tally.gone > 50, "{tally:?}: the engine's cuts must reach the fallback");
    }

    /// Against an engine of protocol version 4 (no tail, no prefix), the same histories: every request is whole and the prompts are the reference's.
    #[test]
    fn against_an_engine_of_the_older_protocol_every_request_is_whole() {
        let tok = syn();
        let mut tally = Tally::default();
        for seed in 0..400u64 {
            if let Err(e) = scenario(seed, &tok, &mut tally, 4, false, false) {
                panic!("{e}\n{tally:?}");
            }
        }
        assert_eq!(tally.prefix_sent, 0);
        assert!(tally.requests > 800);
    }

    // ---------------------------------------------------------------- factors, sessions, the index

    const SYS: &str = "<｜begin▁of▁sentence｜><｜System｜>You are a careful assistant who answers briefly and well, and who never gives a long answer where a short one will do.";

    fn turn(prefix: &str, msg: &str) -> String {
        format!("{prefix}<｜User｜>{msg}<｜Assistant｜></think>")
    }

    fn reply(tok: &Tokenizer, s: &str) -> Vec<u32> {
        let mut r = tok.encode(s);
        r.push(1);
        r
    }

    #[test]
    fn the_factor_follows_the_prompts_token_count() {
        assert_eq!([0, 1, (1 << 20) - 1].map(factor_for), [16, 16, 16]);
        assert_eq!([1 << 20, (2 << 20) - 1].map(factor_for), [32, 32]);
        assert_eq!([2 << 20, 4 << 20, 9 << 20].map(factor_for), [64, 64, 64]);
    }

    #[test]
    fn a_match_never_crosses_factors() {
        let tok = syn();
        let text = "<｜begin▁of▁sentence｜>a conversation long enough to be matched, with its system prompt and a few turns<｜User｜>hi<｜Assistant｜></think>";
        let mut t = SessionTable::new(8, 1 << 20);
        put(&mut t, 1, text, false);
        t.sessions.get_mut(&1).unwrap().factor = 32;
        assert_eq!(best_of(&t, &tok, text, false, None, &[]), Some(1), "any factor: found");
        assert_eq!(best_of(&t, &tok, text, false, Some(32), &[]), Some(1));
        assert_eq!(best_of(&t, &tok, text, false, Some(16), &[]), None, "a factor-16 request never matches a factor-32 session");
        assert_eq!(best_of(&t, &tok, text, false, Some(64), &[]), None);
    }

    #[test]
    fn a_reply_that_crosses_a_boundary_labels_its_session_with_the_new_factor() {
        let tok = syn();
        let mut t = SessionTable::new(8, 64 << 20);
        let p1 = turn(SYS, "hello");
        let a = serve(&mut t, &tok, &p1, &[], false);
        assert_eq!(t.sessions[&a.session_id].factor, 16);
        let (b, _) = t.plan(&p1, &tok, 1, false);
        let big = vec![tok.encode("x")[0]; (1 << 20) + 8];
        finish_plan(&mut t, &tok, &b, &p1, &big, &[]);
        assert_eq!(t.sessions[&b.session_id].factor, 32, "prompt and reply together hold over 2^20 ids");
    }

    #[test]
    fn a_conversation_past_a_boundary_gets_a_new_session_and_the_old_one_stays() {
        let tok = syn();
        let mut t = SessionTable::new(8, 64 << 20);
        let p1 = turn(SYS, "hello");
        let a = serve(&mut t, &tok, &p1, &reply(&tok, "Hi."), false);
        assert_eq!(t.sessions[&a.session_id].factor, 16);
        // the next turn is over 1M ids (one id a byte here): factor 32, so session `a` is not offered
        let big = format!("{p1}Hi.<｜end▁of▁sentence｜><｜User｜>{}<｜Assistant｜></think>", "x".repeat((1 << 20) + 100));
        let (b, _) = t.plan(&big, &tok, 1, false);
        assert!(b.total() > 1 << 20);
        assert_ne!(b.session_id, a.session_id, "a new session, not the factor-16 one in place");
        assert_eq!((b.base_session_id, b.from.ids), (0, 0), "no fork of it and no id of its history");
        assert_eq!(t.sessions[&b.session_id].factor, 32);
        finish_plan(&mut t, &tok, &b, &big, &reply(&tok, "ok"), &[]);
        // the old record is kept: a sibling still under the boundary continues it in place
        let sib = turn(&format!("{p1}Hi.<｜end▁of▁sentence｜>"), "next");
        let (c, _) = t.plan(&sib, &tok, 1, false);
        assert_eq!((c.session_id, c.base_session_id), (a.session_id, 0));
        assert!(c.from.ids > 0);
        finish_plan(&mut t, &tok, &c, &sib, &reply(&tok, "ok"), &[]);
        // and the big conversation's next turn continues the new session
        let big2 = turn(&format!("{big}ok<｜end▁of▁sentence｜>"), "more");
        let (d, _) = t.plan(&big2, &tok, 1, false);
        assert_eq!(d.session_id, b.session_id);
    }

    #[test]
    fn the_saved_table_carries_each_sessions_factor() {
        let dir = table_dir("factor");
        let path = dir.join("front.table");
        let tok = syn();
        let mut t = SessionTable::new(8, 1 << 20);
        let text = "<｜begin▁of▁sentence｜>one conversation at factor thirty-two<｜User｜>hi<｜Assistant｜></think>";
        put_fresh(&mut t, 3, text, false);
        t.sessions.get_mut(&3).unwrap().factor = 32;
        assert_eq!(&t.snapshot(3600).0[..8], TABLE_MAGIC);
        assert_eq!(t.save(&path, 3600), Ok(1));
        let mut u = SessionTable::new(8, 1 << 20);
        assert_eq!(u.load(&path, &tok, 3600), Ok(1));
        assert_eq!(u.sessions[&3].factor, 32);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A saved table of an older layout holding one session of `text` (`DSFT0002` has the flags word, `DSFT0001` not).
    fn old_table(magic: &[u8; 8], id: u64, text: &str, ids: &[u32]) -> Vec<u8> {
        let now = unix_now();
        let mut body = Vec::new();
        let mut head = vec![id, 5, now, text.len() as u64, text.len() as u64];
        if magic == TABLE_MAGIC_V2 {
            head.push(0);
        }
        for v in head {
            body.extend(v.to_le_bytes());
        }
        body.extend(text.as_bytes());
        body.extend((ids.len() as u64).to_le_bytes());
        for x in ids {
            body.extend(x.to_le_bytes());
        }
        let mut f = magic.to_vec();
        for v in [now, id + 1, 9, 1, body.len() as u64, fnv(&body)] {
            f.extend(v.to_le_bytes());
        }
        f.extend(body);
        f
    }

    #[test]
    fn an_old_format_table_loads_at_factor_16() {
        let tok = syn();
        let text = "<｜begin▁of▁sentence｜>an older table's conversation, long enough to be matched by a prompt<｜User｜>hi<｜Assistant｜></think>";
        let ids = tok.encode(text);
        for (name, magic) in [("v2", TABLE_MAGIC_V2), ("v1", TABLE_MAGIC_V1)] {
            let dir = table_dir(name);
            let path = dir.join("front.table");
            write_table(&path, &old_table(magic, 7, text, &ids)).unwrap();
            let mut t = SessionTable::new(8, 1 << 20);
            assert_eq!(t.load(&path, &tok, 3600), Ok(1), "{name}");
            assert_eq!(t.sessions[&7].factor, 16, "{name}: every old record is factor 16");
            assert_eq!(best_of(&t, &tok, text, false, Some(16), &[]), Some(7));
            assert_eq!(best_of(&t, &tok, text, false, Some(32), &[]), None);
            std::fs::remove_dir_all(&dir).unwrap();
        }
    }

    #[test]
    fn the_id_limit_holds_four_whole_window_conversations_with_their_siblings() {
        let mut t = SessionTable::new(256, 16 << 20);
        let four_4m = 4 * (4 << 20);
        // what a 4M engine reports it keeps (HELLO keep_positions) alone would not do for four 4M conversations
        // plus the 1M session each left behind at its boundary
        t.keep_for(8 << 20, 4 << 20);
        assert_eq!(t.max_tokens, four_4m + 4 * (1 << 20));
        // a larger keep_positions wins; a 1M engine keeps the 16M-id floor
        t.keep_for(40 << 20, 4 << 20);
        assert_eq!(t.max_tokens, 40 << 20);
        t.keep_for(8 << 20, 1 << 20);
        assert_eq!(t.max_tokens, 16 << 20);
    }

    #[test]
    fn a_periodic_save_writes_only_after_a_change() {
        let dir = table_dir("periodic");
        let path = dir.join("front.table");
        let table = std::sync::Mutex::new(SessionTable::new(8, 1 << 20));
        let mut saved = table.lock().unwrap().changes();
        // nothing changed: nothing written
        assert!(save_if_changed(&table, &mut saved, &path, 3600).is_none());
        assert!(!path.exists());
        // a change: the file is written, and a second look finds nothing new
        {
            let mut t = table.lock().unwrap();
            put_fresh(&mut t, 7, "<｜begin▁of▁sentence｜>a conversation long enough to be kept<｜User｜>hi<｜Assistant｜></think>", false);
            t.changes += 1;
        }
        assert_eq!(save_if_changed(&table, &mut saved, &path, 3600), Some(Ok(1)));
        assert!(path.exists() && !dir.join("front.table.new").exists());
        assert_eq!(std::fs::metadata(&path).unwrap().mode() & 0o777, 0o600);
        std::fs::remove_file(&path).unwrap();
        assert!(save_if_changed(&table, &mut saved, &path, 3600).is_none());
        assert!(!path.exists(), "no change, no write");
        // a failed write is tried again at the next call
        let bad = dir.join("missing/front.table");
        table.lock().unwrap().changes += 1;
        assert!(matches!(save_if_changed(&table, &mut saved, &bad, 3600), Some(Err(_))));
        assert_eq!(save_if_changed(&table, &mut saved, &path, 3600), Some(Ok(1)));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_planned_and_a_finished_request_count_as_changes() {
        let tok = syn();
        let mut t = SessionTable::new(8, 1 << 20);
        let c0 = t.changes();
        let p = turn(SYS, "hello");
        let (plan, _) = t.plan(&p, &tok, 1, false);
        assert!(t.changes() > c0);
        let c1 = t.changes();
        finish_plan(&mut t, &tok, &plan, &p, &reply(&tok, "Hi."), &[]);
        assert!(t.changes() > c1);
        let c2 = t.changes();
        t.abandon(plan.session_id);
        assert!(t.changes() > c2);
    }

    #[test]
    fn a_leftover_temporary_file_does_not_touch_the_saved_table() {
        let dir = table_dir("leftover");
        let path = dir.join("front.table");
        let mut t = SessionTable::new(8, 1 << 20);
        put_fresh(&mut t, 3, "<｜begin▁of▁sentence｜>the first table's conversation<｜User｜>hi<｜Assistant｜></think>", false);
        assert_eq!(t.save(&path, 3600), Ok(1));
        let first = std::fs::read(&path).unwrap();
        // a crash in the middle of the next save: half a file under the temporary name
        let tmp = dir.join("front.table.new");
        std::fs::write(&tmp, &first[..first.len() / 2]).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), first, "the saved file is as it was");
        // the next save replaces the leftover and the file whole
        put_fresh(&mut t, 4, "<｜begin▁of▁sentence｜>and a second conversation<｜User｜>hi<｜Assistant｜></think>", false);
        assert_eq!(t.save(&path, 3600), Ok(2));
        assert!(!tmp.exists());
        assert_ne!(std::fs::read(&path).unwrap(), first);
        // at a load: the previous table loads, the leftover goes, the table stays
        std::fs::write(&path, &first).unwrap();
        std::fs::write(&tmp, b"half").unwrap();
        let tok = syn();
        let mut u = SessionTable::new(8, 1 << 20);
        assert!(u.load(&path, &tok, 3600).is_ok(), "the file itself is taken whole");
        assert!(!tmp.exists() && path.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_session_in_a_request_is_saved_as_its_last_finished_turn() {
        let tok = syn();
        let dir = table_dir("busy");
        let path = dir.join("front.table");
        let mut t = SessionTable::new(8, 1 << 20);
        let p1 = turn(SYS, "hello");
        let (plan, _) = t.plan(&p1, &tok, 1, false);
        assert_eq!(t.snapshot(3600).1, 0, "a session with no finished turn has nothing to save");
        let gen = reply(&tok, "Hi there.");
        finish_plan(&mut t, &tok, &plan, &p1, &gen, &[]);
        assert_eq!(t.save(&path, 3600), Ok(1));
        // the next turn of the conversation is in its request: the session is busy, and still saved
        let p2 = turn(&format!("{p1}Hi there.<｜end▁of▁sentence｜>"), "next");
        let (plan2, _) = t.plan(&p2, &tok, 1, false);
        assert_eq!(plan2.session_id, plan.session_id);
        assert!(t.sessions[&plan.session_id].busy);
        assert_eq!(t.save(&path, 3600), Ok(1));
        let mut u = SessionTable::new(8, 1 << 20);
        assert_eq!(u.load(&path, &tok, 3600), Ok(1));
        let s = &u.sessions[&plan.session_id];
        assert_eq!(s.prompt_len, p1.len(), "its last finished turn, not the turn in flight");
        assert_eq!(s.n_hist as usize, plan.total() + gen.len() - 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_load_leaves_the_file_so_a_second_load_finds_the_same_table() {
        let tok = syn();
        let dir = table_dir("twice");
        let path = dir.join("front.table");
        let mut t = SessionTable::new(8, 1 << 20);
        let p1 = turn(SYS, "hello");
        let plan = serve(&mut t, &tok, &p1, &reply(&tok, "Hi there."), false);
        assert_eq!(t.save(&path, 3600), Ok(1));
        let bytes = std::fs::read(&path).unwrap();
        let mut a = SessionTable::new(8, 1 << 20);
        assert_eq!(a.load(&path, &tok, 3600), Ok(1));
        assert_eq!(std::fs::read(&path).unwrap(), bytes, "the file is as it was");
        // a crash right after the start: the next start loads the same table
        let mut b = SessionTable::new(8, 1 << 20);
        assert_eq!(b.load(&path, &tok, 3600), Ok(1));
        assert_eq!(b.sessions[&plan.session_id].cuts, a.sessions[&plan.session_id].cuts);
        // the loaded table counts as changed, so the first periodic save writes it again
        let table = std::sync::Mutex::new(b);
        let mut saved = 0;
        assert_eq!(save_if_changed(&table, &mut saved, &path, 3600), Some(Ok(1)));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_saved_table_round_trips() {
        let dir = table_dir("roundtrip");
        let path = dir.join("front.table");
        let mut t = SessionTable::new(8, 1 << 20);
        put_fresh(&mut t, 3, "<｜begin▁of▁sentence｜>one conversation<｜User｜>hi<｜Assistant｜></think>", false);
        put_fresh(&mut t, 9, "<｜begin▁of▁sentence｜>another, ablated<｜User｜>hi<｜Assistant｜></think>", true);
        assert_eq!(t.save(&path, 3600), Ok(2));
        // the file is exactly the snapshot, and saving the unchanged table again gives the same sessions
        let (snap, n) = t.snapshot(3600);
        assert_eq!(n, 2);
        let on_disk = std::fs::read(&path).unwrap();
        // the header's first field is the write time; the rest is the table
        assert_eq!(on_disk[16..], snap[16..]);
        assert_eq!(on_disk.len(), snap.len());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_index_finds_the_session_that_shares_the_latest_cut() {
        // conversations that share beginnings of every length (a common system prompt, then branches at added tokens), against every
        // session read whole: the index's best match shares the latest cut that any session shares with the prompt
        let tok = syn();
        let mut seed = 7u64;
        let seg = |r: &mut u64| -> String { format!("<｜User｜>{}<｜Assistant｜></think>{}<｜end▁of▁sentence｜>", words_r(r, 1, 12), words_r(r, 1, 40)) };
        let sys: String = format!("<｜begin▁of▁sentence｜><｜System｜>{}", words(&mut seed, 40));
        let mut t = SessionTable::new(1 << 20, 1 << 30);
        let mut texts: Vec<(u64, String, bool)> = Vec::new();
        for id in 1..=300u64 {
            let base = if texts.is_empty() || lcg(&mut seed).is_multiple_of(4) {
                sys.clone()
            } else {
                // a prefix of another conversation that ends where a segment ends
                let (_, b, _) = &texts[lcg(&mut seed) as usize % texts.len()];
                let ends: Vec<usize> = b.match_indices("<｜end▁of▁sentence｜>").map(|(i, m)| i + m.len()).collect();
                if ends.is_empty() { sys.clone() } else { b[..ends[lcg(&mut seed) as usize % ends.len()]].to_string() }
            };
            let mut text = base;
            for _ in 0..1 + lcg(&mut seed) % 5 {
                text.push_str(&seg(&mut seed));
            }
            let ablit = lcg(&mut seed).is_multiple_of(5);
            put(&mut t, id, &text, ablit);
            texts.push((id, text, ablit));
        }
        let cuts_of = |text: &str| -> Vec<usize> { scan(text, &Resume::start(), 0, &tok).0.iter().map(|e| e.bytes).collect() };
        let sets: Vec<std::collections::HashSet<usize>> = texts.iter().map(|x| cuts_of(&x.1).into_iter().collect()).collect();
        for q in 0..1500 {
            let (_, b, _) = &texts[lcg(&mut seed) as usize % texts.len()];
            let ends: Vec<usize> = b.match_indices("<｜end▁of▁sentence｜>").map(|(i, m)| i + m.len()).collect();
            let mut cut_at = if ends.is_empty() || q % 3 == 0 { lcg(&mut seed) as usize % (b.len() + 1) } else { ends[lcg(&mut seed) as usize % ends.len()] };
            while !b.is_char_boundary(cut_at) {
                cut_at -= 1;
            }
            let mut p = b[..cut_at].to_string();
            p.push_str(&seg(&mut seed));
            let ablit = q % 7 == 0;
            let req_cuts = cuts_of(&p);
            // what a session shares at least: the deepest index level its text agrees at, or its latest cut shared with the prompt
            let brute = texts
                .iter()
                .zip(&sets)
                .filter(|(x, _)| x.2 == ablit)
                .map(|(x, theirs)| {
                    let n = common_prefix(p.as_bytes(), x.1.as_bytes());
                    let cut = req_cuts.iter().rev().copied().find(|e| *e <= n && theirs.contains(e)).unwrap_or(0);
                    let level = (0..).map(level_len).take_while(|&l| l <= n).last().unwrap_or(0);
                    cut.max(level)
                })
                .max()
                .unwrap_or(0);
            let got = t.pick(&Prompt::new(&p, &tok, ablit, &[]), None).top.map_or(0, |(_, n)| n);
            if brute >= MIN_SHARED {
                assert_eq!(got, brute, "prompt {q} of {} bytes", p.len());
            } else {
                assert!(got < MIN_SHARED, "prompt {q}: {got} found where nothing shares {MIN_SHARED} bytes");
            }
        }
        // removed sessions leave the index: nothing names them
        for id in 1..=300u64 {
            t.remove(id);
        }
        assert!(t.index.is_empty() && t.sessions.is_empty());
    }

    #[test]
    fn the_table_keeps_what_the_engine_keeps() {
        let mut t = SessionTable::new(256, 16 << 20);
        t.keep_for(0, 1 << 20); // an engine that keeps nothing (or an echo engine): the table's own limits, never less
        assert_eq!((t.max_sessions, t.max_tokens), (256, 16 << 20));
        t.keep_for(300_000_000, 1 << 20); // an engine that reports about 300M positions
        assert_eq!((t.max_sessions, t.max_tokens), (300_000_000 / 4096, 300_000_000));
    }

    /// The ids of a conversation's next turn are the engine's own: what the model spelled stays as it spelled it, and the request names
    /// the cut before its last id (the last sampled id is never in the engine's history) and carries that id again with the new text.
    #[test]
    fn reuses_the_models_own_ids() {
        let Ok(real) = Tokenizer::from_file(TOKENIZER) else { return };
        {
            let tok = &real;
            let mut rig = Rig::new(tok, 5, false, 3);
            let sys = "<｜begin▁of▁sentence｜><｜System｜>You are a careful assistant who answers briefly and well.";
            let p1 = format!("{sys}<｜User｜>hello<｜Assistant｜></think>");
            let first = rig.turn(&p1, false, &[], &{
                // the model spelled "Hi there" in single letters, which BPE would never produce
                let mut gen: Vec<u32> = "Hi there".chars().map(|c| tok.encode(&c.to_string())[0]).collect();
                gen.push(tok.token_id("<｜end▁of▁sentence｜>").unwrap());
                gen
            });
            assert_eq!(first.built, tok.encode(&p1));
            let p2 = format!("{p1}Hi there<｜end▁of▁sentence｜><｜User｜>next<｜Assistant｜></think>");
            let second = rig.turn(&p2, false, &[], &[1]);
            assert_eq!(second.plan.session_id, first.plan.session_id);
            assert!(second.prefix, "the second turn names the entry");
            let mut want = first.built.clone();
            want.extend("Hi there".chars().map(|c| tok.encode(&c.to_string())[0]));
            assert_eq!(second.plan.from.ids as usize, want.len(), "the cut is before the end id");
            want.extend(tok.encode("<｜end▁of▁sentence｜><｜User｜>next<｜Assistant｜></think>"));
            assert_eq!(second.built, want);
            assert_eq!(tok.decode(&second.built), p2);
            // the same system prompt in another conversation forks instead of taking over the session
            let p3 = format!("{sys}<｜User｜>something else entirely<｜Assistant｜></think>");
            let third = rig.turn(&p3, false, &[], &[1]);
            assert_ne!(third.plan.session_id, first.plan.session_id);
            assert_eq!(third.plan.base_session_id, first.plan.session_id);
            assert_eq!(tok.decode(&third.built), p3);
        }
    }

    /// The model spelled a special token's text in pieces, which no encoder of the text would: the engine holds those ids, the cut after them
    /// counts them, and the next request's prompt keeps them (it is not the text tokenised again).
    #[test]
    fn the_models_own_ids_are_kept_where_the_text_tokenised_again_would_differ() {
        let tok = syn();
        let mut rig = Rig::new(&tok, 5, false, 3);
        let p1 = turn(SYS, "hello");
        let mut gen: Vec<u32> = "ok <｜User｜> then".chars().flat_map(|c| tok.encode(&c.to_string())).collect();
        gen.push(1);
        let first = rig.turn(&p1, false, &[], &gen);
        let p2 = turn(&format!("{p1}ok <｜User｜> then<｜end▁of▁sentence｜>"), "next");
        let second = rig.turn(&p2, false, &[], &[1]);
        assert!(second.prefix);
        assert_eq!(second.plan.from.ids as usize, first.built.len() + gen.len() - 1, "the cut counts the ids the model spelled");
        assert_eq!(tok.decode(&second.built), p2);
        assert_ne!(second.built, tok.encode(&p2), "the model's spelling stays: the prompt is not the text tokenised again");
        assert_eq!(&second.built[..first.built.len() + gen.len() - 1], &[&first.built[..], &gen[..gen.len() - 1]].concat()[..]);
    }

    #[test]
    fn the_plan_says_where_it_starts() {
        let tok = syn();
        let mut t = SessionTable::new(8, 1 << 20);
        // pi-like: a long system text that ends with the time, then the tools and the first message
        let body = "You are an expert coding assistant. Read files, run commands and edit code.\n".repeat(40);
        let conv = |time: &str, msg: &str| {
            format!("<｜begin▁of▁sentence｜><｜System｜>{body}Current date and time: Monday, October 12, 2026 at {time}\n\n## Tools\n\nnone<｜User｜>{msg}<｜Assistant｜></think>")
        };
        let a = conv("09:14:08 AM", "fix the build");
        let (pa, _) = t.plan(&a, &tok, 1, false);
        assert_eq!(pa.describe(a.len()), format!("session {}, new: no session shares {MIN_SHARED} bytes", pa.session_id));
        finish_plan(&mut t, &tok, &pa, &a, &reply(&tok, "Done."), &[]);
        // another conversation, a minute later: a fork of the first, which has every id before the time
        let b = conv("09:15:31 AM", "write the tests");
        let (pb, _) = t.plan(&b, &tok, 1, false);
        assert_eq!(pb.base_session_id, pa.session_id);
        let head = b.find("<｜System｜>").unwrap() + "<｜System｜>".len();
        assert!(pb.from.bytes == head, "the cut is the last edge before the time: {} of {head}", pb.from.bytes);
        let d = pb.describe(b.len());
        assert!(d.starts_with(&format!("session {}, a fork of {}: it shares ", pb.session_id, pa.session_id)), "{d}");
        assert!(d.ends_with(&format!("{} of {} bytes", pb.shared, b.len())) && !d.contains("in common"), "{d}");
        // the next turn of the first: in place
        let a2 = turn(&format!("{a}Done.<｜end▁of▁sentence｜>"), "thanks");
        let (pa2, _) = t.plan(&a2, &tok, 1, false);
        assert_eq!(pa2.session_id, pa.session_id);
        assert!(pa2.describe(a2.len()).starts_with(&format!("session {} in place: it shares ", pa.session_id)));
        finish_plan(&mut t, &tok, &pa2, &a2, &reply(&tok, "Sure."), &[]);
        // a prompt that shares under 64 bytes with a session: a new session, which names no base and no session as a match
        let c = format!("<｜begin▁of▁sentence｜><｜System｜>{}<｜User｜>hi<｜Assistant｜></think>", &body[..18]);
        let (pc, _) = t.plan(&c, &tok, 1, false);
        assert_eq!((pc.base_session_id, pc.matched), (0, 0));
        assert_eq!(pc.describe(c.len()), format!("session {}, new: no session shares {MIN_SHARED} bytes", pc.session_id));
    }

    #[test]
    fn the_log_line_for_a_session_that_shares_too_little_names_it() {
        let p = Plan { session_id: 7, base_session_id: 0, from: Cut::START, epoch: 0, state: Resume::start(), suffix: Vec::new(), matched: 5, shared: 30 };
        assert_eq!(p.describe(100), "session 7, new: the best match, session 5, shares 30 bytes (under 64)");
    }

    #[test]
    fn a_saved_table_finds_its_sessions_again() {
        let tok = syn();
        let dir = std::env::temp_dir().join(format!("dsfront-table-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("front.table");
        let mut t = SessionTable::new(8, 1 << 20);
        let p1 = turn(SYS, "hello");
        let gen = reply(&tok, "Hi there.");
        let plan = serve(&mut t, &tok, &p1, &gen, false);
        let (busy, _) = t.plan("<｜begin▁of▁sentence｜><｜System｜>Another conversation, still in its request, which is not saved.", &tok, 1, false);
        let repo = dir.join("repo");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        std::fs::create_dir(repo.join("run")).unwrap();
        let e = t.save(&repo.join("run/front.table"), 3600).unwrap_err();
        assert!(e.contains("git tree"), "{e}");
        assert!(!repo.join("run/front.table").exists() && !repo.join("run/front.table.new").exists());
        std::fs::remove_dir_all(&repo).unwrap();
        assert_eq!(t.save(&path, 3600), Ok(1));
        let m = std::fs::metadata(&path).unwrap();
        assert_eq!(m.mode() & 0o777, 0o600);
        assert_eq!(std::fs::metadata(&dir).unwrap().mode() & 0o777, 0o700);
        let mut u = SessionTable::new(8, 1 << 20);
        assert_eq!(u.load(&path, &tok, 3600), Ok(1));
        assert!(path.exists(), "the file stays until a save replaces it");
        let mut again = SessionTable::new(8, 1 << 20);
        assert_eq!(again.load(&path, &tok, 3600), Ok(1), "a crash right after the start finds the same table");
        assert_eq!(again.next_id, u.next_id);
        assert!(u.next_id > busy.session_id, "ids go on after the saved table's");
        let p2 = turn(&format!("{p1}Hi there.<｜end▁of▁sentence｜>"), "next");
        let (plan2, _) = u.plan(&p2, &tok, 1, false);
        assert_eq!(plan2.session_id, plan.session_id);
        assert_eq!(plan2.from.ids as usize, plan.total() + gen.len() - 1);
        assert_eq!(plan2.epoch, 5, "the epoch the engine gave survives the restart of the front-end");
        // a second load finds the file; with none there nothing is taken
        assert_eq!(u.load(&path, &tok, 3600), Ok(1));
        std::fs::remove_file(&path).unwrap();
        assert_eq!(u.load(&path, &tok, 3600), Err(String::new()));
        // a damaged file, a file open to others: nothing taken
        t.save(&path, 3600).unwrap();
        let mut b = std::fs::read(&path).unwrap();
        let k = b.len() - 3;
        b[k] ^= 1;
        std::fs::write(&path, &b).unwrap();
        assert_eq!(SessionTable::new(8, 1 << 20).load(&path, &tok, 3600), Err("incomplete or damaged".into()));
        assert!(path.exists(), "a damaged file stays until a save replaces it");
        t.save(&path, 3600).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(SessionTable::new(8, 1 << 20).load(&path, &tok, 3600).is_err());
        // a table saved long ago, a session not used within the limit: left out
        t.save(&path, 3600).unwrap();
        let mut b = std::fs::read(&path).unwrap();
        b[8..16].copy_from_slice(&1u64.to_le_bytes());
        std::fs::write(&path, &b).unwrap();
        assert!(SessionTable::new(8, 1 << 20).load(&path, &tok, 3600).unwrap_err().starts_with("saved"));
        t.sessions.get_mut(&plan.session_id).unwrap().t_used = 1;
        assert_eq!(t.save(&path, 3600), Ok(0));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn sessions_are_kept_per_weights_path() {
        let tok = syn();
        let mut t = SessionTable::new(8, 1 << 20);
        let p1 = turn(SYS, "hello");
        let plain = serve(&mut t, &tok, &p1, &reply(&tok, "Hi."), false);
        // the same prompt on the other path: a new session that names no base, and reuses nothing
        let (abl, _) = t.plan(&p1, &tok, 1, true);
        assert_ne!(abl.session_id, plain.session_id);
        assert_eq!((abl.base_session_id, abl.from.ids, abl.matched), (0, 0, 0));
        finish_plan(&mut t, &tok, &abl, &p1, &reply(&tok, "Hey."), &[]);
        // the next turn of each conversation continues its own session, in place
        let n1 = format!("{p1}Hi.<｜end▁of▁sentence｜>");
        let (c_plain, _) = t.plan(&turn(&n1, "more"), &tok, 1, false);
        assert_eq!((c_plain.session_id, c_plain.base_session_id), (plain.session_id, 0));
        let n2 = format!("{p1}Hey.<｜end▁of▁sentence｜>");
        let (c_abl, _) = t.plan(&turn(&n2, "more"), &tok, 1, true);
        assert_eq!((c_abl.session_id, c_abl.base_session_id), (abl.session_id, 0));
        // a fork takes its base from the request's own path
        finish_plan(&mut t, &tok, &c_abl, &turn(&n2, "more"), &reply(&tok, "Ok."), &[]);
        finish_plan(&mut t, &tok, &c_plain, &turn(&n1, "more"), &reply(&tok, "Ok."), &[]);
        let other = turn(SYS, "something else entirely");
        let (f_abl, _) = t.plan(&other, &tok, 1, true);
        assert_eq!(f_abl.base_session_id, c_abl.session_id);
        let (f_plain, _) = t.plan(&other, &tok, 1, false);
        assert_eq!(f_plain.base_session_id, c_plain.session_id);
        // a busy session of the other path is no candidate either
        let lonely_text = "<｜begin▁of▁sentence｜><｜System｜>A prompt that only the ablit path has seen so far, long enough to share.";
        let (lonely, _) = t.plan(&turn(lonely_text, "x"), &tok, 1, true);
        let (twin, _) = t.plan(&turn(lonely_text, "y"), &tok, 1, false);
        assert_eq!(twin.base_session_id, 0);
        assert!(twin.matched != lonely.session_id && twin.session_id != lonely.session_id);
    }

    /// The first table layout, written by hand: a session has five words before its text and no flags.
    fn write_v1_table(path: &Path, sessions: &[(u64, &str, &[u32])]) {
        let mut body = Vec::new();
        for &(id, text, ids) in sessions {
            for v in [id, 1u64, unix_now(), text.len() as u64, text.len() as u64] {
                body.extend(v.to_le_bytes());
            }
            body.extend(text.as_bytes());
            body.extend((ids.len() as u64).to_le_bytes());
            for &x in ids {
                body.extend(x.to_le_bytes());
            }
        }
        let mut f = TABLE_MAGIC_V1.to_vec();
        for v in [unix_now(), 1000, 5, sessions.len() as u64, body.len() as u64, fnv(&body)] {
            f.extend(v.to_le_bytes());
        }
        f.extend(body);
        let _ = std::fs::remove_file(path);
        let mut o = std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(path).unwrap();
        o.write_all(&f).unwrap();
    }

    #[test]
    fn a_saved_table_carries_the_weights_path() {
        let tok = syn();
        let dir = std::env::temp_dir().join(format!("dsfront-flag-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("front.table");
        let mut t = SessionTable::new(8, 1 << 20);
        let p1 = turn(SYS, "hello");
        let gen = reply(&tok, "Hi.");
        let plain = serve(&mut t, &tok, &p1, &gen, false);
        let p2 = turn("<｜begin▁of▁sentence｜><｜System｜>Another conversation altogether, on the ablit path, which is as long as the other one.", "hello");
        let abl = serve(&mut t, &tok, &p2, &gen, true);
        assert_eq!(t.save(&path, 3600), Ok(2));
        let mut u = SessionTable::new(8, 1 << 20);
        assert_eq!(u.load(&path, &tok, 3600), Ok(2));
        // each is found again by a request of its own path, and by that one only
        let next = |p: &str| turn(&format!("{p}Hi.<｜end▁of▁sentence｜>"), "again");
        let (a, _) = u.plan(&next(&p2), &tok, 1, true);
        assert_eq!((a.session_id, a.base_session_id), (abl.session_id, 0));
        let (b, _) = u.plan(&next(&p1), &tok, 1, false);
        assert_eq!((b.session_id, b.base_session_id), (plain.session_id, 0));
        let (c, _) = u.plan(&next(&p1), &tok, 1, true);
        assert_eq!((c.from.ids, c.base_session_id), (0, 0));
        // a table of the first layout loads, every session on the plain path
        let ids = tok.encode(&p1);
        write_v1_table(&path, &[(77, &p1, &ids)]);
        let mut v = SessionTable::new(8, 1 << 20);
        assert_eq!(v.load(&path, &tok, 3600), Ok(1));
        let (on, _) = v.plan(&turn(&p1, "more"), &tok, 1, true);
        assert_eq!(on.from.ids, 0, "an older table's session is not on the ablit path");
        let mut w = SessionTable::new(8, 1 << 20);
        write_v1_table(&path, &[(77, &p1, &ids)]);
        assert_eq!(w.load(&path, &tok, 3600), Ok(1));
        let (off, _) = w.plan(&turn(&p1, "more"), &tok, 1, false);
        assert_eq!((off.session_id, off.from.ids as usize, off.epoch), (77, ids.len(), 0));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // ---------------------------------------------------------------- images

    const PH: &str = crate::encoding::IMAGE_PLACEHOLDER;

    /// A prompt with images as the server prepares it: the text with each placeholder written once per
    /// row, and the images' keys (`hashes` in order, `rows` rows each).
    fn with_images(raw: &str, rows: u32, hashes: &[u64]) -> (String, Vec<ImageKey>) {
        let (text, ats) = crate::vision::expand_text(raw, &vec![rows; hashes.len()]).unwrap();
        let keys = ats.iter().zip(hashes).map(|(&at, &hash)| ImageKey { at, rows, hash }).collect();
        (text, keys)
    }

    #[test]
    fn images_are_matched_by_content() {
        let tok = crate::tokenizer::synthetic::tokenizer_with_images();
        let mut rig = Rig::new(&tok, 5, false, 1);
        let first = format!("{SYS}<｜User｜>what is on {PH}?<｜Assistant｜></think>");
        let (x1, k1) = with_images(&first, 3, &[11]);
        let gen = reply(&tok, "A cat.");
        let p1 = rig.turn(&x1, false, &k1, &gen);
        assert_eq!((p1.plan.base_session_id, p1.plan.from.ids), (0, 0));
        assert_eq!(tok.decode(&p1.built), x1, "the ids spell the expanded text");
        assert_eq!(p1.built.iter().filter(|&&i| i == tok.token_id(PH).unwrap()).count(), 3);
        // the client re-sends its history with the same image: the next turn continues the session in place; the images' spans are
        // positions in the whole prompt, so a request with images carries the whole prompt
        let hist = format!("{first}A cat.<｜end▁of▁sentence｜>");
        let (x2, k2) = with_images(&turn(&hist, "and its colour?"), 3, &[11]);
        let p2 = rig.turn(&x2, false, &k2, &reply(&tok, "Grey."));
        assert_eq!((p2.plan.session_id, p2.plan.base_session_id), (p1.plan.session_id, 0));
        assert_eq!(p2.plan.from.ids as usize, p1.built.len() + gen.len() - 1, "the plan continues from the cut before the reply's last id");
        assert!(!p2.prefix, "no prefix with images");
        assert_eq!(tok.decode(&p2.built), x2);
        // the same text with another image at that place: the cut before the image is shared, none after it; the ids are a cold run's
        let (x3, k3) = with_images(&turn(&hist, "and its colour?"), 3, &[12]);
        assert_eq!(x3, x2);
        let p3 = rig.turn(&x3, false, &k3, &gen);
        assert!(p3.plan.from.bytes <= first.find(PH).unwrap(), "a cut before the changed image: {}", p3.plan.from.bytes);
        assert_eq!(p3.built, p2.built, "the cold ids are the continued session's");
    }

    /// The encoder's kind is part of an image's hash (`vision::ENGINE_ENCODER_KIND`): the same image made by the container and by the
    /// engine is two images to the table, so a session made under one is never continued under the other, in either direction.
    #[test]
    fn a_session_made_under_one_encoder_kind_is_not_continued_under_the_other() {
        use crate::vision::ENGINE_ENCODER_KIND as KIND;
        let tok = crate::tokenizer::synthetic::tokenizer_with_images();
        for (made, then, name) in [(0u64, KIND, "container then engine"), (KIND, 0, "engine then container")] {
            let mut rig = Rig::new(&tok, 5, false, 1);
            let first = format!("{SYS}<｜User｜>what is on {PH}?<｜Assistant｜></think>");
            let (x1, k1) = with_images(&first, 3, &[11 ^ made]);
            let gen = reply(&tok, "A cat.");
            let p1 = rig.turn(&x1, false, &k1, &gen);
            let hist = format!("{first}A cat.<｜end▁of▁sentence｜>");
            // the same image, the same kind: continued in place (the control)
            let (x2, k2) = with_images(&turn(&hist, "and its colour?"), 3, &[11 ^ made]);
            let p2 = rig.turn(&x2, false, &k2, &reply(&tok, "Grey."));
            assert_eq!(p2.plan.session_id, p1.plan.session_id, "{name}: the same kind continues the session");
            // the same image under the other kind: no candidate past the image; a cut at most before it
            let (x3, k3) = with_images(&turn(&hist, "and its colour?"), 3, &[11 ^ then]);
            let p3 = rig.turn(&x3, false, &k3, &gen);
            assert!(p3.plan.from.bytes <= first.find(PH).unwrap(), "{name}: a cut before the image, not after it: {}", p3.plan.from.bytes);
        }
    }

    #[test]
    fn an_image_after_the_shared_text_keeps_the_earlier_turns() {
        let tok = crate::tokenizer::synthetic::tokenizer_with_images();
        let mut t = SessionTable::new(8, 1 << 20);
        let p1 = turn(SYS, "hello");
        let gen = reply(&tok, "Hi.");
        let plain = serve(&mut t, &tok, &p1, &gen, false);
        // the second turn brings the first image: it lies after everything the session holds
        let (x2, k2) = with_images(&turn(&format!("{p1}Hi.<｜end▁of▁sentence｜>"), &format!("see {PH}")), 4, &[7]);
        let (p2, _) = t.plan_images(&x2, &tok, 1, false, &k2);
        assert_eq!((p2.session_id, p2.base_session_id), (plain.session_id, 0));
        assert_eq!(p2.from.ids as usize, plain.total() + gen.len() - 1);
        finish_plan(&mut t, &tok, &p2, &x2, &reply(&tok, "A dog."), &k2);
        // a different conversation with the same system prompt and no image forks the session: its
        // shared text ends before the image
        let other = turn(SYS, "something else entirely");
        let (f, _) = t.plan(&other, &tok, 1, false);
        assert_eq!(f.base_session_id, plain.session_id);
        // a conversation that holds the image too, at the same place, forks it with the image
        let (x4, k4) = with_images(&turn(&format!("{p1}Hi.<｜end▁of▁sentence｜>"), &format!("see {PH} now")), 4, &[7]);
        let (p4, _) = t.plan_images(&x4, &tok, 1, false, &k4);
        assert_eq!(p4.base_session_id, plain.session_id);
        assert!(p4.from.ids as usize >= plain.total() + gen.len() - 1);
    }

    #[test]
    fn image_sessions_are_kept_per_weights_path() {
        let tok = crate::tokenizer::synthetic::tokenizer_with_images();
        let mut t = SessionTable::new(8, 1 << 20);
        let raw = format!("{SYS}<｜User｜>what is on {PH}?<｜Assistant｜></think>");
        let (x, k) = with_images(&raw, 3, &[5]);
        let plain = serve_images(&mut t, &tok, &x, &reply(&tok, "Ok."), false, &k);
        let (abl, _) = t.plan_images(&x, &tok, 1, true, &k);
        assert_ne!(abl.session_id, plain.session_id);
        assert_eq!((abl.base_session_id, abl.from.ids, abl.matched), (0, 0, 0));
        finish_plan(&mut t, &tok, &abl, &x, &reply(&tok, "Ok."), &k);
        // each path continues its own session
        let (x2, k2) = with_images(&turn(&format!("{raw}Ok.<｜end▁of▁sentence｜>"), "more"), 3, &[5]);
        let (a, _) = t.plan_images(&x2, &tok, 1, true, &k2);
        assert_eq!((a.session_id, a.base_session_id), (abl.session_id, 0));
        let (b, _) = t.plan_images(&x2, &tok, 1, false, &k2);
        assert_eq!((b.session_id, b.base_session_id), (plain.session_id, 0));
    }

    #[test]
    fn a_saved_table_keeps_its_images() {
        let tok = crate::tokenizer::synthetic::tokenizer_with_images();
        let dir = std::env::temp_dir().join(format!("dsfront-images-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("front.table");
        let mut t = SessionTable::new(8, 1 << 20);
        let first = format!("{SYS}<｜User｜>what is on {PH}?<｜Assistant｜></think>");
        let (x1, k1) = with_images(&first, 3, &[11]);
        let gen = reply(&tok, "A cat.");
        let p1 = serve_images(&mut t, &tok, &x1, &gen, false, &k1);
        assert_eq!(t.save(&path, 3600), Ok(1));
        let mut u = SessionTable::new(8, 1 << 20);
        assert_eq!(u.load(&path, &tok, 3600), Ok(1));
        let hist = format!("{first}A cat.<｜end▁of▁sentence｜>");
        let (x2, k2) = with_images(&turn(&hist, "and its colour?"), 3, &[11]);
        let (p2, _) = u.plan_images(&x2, &tok, 1, false, &k2);
        assert_eq!((p2.session_id, p2.from.ids as usize), (p1.session_id, p1.total() + gen.len() - 1));
        let (x3, k3) = with_images(&turn(&hist, "and its colour?"), 3, &[12]);
        let (p3, _) = u.plan_images(&x3, &tok, 1, false, &k3);
        assert!(p3.from.bytes <= first.find(PH).unwrap(), "no cut past the changed image");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The same lockstep as the text histories, with images: the same image, another image at the same place, an image added later, the
    /// other weights path. A request with images carries its whole prompt, so what the engine builds is the reference's ids; a candidate whose
    /// images inside the shared text differ is no candidate (the reference plan starts cold where a cut before the image is still shared).
    #[test]
    fn image_histories_build_the_same_prompts_as_the_reference() {
        let tok = crate::tokenizer::synthetic::tokenizer_with_images();
        let mut requests = 0;
        for seed in 0..600u64 {
            let mut rng = seed.wrapping_mul(0x9e3779b97f4a7c15) | 1;
            let mut rig = Rig::new(&tok, 5, false, seed);
            let mut reft = RefTable::default();
            let rows = 3 + lcg(&mut rng) as u32 % 3;
            let mut hist = format!("{SYS}<｜User｜>what is on {PH}?<｜Assistant｜></think>");
            let mut base = 11u64;
            for k in 0..4 {
                let hashes: Vec<u64> = (0..hist.matches(PH).count() as u64).map(|i| base + i).collect();
                let (text, keys) = with_images(&hist, rows, &hashes);
                let ablit = k > 0 && lcg(&mut rng).is_multiple_of(9);
                let gen = reply(&tok, &words(&mut rng, 4));
                let sent = rig.turn(&text, ablit, &keys, &gen);
                let rp = reft.plan(&text, &tok, ablit, &keys);
                requests += 1;
                assert_eq!(tok.decode(&sent.built), text, "seed {seed} request {k}");
                assert!(!sent.prefix && !sent.gone);
                assert_eq!(sent.built, rp.ids, "seed {seed} request {k}: the prompts differ");
                reft.finish(&rp, &text, &gen, &tok, ablit, &keys);
                let said = tok.decode(&gen);
                hist = turn(&format!("{hist}{said}"), &format!("again {}", lcg(&mut rng) % 100));
                match lcg(&mut rng) % 4 {
                    0 => base = 12 + lcg(&mut rng) % 3, // another image in the same place
                    1 => hist = hist.replacen("again", &format!("and {PH} again"), 1),
                    _ => {}
                }
            }
        }
        assert_eq!(requests, 2400);
    }

    // ---------------------------------------------------------------- the cuts a finished turn leaves

    #[test]
    fn a_finished_session_has_a_cut_at_each_added_token_of_its_text_and_none_past_the_history() {
        let tok = syn();
        let mut t = SessionTable::new(8, 1 << 20);
        let p = "<｜begin▁of▁sentence｜><｜System｜>You are a careful assistant.<｜User｜>héllo wörld ✓ <think> x</think><｜Assistant｜></think>".repeat(3);
        let gen = reply(&tok, "reply ✓<｜User｜>also");
        let a = serve(&mut t, &tok, &p, &gen, false);
        let s = &t.sessions[&a.session_id];
        let text = format!("{p}reply ✓<｜User｜>also<｜end▁of▁sentence｜>");
        let ids = tok.encode(&text);
        assert_eq!(s.n_hist as usize, ids.len() - 1, "the last id is not in the engine's history");
        let mut want = Vec::new();
        for (a, e) in tok.added_spans(&text) {
            for p in [a, e] {
                if p > 0 {
                    want.push(p);
                }
            }
        }
        want.dedup();
        want.retain(|&p| tok.encode(&text[..p]).len() <= s.n_hist as usize);
        let got: Vec<usize> = s.cuts.iter().map(|c| c.bytes).collect();
        assert_eq!(got, want);
        for c in &s.cuts {
            assert_eq!(c.ids as usize, tok.encode(&text[..c.bytes]).len(), "ids before byte {}", c.bytes);
            assert_eq!(c.hash, digest128(&Sha256::new().chain_update(&text.as_bytes()[..c.bytes])), "SHA-256 of the text up to byte {}", c.bytes);
        }
        assert!(got.len() > 10 && s.cuts.last().unwrap().bytes < text.len(), "the end token's own edge is past the history");
        // the keys are those of the text, level by level
        assert_eq!(s.keys, scan(&text, &Resume::start(), 0, &tok).1);
    }

    /// A request that ended without a DONE changes nothing in its session but that it is no longer in a request; a session that never
    /// finished a turn goes (the caller releases it on the engine).
    #[test]
    fn a_request_that_ends_without_a_done_leaves_its_session_as_it_was() {
        let tok = syn();
        let mut t = SessionTable::new(8, 1 << 20);
        let p1 = turn(SYS, "hello");
        let a = serve(&mut t, &tok, &p1, &reply(&tok, "Hi."), false);
        let before = (t.sessions[&a.session_id].cuts.clone(), t.sessions[&a.session_id].epoch, t.sessions[&a.session_id].n_hist);
        let p2 = turn(&format!("{p1}Hi.<｜end▁of▁sentence｜>"), "next");
        let (b, _) = t.plan(&p2, &tok, 1, false);
        assert_eq!(b.session_id, a.session_id);
        assert!(t.sessions[&a.session_id].busy);
        assert!(!t.give_up(b.session_id), "a session with a finished turn stays");
        let s = &t.sessions[&a.session_id];
        assert!(!s.busy);
        assert_eq!((s.cuts.clone(), s.epoch, s.n_hist), before);
        // a fork whose request never finished has no history: it goes
        let other = turn(SYS, "something else entirely");
        let (f, _) = t.plan(&other, &tok, 1, false);
        assert_eq!(f.base_session_id, a.session_id);
        assert!(t.sessions.contains_key(&f.session_id));
        // it is no candidate while it lives
        let (g, _) = t.plan(&other, &tok, 1, false);
        assert_eq!(g.base_session_id, a.session_id);
        assert!(t.give_up(f.session_id), "the caller releases it on the engine");
        assert!(!t.sessions.contains_key(&f.session_id));
        assert!(t.give_up(g.session_id));
    }

    #[test]
    fn eight_requests_for_one_session_at_once_make_one_in_place_and_seven_forks() {
        let tok = std::sync::Arc::new(syn());
        let table = std::sync::Arc::new(Mutex::new(SessionTable::new(64, 1 << 24)));
        let p1 = turn(SYS, "hello");
        let a = {
            let mut t = table.lock().unwrap();
            serve(&mut t, &tok, &p1, &reply(&tok, "Hi."), false)
        };
        let next = turn(&format!("{p1}Hi.<｜end▁of▁sentence｜>"), "next");
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let (table, tok, next) = (table.clone(), tok.clone(), next.clone());
                std::thread::spawn(move || plan_shared(&table, &next, &tok, 1, false, &[]).0)
            })
            .collect();
        let plans: Vec<Plan> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let in_place = plans.iter().filter(|p| p.session_id == a.session_id && p.base_session_id == 0).count();
        let forks = plans.iter().filter(|p| p.base_session_id == a.session_id && p.session_id != a.session_id).count();
        assert_eq!((in_place, forks), (1, 7), "one request takes the session, the others fork it");
        let ids: std::collections::HashSet<u64> = plans.iter().map(|p| p.session_id).collect();
        assert_eq!(ids.len(), 8, "no two requests share a session");
        assert!(plans.iter().all(|p| p.from.ids > 0 && p.epoch == 5));
    }

    /// The race, made to happen: between a request's choice (and the tokenising of its suffix, outside the lock) and its commit another
    /// request takes the same session. The first is chosen again under the lock and forks; it never goes in place on a busy session.
    #[test]
    fn a_choice_the_table_has_since_changed_is_made_over() {
        let tok = syn();
        let table = Mutex::new(SessionTable::new(64, 1 << 24));
        let p1 = turn(SYS, "hello");
        let a = serve(&mut table.lock().unwrap(), &tok, &p1, &reply(&tok, "Hi."), false);
        let next = turn(&format!("{p1}Hi.<｜end▁of▁sentence｜>"), "next");
        let mut rival: Option<Plan> = None;
        let mut calls = 0;
        let (first, _) = plan_shared_with(&table, &next, &tok, 1, false, &[], &mut || {
            calls += 1;
            if calls == 1 {
                rival = Some(plan_shared(&table, &next, &tok, 1, false, &[]).0);
            }
        });
        let rival = rival.unwrap();
        assert_eq!((rival.session_id, rival.base_session_id), (a.session_id, 0), "the rival took the session in place");
        assert_eq!(first.base_session_id, a.session_id, "the first was chosen again and forks it");
        assert_ne!(first.session_id, a.session_id);
        assert!(calls >= 2, "the commit was tried again: {calls}");
    }

    // ---------------------------------------------------------------- the table file

    /// A conversation as a table of text and ids holds it, and the file the `DSFT0003` writer makes of it.
    struct V3 {
        id: u64,
        text: String,
        ids: Vec<u32>,
        prompt_len: usize,
        ablit: bool,
        factor: u64,
        images: Vec<ImageKey>,
    }

    fn v3_table(sessions: &[V3]) -> Vec<u8> {
        let now = unix_now();
        let mut body = Vec::new();
        for s in sessions {
            let flags = (if s.ablit { TABLE_FLAG_ABLIT } else { 0 }) | (if s.images.is_empty() { 0 } else { TABLE_FLAG_IMAGES });
            for v in [s.id, 3, now, s.prompt_len as u64, s.text.len() as u64, flags, s.factor] {
                body.extend(v.to_le_bytes());
            }
            body.extend(s.text.as_bytes());
            body.extend((s.ids.len() as u64).to_le_bytes());
            for &x in &s.ids {
                body.extend(x.to_le_bytes());
            }
            if !s.images.is_empty() {
                body.extend((s.images.len() as u64).to_le_bytes());
                for k in &s.images {
                    for v in [k.at as u64, k.rows as u64, k.hash] {
                        body.extend(v.to_le_bytes());
                    }
                }
            }
        }
        let mut f = TABLE_MAGIC_V3.to_vec();
        for v in [now, 100_000, 9, sessions.len() as u64, body.len() as u64, fnv(&body)] {
            f.extend(v.to_le_bytes());
        }
        f.extend(body);
        f
    }

    /// A `DSFT0003` file, made by its writer, is read once and becomes cuts: no conversation is lost, each continues at the
    /// cut the reference plan reuses, with no epoch (its first request carries the whole prompt and the DONE teaches the epoch).
    #[test]
    fn a_file_of_the_last_layout_is_read_once_and_becomes_cuts() {
        let tok = syn();
        let dir = table_dir("v3");
        let path = dir.join("front.table");
        let mut seed = 5u64;
        let mut reft = RefTable::default();
        let mut convs: Vec<(String, u64)> = Vec::new();
        for i in 0..40u64 {
            let sys = format!("<｜begin▁of▁sentence｜><｜System｜>{}", words(&mut seed, 30));
            let mut text = sys;
            for _ in 0..1 + lcg(&mut seed) % 4 {
                text = turn(&text, &words(&mut seed, 6));
                let rp = reft.plan(&text, &tok, i % 5 == 0, &[]);
                let gen = reply(&tok, &words(&mut seed, 8));
                reft.finish(&rp, &text, &gen, &tok, i % 5 == 0, &[]);
                text = format!("{text}{}", tok.decode(&gen));
            }
            convs.push((text, i));
        }
        let sessions: Vec<V3> = reft
            .s
            .iter()
            .map(|(id, s)| V3 { id: *id, text: s.text.clone(), ids: s.ids.clone(), prompt_len: s.prompt_len, ablit: s.ablit, factor: s.factor as u64, images: s.images.clone() })
            .collect();
        assert!(sessions.len() >= 30, "{} conversations", sessions.len());
        write_table(&path, &v3_table(&sessions)).unwrap();
        let mut t = SessionTable::new(1000, 1 << 40);
        assert_eq!(t.load(&path, &tok, 3600), Ok(sessions.len()), "no conversation lost");
        for s in &sessions {
            assert_eq!(t.sessions[&s.id].n_hist as usize, s.ids.len());
            assert_eq!(t.sessions[&s.id].epoch, 0);
        }
        // each conversation's next turn goes where the reference plan sends it, reusing what the reference plan reuses (less an end id's position
        // at the very end of an output, which is not in the engine's history)
        for (k, (text, i)) in convs.iter().enumerate() {
            let next = turn(&text.to_string(), &format!("next {k}"));
            let rp = reft.plan(&next, &tok, i % 5 == 0, &[]);
            let (np, _) = t.plan(&next, &tok, 1, i % 5 == 0);
            assert_eq!(np.session_id, rp.session, "conversation {k}");
            assert_eq!(np.base_session_id, rp.base);
            assert_eq!(np.total(), rp.ids.len(), "conversation {k}: the same number of ids as the reference plan");
            assert!(np.from.ids as usize <= rp.reused && rp.reused - np.from.ids as usize <= 1, "conversation {k}: {} of {}", np.from.ids, rp.reused);
            assert_eq!(np.epoch, 0, "a restored session names no epoch");
            assert!(!np.prefix_mode(5, false), "so the request carries its whole prompt");
            t.give_up(np.session_id);
        }
        // saved as DSFT0004 and read back: the same sessions, cuts and epochs
        let mut u = SessionTable::new(1000, 1 << 40);
        let mut t2 = SessionTable::new(1000, 1 << 40);
        t2.load(&path, &tok, 3600).unwrap();
        t2.sessions.get_mut(&sessions[0].id).unwrap().epoch = 99;
        assert_eq!(t2.save(&path, 3600), Ok(sessions.len()));
        assert_eq!(&std::fs::read(&path).unwrap()[..8], b"DSFT0004");
        assert_eq!(u.load(&path, &tok, 3600), Ok(sessions.len()));
        for s in &sessions {
            let (a, b) = (&t2.sessions[&s.id], &u.sessions[&s.id]);
            assert_eq!((&a.cuts, &a.keys, a.epoch, a.n_hist, a.prompt_len, a.factor, a.ablit), (&b.cuts, &b.keys, b.epoch, b.n_hist, b.prompt_len, b.factor, b.ablit));
        }
        assert_eq!(u.sessions[&sessions[0].id].epoch, 99);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Two texts that differ in one byte share no edge's hash and no index level's key from that byte on, and share all of them before it:
    /// 100,000 random pairs (a sanity check of the wiring, not a proof of the hash).
    #[test]
    fn two_texts_that_differ_in_one_byte_never_share_a_key_after_it() {
        let tok = syn();
        let mut rng = 99u64;
        let mut levels_checked = [0usize; 6];
        for pair in 0..100_000u32 {
            let mut text = String::from("<｜begin▁of▁sentence｜>");
            let mut words_at: Vec<usize> = Vec::new();
            let n = 150 + lcg(&mut rng) as usize % 250;
            for _ in 0..n {
                if lcg(&mut rng).is_multiple_of(3) {
                    text.push_str("<｜User｜>");
                }
                words_at.push(text.len());
                text.push_str(words_r(&mut rng, 3, 6).as_str());
                text.push(' ');
            }
            text.push_str("<｜Assistant｜></think>");
            let w = words_at[lcg(&mut rng) as usize % words_at.len()];
            let end = w + text[w..].find('<').unwrap_or(text.len() - w);
            // an ASCII letter of a word: the text stays UTF-8 and no added token changes
            let Some(k) = text.as_bytes()[w..end].iter().position(|b| b.is_ascii_lowercase()) else { continue };
            let q = w + k;
            let mut other = text.clone().into_bytes();
            other[q] = if other[q] == b'x' { b'y' } else { b'x' };
            let other = String::from_utf8(other).unwrap();
            let (ea, ka) = scan(&text, &Resume::start(), 0, &tok);
            let (eb, kb) = scan(&other, &Resume::start(), 0, &tok);
            assert_eq!((ea.len(), ka.len()), (eb.len(), kb.len()), "pair {pair}");
            for (a, b) in ea.iter().zip(&eb) {
                assert_eq!(a.bytes, b.bytes);
                assert_eq!(a.hash == b.hash, a.bytes <= q, "pair {pair}: edge at {} against a change at {q}", a.bytes);
            }
            for (l, (a, b)) in ka.iter().zip(&kb).enumerate() {
                assert_eq!(a == b, level_len(l) <= q, "pair {pair}: level {l} against a change at {q}");
                levels_checked[l.min(5)] += 1;
            }
        }
        assert!(levels_checked[..4].iter().all(|&n| n > 10_000), "{levels_checked:?}: every level must be reached");
    }

    /// The record carries the whole 128-bit hash: both halves survive a save and a load, and a hash cut to 64 bits would not match.
    #[test]
    fn the_wide_hash_survives_a_saved_table() {
        let tok = syn();
        let dir = table_dir("wide");
        let path = dir.join("front.table");
        let mut t = SessionTable::new(8, 1 << 20);
        put_fresh(&mut t, 3, &turn(SYS, "hello"), false);
        {
            let s = t.sessions.get_mut(&3).unwrap();
            for (k, c) in s.cuts.iter_mut().enumerate() {
                c.hash = ((0xABCD_0000_0000_0000u128 + k as u128) << 64) | (0x1234_5678_9ABC_DEF0u128 + k as u128);
            }
            s.keys = vec![u128::MAX - 1, 5u128 << 100];
        }
        assert_eq!(t.save(&path, 3600), Ok(1));
        let mut u = SessionTable::new(8, 1 << 20);
        assert_eq!(u.load(&path, &tok, 3600), Ok(1));
        assert_eq!(u.sessions[&3].cuts, t.sessions[&3].cuts);
        assert_eq!(u.sessions[&3].keys, t.sessions[&3].keys);
        assert!(u.sessions[&3].cuts.iter().all(|c| c.hash >> 64 != 0));
        assert_eq!(std::mem::size_of::<Cut>(), 32);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Hashing speed on a 16 MB rendered prompt (median of 5): run with `-- --ignored --nocapture` in a release build.
    #[test]
    #[ignore]
    fn hash_speed() {
        let tok = syn();
        let mut rng = 5u64;
        let mut text = String::from("<｜begin▁of▁sentence｜>");
        while text.len() < 16 << 20 {
            text.push_str("<｜User｜>");
            text.push_str(&words(&mut rng, 40));
            text.push_str("<｜Assistant｜></think>");
            text.push_str(&words(&mut rng, 80));
            text.push_str("<｜end▁of▁sentence｜>");
        }
        let mut raw = Vec::new();
        let mut scans = Vec::new();
        let mut spans = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            std::hint::black_box(tok.added_spans(&text).len());
            spans.push(t.elapsed().as_secs_f64());
            let t = Instant::now();
            let mut h = Sha256::new();
            h.update(text.as_bytes());
            std::hint::black_box(digest128(&h));
            raw.push(t.elapsed().as_secs_f64());
            let t = Instant::now();
            let p = Prompt::new(&text, &tok, false, &[]);
            std::hint::black_box(p.edges.len());
            scans.push(t.elapsed().as_secs_f64());
        }
        raw.sort_by(|a, b| a.partial_cmp(b).unwrap());
        scans.sort_by(|a, b| a.partial_cmp(b).unwrap());
        spans.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mb = text.len() as f64 / 1e6;
        eprintln!(
            "hash speed: {mb:.1} MB; SHA-256 alone {:.0} MB/s; the added-token scan alone {:.0} MB/s; Prompt::new (scan, hash at {} edges with their states) {:.0} MB/s",
            mb / raw[2],
            mb / spans[2],
            Prompt::new(&text, &tok, false, &[]).edges.len(),
            mb / scans[2]
        );
    }

    #[test]
    fn a_damaged_cut_table_is_refused_whole() {
        let tok = syn();
        let dir = table_dir("badcuts");
        let path = dir.join("front.table");
        let mut t = SessionTable::new(8, 1 << 20);
        put_fresh(&mut t, 3, &turn(SYS, "hello"), false);
        // cuts out of order: the record is well formed (its hash is right) and is not taken
        t.sessions.get_mut(&3).unwrap().cuts.reverse();
        assert_eq!(t.save(&path, 3600), Ok(1));
        assert_eq!(SessionTable::new(8, 1 << 20).load(&path, &tok, 3600), Ok(0), "a session whose cuts are out of order is left out");
        // a count that cannot fit in the file
        let mut body = Vec::new();
        write_session(&mut body, 3, &t.sessions[&3]);
        let k = 8 * 9;
        body[k..k + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        let mut f = TABLE_MAGIC.to_vec();
        for v in [unix_now(), 9, 9, 1, body.len() as u64, fnv(&body)] {
            f.extend(v.to_le_bytes());
        }
        f.extend(&body);
        write_table(&path, &f).unwrap();
        assert!(SessionTable::new(8, 1 << 20).load(&path, &tok, 3600).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // ---------------------------------------------------------------- memory

    /// Heap use of the calling thread only, for the tests that bound memory.
    mod probe {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;

        thread_local! {
            static ON: Cell<bool> = const { Cell::new(false) };
            static CUR: Cell<isize> = const { Cell::new(0) };
            static PEAK: Cell<isize> = const { Cell::new(0) };
        }

        pub struct Counting;

        fn note(d: isize) {
            let _ = ON.try_with(|on| {
                if on.get() {
                    CUR.with(|c| {
                        c.set(c.get() + d);
                        PEAK.with(|p| p.set(p.get().max(c.get())));
                    });
                }
            });
        }

        unsafe impl GlobalAlloc for Counting {
            unsafe fn alloc(&self, l: Layout) -> *mut u8 {
                note(l.size() as isize);
                System.alloc(l)
            }
            unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
                note(l.size() as isize);
                System.alloc_zeroed(l)
            }
            unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
                note(-(l.size() as isize));
                System.dealloc(p, l)
            }
            unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
                note(n as isize - l.size() as isize);
                System.realloc(p, l, n)
            }
        }

        /// `f`'s result, the bytes it left allocated and the most it had allocated at once (this thread).
        pub fn measure<R>(f: impl FnOnce() -> R) -> (R, usize, usize) {
            CUR.with(|c| c.set(0));
            PEAK.with(|p| p.set(0));
            ON.with(|o| o.set(true));
            let r = f();
            ON.with(|o| o.set(false));
            (r, CUR.with(|c| c.get()).max(0) as usize, PEAK.with(|p| p.get()).max(0) as usize)
        }
    }

    #[global_allocator]
    static COUNTING: probe::Counting = probe::Counting;

    /// Token lengths as a real vocabulary has them: mostly short, some of an added token's size.
    fn len_of_id(id: u32) -> usize {
        [1, 1, 2, 3, 4, 5, 9, 13, 23, 29][id as usize % 10]
    }

    #[test]
    fn a_token_end_found_by_marks_is_the_one_a_search_of_every_end_finds() {
        let mut st = 7u64;
        for n in [0usize, 1, 2, 31, 32, 33, 63, 64, 65, 95, 96, 97, 300, 1000] {
            for _ in 0..6 {
                let ids: Vec<u32> = (0..n).map(|_| lcg(&mut st) as u32 % 1000).collect();
                let (marks, total) = mark_ids(&ids, len_of_id);
                assert_eq!(marks.len(), n / 32);
                let mut ends = Vec::new();
                let mut at = 0;
                for &id in &ids {
                    at += len_of_id(id);
                    ends.push(at);
                }
                assert_eq!(total, at);
                for p in 0..=total + 3 {
                    let want = ends.binary_search(&p).ok().map(|k| k + 1);
                    assert_eq!(token_end_in(&marks, &ids, total, p, len_of_id), want, "n {n} p {p}");
                }
            }
        }
    }

    #[test]
    fn a_text_over_four_gigabytes_has_no_marks_and_never_matches_an_edge() {
        let ids = vec![0u32; 64];
        let (marks, total) = mark_ids(&ids, |_| 1 << 26);
        assert_eq!(total, 1 << 32);
        assert!(marks.is_empty());
        assert_eq!(token_end_in(&marks, &ids, total, 1 << 26, |_| 1 << 26), None);
        let (marks, total) = mark_ids(&ids, |_| 1 << 25);
        assert_eq!((marks.len(), marks[1]), (2, 1 << 31));
        assert_eq!(token_end_in(&marks, &ids, total, 33 << 25, |_| 1 << 25), Some(33));
    }

    /// A finished session of about `cuts` cuts (a prompt of `cuts / 2` user turns, no output), through the table's own steps.
    fn add(t: &mut SessionTable, tok: &Tokenizer, i: usize, cuts: usize) -> u64 {
        let text = format!("<｜begin▁of▁sentence｜>session number {i:05} {}", "<｜User｜>q".repeat(cuts / 2));
        serve(t, tok, &text, &[], false).session_id
    }

    #[test]
    fn the_byte_limit_drops_the_oldest_idle_sessions_and_spares_a_busy_one() {
        let tok = syn();
        let one = {
            let mut s = SessionTable::new(1000, 1 << 40);
            let id = add(&mut s, &tok, 0, 2000);
            s.sessions[&id].bytes()
        };
        let limit = 8 * one + one / 2;
        let mut t = SessionTable::new(1000, 1 << 40);
        t.set_byte_limits(limit, 0);
        let busy_text = format!("<｜begin▁of▁sentence｜>the busy conversation {}", "z".repeat(10_000));
        let (busy, _) = t.plan(&busy_text, &tok, 1, false);
        let mut made = Vec::new();
        for i in 0..12 {
            // a session is taken in by the plan that starts it, which is where the limit is kept; its record grows at its finish
            let text = format!("<｜begin▁of▁sentence｜>session number {i:05} {}", "<｜User｜>q".repeat(1000));
            let (p, _) = t.plan(&text, &tok, 1, false);
            assert!(t.bytes() <= limit, "after session {i}: {} bytes", t.bytes());
            finish_plan(&mut t, &tok, &p, &text, &[], &[]);
            made.push(p.session_id);
        }
        assert!(t.len() < 13, "only the byte limit can have dropped any: sessions, ids and positions are far from theirs");
        assert!(t.sessions.contains_key(&busy.session_id), "a session in a request stays");
        assert!(!t.sessions.contains_key(&made[0]) && t.sessions.contains_key(made.last().unwrap()), "the oldest goes first");
    }

    #[test]
    fn a_session_counts_its_capacity_not_its_length() {
        let mut t = SessionTable::new(8, 1 << 20);
        put(&mut t, 1, "x", false);
        let lean = t.sessions[&1].bytes();
        let s = t.sessions.get_mut(&1).unwrap();
        s.cuts = Vec::with_capacity(100_000);
        s.keys = Vec::with_capacity(5_000);
        assert!(t.sessions[&1].bytes() >= lean + std::mem::size_of::<Cut>() * 100_000 + 8 * 5_000 - 1);
    }

    #[test]
    fn the_counted_bytes_follow_what_the_table_allocates() {
        let tok = syn();
        let ((t, ids), held, _) = probe::measure(|| {
            let mut t = SessionTable::new(1000, 1 << 40);
            let ids: Vec<u64> = (0..24).map(|i| add(&mut t, &tok, i, 4000 + 90 * i)).collect();
            (t, ids)
        });
        assert_eq!(ids.len(), 24);
        let counted = t.bytes();
        let diff = counted.abs_diff(held) as f64 / held as f64;
        assert!(diff < 0.10, "counted {counted}, allocated {held}: {:.1} % apart", diff * 100.0);
        assert!(counted >= held * 9 / 10, "never well under what is held");
    }

    #[test]
    fn a_finished_session_gives_back_the_slack_its_buffers_grew() {
        let tok = syn();
        let mut t = SessionTable::new(8, 1 << 24);
        let prompt = format!("<｜begin▁of▁sentence｜>{}", "<｜User｜>p".repeat(20_000));
        let a = serve(&mut t, &tok, &prompt, &tok.encode(&"<｜User｜>r".repeat(5_000)), false);
        let s = &t.sessions[&a.session_id];
        assert!(s.cuts.len() > 40_000);
        assert!(s.cuts.capacity() <= s.cuts.len() + s.cuts.len() / 4 + 4096, "cuts {} of {}", s.cuts.len(), s.cuts.capacity());
        assert!(s.keys.capacity() <= s.keys.len() + 4096);
    }

    #[test]
    fn over_the_process_limit_idle_sessions_go_oldest_first_until_the_excess_is_gone() {
        let tok = syn();
        let mut t = SessionTable::new(1000, 1 << 40);
        let ids: Vec<u64> = (0..8).map(|i| add(&mut t, &tok, i, 1000)).collect();
        let (busy, _) = t.plan(&format!("<｜begin▁of▁sentence｜>busy and old {}", "<｜User｜>b".repeat(500)), &tok, 1, false);
        t.sessions.get_mut(&busy.session_id).unwrap().last_used = 0;
        let one = t.sessions[&ids[0]].bytes();
        let mut g = Governor::default();
        assert_eq!(g.step(&mut t, 1000, 1000), Governed::default(), "at the limit nothing goes");
        let r = g.step(&mut t, 1000 + 2 * one + 1, 1000);
        assert!(r.over);
        assert_eq!(r.released, ids[..3].to_vec(), "three sessions cover two and a bit of them, the oldest first");
        assert!(t.sessions.contains_key(&busy.session_id), "a session in a request stays however old");
    }

    /// A table of `n` idle sessions of one size and the size.
    fn idle_table(n: usize) -> (SessionTable, Vec<u64>, usize) {
        let tok = syn();
        let mut t = SessionTable::new(1000, 1 << 40);
        let ids: Vec<u64> = (0..n).map(|i| add(&mut t, &tok, i, 500)).collect();
        let one = t.sessions[&ids[0]].bytes();
        (t, ids, one)
    }

    #[test]
    fn memory_that_does_not_fall_after_a_round_stops_the_dropping_until_it_has_been_under_the_limit() {
        let (mut t, ids, one) = idle_table(20);
        let limit = 1_000_000;
        let mut g = Governor::default();
        let first = g.step(&mut t, limit + 2 * one, limit);
        assert_eq!(first.released.len(), 2);
        let left = t.len();
        // the reading has not moved: one more look says so, once, and drops nothing
        let second = g.step(&mut t, limit + 2 * one, limit);
        assert!(second.over && second.released.is_empty());
        let line = second.held_elsewhere.expect("the log line");
        assert!(line.contains("RSS") && line.contains("counts") && line.contains("limit"), "{line}");
        for _ in 0..5 {
            let r = g.step(&mut t, limit + 2 * one, limit);
            assert!(r.over && r.released.is_empty() && r.held_elsewhere.is_none(), "quiet and still");
        }
        assert_eq!(t.len(), left);
        // under the limit once: dropping works again
        assert_eq!(g.step(&mut t, limit - 1, limit), Governed::default());
        assert_eq!(g.step(&mut t, limit + 2 * one, limit).released.len(), 2);
        assert!(ids.len() > t.len());
    }

    #[test]
    fn memory_that_falls_by_a_quarter_of_what_was_freed_lets_the_next_round_go_on() {
        let (mut t, _, one) = idle_table(20);
        let limit = 1_000_000;
        let mut g = Governor::default();
        assert_eq!(g.step(&mut t, limit + 4 * one, limit).released.len(), 4);
        let r = g.step(&mut t, limit + 4 * one - one, limit); // fell by exactly a quarter of the 4 freed
        assert!(r.held_elsewhere.is_none() && !r.released.is_empty());
    }

    #[test]
    fn a_stretch_over_the_limit_never_takes_the_table_below_half_of_what_it_held_at_its_start() {
        let (mut t, _, one) = idle_table(20);
        let start = t.bytes();
        let mut g = Governor::default();
        let mut rss = 1_000_000 + 1000 * one;
        for _ in 0..40 {
            g.step(&mut t, rss, 1_000_000);
            rss -= 5 * one; // falling fast enough: never stalled
        }
        assert!(t.bytes() >= start / 2 && t.bytes() < start / 2 + one, "{} of {start}", t.bytes());
    }

    #[test]
    fn the_status_line_names_what_the_table_and_the_process_hold() {
        assert_eq!(
            front_status(3 << 30, 120, 200_000_000, Some(2 << 30), Some(3 << 30)),
            "session table: 3072 MiB counted, 120 sessions, 200000000 tokens, 16.1 B/token; RSS 2048 MiB, high-water 3072 MiB"
        );
        assert!(front_status(0, 0, 0, None, None).contains("0.0 B/token; RSS ? MiB"));
        assert_eq!(parse_status_kb("VmHWM:\t 5 kB\n", "VmHWM:"), Some(5120));
    }

    #[test]
    fn image_rows_in_flight_take_room_from_the_table_up_to_their_limit() {
        let tok = syn();
        let mut t = SessionTable::new(1000, 1 << 40);
        t.set_byte_limits(150_000, 60_000);
        for i in 0..9 {
            add(&mut t, &tok, i, 450);
        }
        let held = t.len();
        assert!(held >= 6, "{held} sessions of about 20 KB under 150 KB");
        t.set_image_bytes(10_000_000);
        assert_eq!(t.bytes() - t.sessions.values().map(|s| s.bytes()).sum::<usize>(), 60_000, "counted up to the limit");
        let text = format!("<｜begin▁of▁sentence｜>session number 99999 {}", "k".repeat(4_000));
        t.plan(&text, &tok, 1, false);
        assert!(t.len() < held && t.bytes() <= 150_000, "the rows' share pushed sessions out: {} of {held}", t.len());
    }

    #[test]
    fn the_process_memory_is_read_from_the_kernels_line() {
        assert_eq!(parse_vm_rss("Name:\tx\nVmRSS:\t  1234 kB\nThreads:\t3\n"), Some(1234 * 1024));
        assert_eq!(parse_vm_rss("Name:\tx\n"), None);
        assert!(vm_rss_bytes().unwrap() > 1 << 20);
    }

    /// `n` sessions of `cuts` cuts each (and the shapes a real table has: both weights paths, the three factors, an image now and then),
    /// straight into the table.
    fn bulk(t: &mut SessionTable, n: usize, cuts: usize) {
        for i in 0..n {
            let id = 100 + i as u64;
            let list: Vec<Cut> = (1..=cuts).map(|k| Cut { bytes: 10 * k, ids: 9 * k as u32, hash: (k as u128).wrapping_mul(0x9e3779b97f4a7c15f39cc0605cedc835) ^ i as u128 }).collect();
            t.sessions.insert(
                id,
                Session {
                    cuts: list,
                    keys: vec![i as u128 * 3, i as u128 * 3 + 1, i as u128 * 3 + 2],
                    prompt_len: 10 * cuts,
                    t_used: unix_now() - (i as u64 % 5),
                    images: if i % 4 == 1 { vec![ImageKey { at: 30, rows: 5, hash: 77 + i as u64 }] } else { Vec::new() },
                    epoch: 7 + i as u64,
                    n_hist: 9 * cuts as u32,
                    ..Session::empty(i % 3 == 0, [16, 32, 64][i % 3], i as u64)
                },
            );
            t.enter(id);
        }
    }

    #[test]
    fn the_saved_file_is_the_one_the_whole_table_snapshot_makes() {
        let tok = syn();
        let dir = table_dir("bytes");
        let path = dir.join("front.table");
        let table = std::sync::Mutex::new(SessionTable::new(1000, 1 << 40));
        {
            let mut t = table.lock().unwrap();
            bulk(&mut t, 23, 20_000);
            put(&mut t, 5, "", false); // no finished turn: left out
            put_fresh(&mut t, 6, "<｜begin▁of▁sentence｜>an old one<｜User｜>x", false);
            t.sessions.get_mut(&6).unwrap().t_used = 1;
        }
        let (n, _) = save_table(&table, &path, 3600).unwrap();
        assert_eq!(n, 23);
        let file = std::fs::read(&path).unwrap();
        let want = table.lock().unwrap().snapshot(3600).0;
        assert_eq!(file.len(), want.len());
        assert!(file[..8] == want[..8] && file[16..] == want[16..], "everything but the time written is the same");
        assert!(!dir.join("front.table.new").exists());
        assert_eq!(std::fs::metadata(&path).unwrap().mode() & 0o777, 0o600);
        // the reader takes it
        let mut u = SessionTable::new(1000, 1 << 40);
        assert_eq!(u.load(&path, &tok, 3600), Ok(23));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_save_holds_a_few_megabytes_and_never_the_lock_across_a_write() {
        let dir = table_dir("peak");
        let path = dir.join("front.table");
        let table = std::sync::Mutex::new(SessionTable::new(1000, 1 << 40));
        bulk(&mut table.lock().unwrap(), 50, 100_000);
        let mut free_at_writes = Vec::new();
        let (r, _, peak) = probe::measure(|| save_table_with(&table, &path, 3600, &mut || free_at_writes.push(table.try_lock().is_ok())));
        assert_eq!(r.unwrap().0, 50);
        assert!(std::fs::metadata(&path).unwrap().len() > 40 << 20, "a file far over the bound");
        assert!(peak < 16 << 20, "peak {} MiB", peak >> 20);
        assert!(free_at_writes.len() >= 5, "written in batches: {}", free_at_writes.len());
        assert!(free_at_writes.iter().all(|&free| free), "the table's lock was held across a write");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_session_changed_between_batches_is_saved_as_it_is_then_and_a_dropped_one_is_left_out() {
        let tok = syn();
        let dir = table_dir("between");
        let path = dir.join("front.table");
        let table = std::sync::Mutex::new(SessionTable::new(1000, 1 << 40));
        bulk(&mut table.lock().unwrap(), 20, 200_000);
        let mut calls = 0;
        let (n, at) = save_table_with(&table, &path, 3600, &mut || {
            calls += 1;
            if calls == 2 {
                let mut t = table.lock().unwrap();
                let victim = *t.sessions.keys().last().unwrap(); // saved in the last batch, which is not yet taken
                t.remove(victim);
            }
        })
        .unwrap();
        assert_eq!(n, 19);
        assert_ne!(table.lock().unwrap().changes(), at, "the change is seen by the next periodic save");
        let mut u = SessionTable::new(1000, 1 << 40);
        assert_eq!(u.load(&path, &tok, 3600), Ok(n), "a whole file that loads");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The bytes of text a real vocabulary spells in a token (an estimate). The synthetic tokenizer spells one, so a synthetic table has four
    /// times the cuts a token that real text has, which only makes its counted size the more conservative.
    const BYTES_PER_TOKEN: usize = 4;

    /// A synthetic load of 70.7M tokens, its conversations' shape taken from real turns through the table's own steps (a user message of
    /// about 500 bytes, an output of about 900, a token a byte here), counts under 32 MiB. The control is the table of text and ids on the
    /// same conversations, counted by allocating its records (the text, the ids, the marks) at the same sizes: over 500 MiB.
    #[test]
    fn a_load_of_70_million_tokens_counts_under_32_mib_where_the_text_and_ids_counted_over_500() {
        let tok = syn();
        let mut t = SessionTable::new(1 << 20, 1 << 40);
        let mut seed = 11u64;
        let mut old_counted = 0usize;
        for c in 0..14usize {
            let mut text = format!("<｜begin▁of▁sentence｜><｜System｜>{}", words(&mut seed, 80));
            let mut last = 0;
            for _ in 0..120 {
                text = turn(&text, &words(&mut seed, 90));
                let gen = reply(&tok, &words(&mut seed, 160));
                last = serve(&mut t, &tok, &text, &gen, c % 4 == 0).session_id;
                text = format!("{text}{}", tok.decode(&gen));
            }
            // the record of text and ids for this conversation: its text, its ids and the marks of its ids, counted by allocating them
            let n_ids = tok.encode(&text).len();
            let (_keep, held, _) = probe::measure(|| (String::with_capacity(BYTES_PER_TOKEN * n_ids), Vec::<u32>::with_capacity(n_ids), Vec::<u32>::with_capacity(n_ids / MARK_EVERY)));
            old_counted += held + (64 + 16 * 5) + INDEX_ENTRY * t.sessions[&last].keys.len();
        }
        let tokens: usize = t.sessions.values().map(|s| s.n_hist as usize).sum();
        let (new_counted, cuts): (usize, usize) = (t.bytes(), t.sessions.values().map(|s| s.cuts.len()).sum());
        let (new_per, old_per) = (new_counted as f64 / tokens as f64, old_counted as f64 / tokens as f64);
        // the same conversations, repeated until they hold the measured load's tokens
        let target = 70_700_000usize;
        let rounds = target / tokens + 1;
        let mut big = SessionTable::new(1 << 20, 1 << 40);
        let base: Vec<Session> = t.sessions.values().cloned().collect();
        let (_, held_big, _) = probe::measure(|| {
            for r in 0..rounds {
                for (i, s) in base.iter().enumerate() {
                    let id = 1 + (r * base.len() + i) as u64;
                    big.sessions.insert(id, s.clone());
                    big.enter(id);
                }
            }
        });
        let counted = big.bytes();
        let big_tokens: usize = big.sessions.values().map(|s| s.n_hist as usize).sum();
        let old_at_target = (old_per * big_tokens as f64) as usize;
        eprintln!(
            "synthetic load: {big_tokens} tokens in {} sessions, {cuts} cuts per {tokens} tokens; counted {:.1} MiB ({new_per:.3} B/token, allocated {:.1} MiB); text-and-ids records on the same conversations {:.0} MiB ({old_per:.2} B/token)",
            big.len(),
            counted as f64 / 1048576.0,
            held_big as f64 / 1048576.0,
            old_at_target as f64 / 1048576.0
        );
        assert!(big_tokens >= target);
        assert!(counted < 32 << 20, "{} MiB counted for {big_tokens} tokens", counted >> 20);
        assert!(counted.abs_diff(held_big) * 10 < held_big, "counted {counted}, allocated {held_big}");
        assert!(old_at_target > 500 << 20, "text-and-ids records count {} MiB for the same tokens", old_at_target >> 20);
    }
}
