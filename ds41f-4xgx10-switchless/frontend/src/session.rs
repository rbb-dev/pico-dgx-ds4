//! Session matching for exact prefix reuse (docs/protocol.md, "Sessions and exact prefix reuse").
//!
//! Clients do not name their conversation, so a request is matched by text: the session whose text
//! (prompt plus generated output) shares the longest prefix with the new prompt. The shared part is
//! sent as that session's own token ids, which is what is in the engine's KV state; the rest is
//! tokenised on its own from a cut next to an added token, where tokenisation restarts anyway.
//!
//! A session belongs to the weights path it was created on (`ablit`, protocol flag ABLIT): a request is
//! matched only against sessions of its own path, so a continuation or a fork never crosses the two.
//!
//! An image is a run of placeholder tokens (docs/vision.md): the prompt's text carries each image as its
//! placeholder written once per row, so the text and its ids still spell each other, and a session records
//! each image's place and content hash (`ImageKey`). The engine matches a session's history to a prompt by
//! token id alone, and every image row has the same id, so ids cannot tell one image from another: a
//! session is a candidate only when the images inside the text it shares with the prompt are the same
//! images (same place, rows and content hash) on both sides. An image that differs inside the shared text
//! rules the session out; it is never named as a base, so the engine cannot carry its rows over.
//!
//! The table outlives a stop (`save`, `load`; docs/engine.md, "Kept sessions"): the engine keeps its
//! sessions across a planned stop, and the table is what finds them again. It holds the conversations'
//! text, so its file is ours alone: mode 0600 in a directory of mode 0700, written whole or not at all.

use crate::tokenizer::Tokenizer;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Bytes of shared text below which a match is not worth a session (a BOS and a few tokens).
const MIN_SHARED: usize = 64;
/// A request continues a session in place when the shared text reaches this close to the end of
/// the session's prompt; the encoding may rewrite the tail of the prompt (`<think>` → `</think>`).
const PROMPT_SLACK: usize = 16;

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

struct Session {
    text: String,
    ids: Vec<u32>,
    /// ends[i] = byte length of the text of ids[..=i]
    ends: Vec<usize>,
    prompt_len: usize,
    busy: bool,
    last_used: u64,
    /// Unix time of its last request: a saved table leaves out what is older than its limit.
    t_used: u64,
    /// The ablated weights path: its state was computed with the ablation direction applied.
    ablit: bool,
    /// The images of its prompt, in order.
    images: Vec<ImageKey>,
    /// Its text's prefix keys (`prefix_keys`), as the index holds them.
    keys: Vec<u64>,
}

pub struct Plan {
    pub session_id: u64,
    /// The session a new session may fork from (0 = none).
    pub base_session_id: u64,
    /// Leading ids taken from a session (the rest is tokenised).
    pub reused: usize,
    pub ids: Vec<u32>,
    /// For the log, counts only: the session whose text shares the most with the prompt (0: none
    /// shares any), the bytes it shares, and the ids its history has in common with `ids`.
    pub matched: u64,
    pub shared: usize,
    pub common: usize,
}

impl Plan {
    /// Where the request starts, for the log (no text, no ids): a continuation of a session, a
    /// fork of one (the engine starts from it if it holds it), or a new session that names none.
    pub fn describe(&self, prompt_bytes: usize) -> String {
        let of = format!("{} of {prompt_bytes} bytes, {} ids in common", self.shared, self.common);
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
    max_tokens: usize,
}

/// The prefix lengths a text is indexed at: MIN_SHARED bytes, then four times longer each (64, 256, 1 KiB, ... 4 MiB at
/// level 8). A prompt and a session that share at least a level's length are in one bucket at that level.
fn level_len(level: usize) -> usize {
    MIN_SHARED << (2 * level)
}

/// The hash of `text`'s first `level_len(i)` bytes, with the level in it, for every level the text reaches: one pass.
fn prefix_keys(text: &[u8]) -> Vec<u64> {
    let mut out = Vec::new();
    let mut h = 1469598103934665603u64;
    for (i, &b) in text.iter().enumerate() {
        h = (h ^ b as u64).wrapping_mul(1099511628211);
        if i + 1 == level_len(out.len()) {
            out.push(h ^ (out.len() as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15));
        }
    }
    out
}

fn common_prefix(a: &[u8], b: &[u8]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    while i + 64 <= n && a[i..i + 64] == b[i..i + 64] {
        i += 64;
    }
    while i < n && a[i] == b[i] {
        i += 1;
    }
    i
}

fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// The table's layout with each session's flags word. The first layout (`DSFT0001`) has none: its sessions
/// are all of the plain path.
const TABLE_MAGIC: &[u8; 8] = b"DSFT0002";
const TABLE_MAGIC_V1: &[u8; 8] = b"DSFT0001";
const TABLE_FLAG_ABLIT: u64 = 1;
/// The session's ids are followed by its images: u64 count, then (at, rows, hash) as three u64 each.
const TABLE_FLAG_IMAGES: u64 = 2;

fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(1469598103934665603u64, |h, &x| (h ^ x as u64).wrapping_mul(1099511628211))
}

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

/// One periodic save: if `table` changed since the count in `saved`, snapshot it under the lock, write it after the
/// lock is let go and set `saved` to that count. None when nothing changed; a failed write leaves `saved`, so the next
/// call tries again.
pub fn save_if_changed(table: &std::sync::Mutex<SessionTable>, saved: &mut u64, path: &Path, max_age_s: u64) -> Option<Result<usize, String>> {
    let (bytes, n, at) = {
        let t = table.lock().unwrap();
        if t.changes() == *saved {
            return None;
        }
        let (b, n) = t.snapshot(max_age_s);
        (b, n, t.changes())
    };
    Some(write_table(path, &bytes).map(|()| {
        *saved = at;
        n
    }))
}

impl SessionTable {
    /// Holds up to `max_sessions` sessions and `max_tokens` ids in total. Ids start from the clock,
    /// so that a table begun afresh never reuses an earlier one's ids, whose sessions the engine may
    /// still keep.
    pub fn new(max_sessions: usize, max_tokens: usize) -> Self {
        SessionTable { sessions: HashMap::new(), index: HashMap::new(), next_id: (unix_now() << 20).max(1), clock: 0, changes: 0, max_sessions, max_tokens }
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
    /// one with no finished turn is left out), and how many
    /// they are. Memory only: the file is written from it after the table's lock is let go.
    pub fn snapshot(&self, max_age_s: u64) -> (Vec<u8>, usize) {
        let now = unix_now();
        let (mut body, mut n) = (Vec::new(), 0u64);
        for (&id, s) in &self.sessions {
            // a session in a request is saved as its last finished turn left it (its text and ids change only
            // at `finish`); one with no finished turn has nothing to save
            if s.text.is_empty() || now.saturating_sub(s.t_used) > max_age_s {
                continue;
            }
            n += 1;
            let flags = (if s.ablit { TABLE_FLAG_ABLIT } else { 0 }) | (if s.images.is_empty() { 0 } else { TABLE_FLAG_IMAGES });
            for v in [id, s.last_used, s.t_used, s.prompt_len as u64, s.text.len() as u64, flags] {
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
        let mut f = TABLE_MAGIC.to_vec();
        for v in [now, self.next_id, self.clock, n, body.len() as u64, fnv(&body)] {
            f.extend(v.to_le_bytes());
        }
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
        if f.len() < 56 || (&f[..8] != TABLE_MAGIC && &f[..8] != TABLE_MAGIC_V1) {
            return Err("not a saved table".into());
        }
        let with_flags = &f[..8] == TABLE_MAGIC;
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
            let (id, last_used, t_used, prompt_len, text_len) = (r.u64()?, r.u64()?, r.u64()?, r.u64()? as usize, r.u64()? as usize);
            let flags = if with_flags { r.u64()? } else { 0 };
            let text = r.take(text_len)?;
            let n_ids = r.u64()? as usize;
            let raw = r.take(n_ids.checked_mul(4).ok_or("a bad count")?)?;
            let ids: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            let mut images = Vec::new();
            if flags & TABLE_FLAG_IMAGES != 0 {
                for _ in 0..r.u64()? {
                    let (at, rows, hash) = (r.u64()? as usize, r.u64()?, r.u64()?);
                    images.push(ImageKey { at, rows: rows as u32, hash });
                }
            }
            let Ok(text) = std::str::from_utf8(text) else { continue };
            let vocab = tok.vocab_size() as u32;
            if id == 0 || now.saturating_sub(t_used) > max_age_s || prompt_len > text.len() || ids.iter().any(|&x| x >= vocab) {
                continue;
            }
            let mut ends = Vec::with_capacity(ids.len());
            let mut at = 0;
            for &x in &ids {
                at += tok.token_bytes(x).len();
                ends.push(at);
            }
            if at != text.len() || tok.decode_bytes(&ids) != text.as_bytes() {
                continue; // another tokenizer's ids
            }
            if images.iter().any(|k| k.at >= text.len() || k.rows < 3) {
                continue;
            }
            let ablit = flags & TABLE_FLAG_ABLIT != 0;
            got.insert(id, Session { text: text.to_string(), ids, ends, prompt_len, busy: false, last_used, t_used, ablit, images, keys: Vec::new() });
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

    /// Choose the session and the token ids for a prompt of the weights path `ablit`: only sessions of
    /// that path are candidates. The chosen session is busy until `finish` or `abandon`. Returns the
    /// plan and the sessions to RELEASE on the engine.
    pub fn plan(&mut self, text: &str, tok: &Tokenizer, threads: usize, ablit: bool) -> (Plan, Vec<u64>) {
        self.plan_images(text, tok, threads, ablit, &[])
    }

    /// `plan` for a prompt with images: `text` has each image's placeholder written once per row, and
    /// `images` says where each run starts in it, how long it is and what the image is. A session whose
    /// images inside the shared text are not these (see the module comment) is no candidate.
    pub fn plan_images(&mut self, text: &str, tok: &Tokenizer, threads: usize, ablit: bool, images: &[ImageKey]) -> (Plan, Vec<u64>) {
        self.clock += 1;
        self.changes += 1;
        let top = self.best(text, ablit, images);
        let best = top.filter(|&(_, n)| n >= MIN_SHARED);
        let (mut plan, cut) = match best {
            None => (Plan { session_id: self.new_id(), base_session_id: 0, reused: 0, ids: Vec::new(), matched: 0, shared: 0, common: 0 }, 0),
            Some((sid, mut shared)) => {
                while !text.is_char_boundary(shared) {
                    shared -= 1;
                }
                let s = &self.sessions[&sid];
                // The last cut point within the shared text that is a token boundary of the session
                // and an edge of an added token in the new text.
                let mut cut = (0, 0); // (byte position, ids before it)
                for (a, e) in tok.added_spans(text) {
                    if a > shared {
                        break;
                    }
                    for p in [a, e] {
                        if p <= shared && p > cut.0 {
                            if let Ok(k) = s.ends.binary_search(&p) {
                                cut = (p, k + 1);
                            }
                        }
                    }
                }
                let ids = s.ids[..cut.1].to_vec();
                let in_place = !s.busy && shared + PROMPT_SLACK >= s.prompt_len;
                let plan = if in_place {
                    Plan { session_id: sid, base_session_id: 0, reused: cut.1, ids, matched: 0, shared: 0, common: 0 }
                } else {
                    Plan { session_id: self.new_id(), base_session_id: sid, reused: cut.1, ids, matched: 0, shared: 0, common: 0 }
                };
                (plan, cut.0)
            }
        };
        plan.ids.extend(tok.encode_parallel(&text[cut..], threads));
        if let Some((sid, n)) = top {
            plan.matched = sid;
            plan.shared = n;
            let s = &self.sessions[&sid].ids;
            plan.common = s.iter().zip(&plan.ids).take_while(|(a, b)| a == b).count();
        }
        let clock = self.clock;
        let s = self.sessions.entry(plan.session_id).or_insert_with(|| Session {
            text: String::new(),
            ids: Vec::new(),
            ends: Vec::new(),
            prompt_len: 0,
            busy: false,
            last_used: clock,
            t_used: 0,
            ablit,
            images: Vec::new(),
            keys: Vec::new(),
        });
        s.busy = true;
        s.last_used = clock;
        s.t_used = unix_now();
        let released = self.evict(plan.session_id);
        (plan, released)
    }

    /// Record what the session holds after a request: the prompt and the generated ids.
    pub fn finish(&mut self, session_id: u64, prompt_text: &str, prompt_ids: &[u32], generated: &[u32], tok: &Tokenizer) {
        self.finish_images(session_id, prompt_text, prompt_ids, generated, tok, &[]);
    }

    /// `finish` for a prompt with images (`plan_images`' text and keys).
    pub fn finish_images(&mut self, session_id: u64, prompt_text: &str, prompt_ids: &[u32], generated: &[u32], tok: &Tokenizer, images: &[ImageKey]) {
        self.changes += 1;
        let Some(s) = self.sessions.get_mut(&session_id) else { return };
        let gen_bytes = tok.decode_bytes(generated);
        let mut text = prompt_text.to_string();
        let mut ids = prompt_ids.to_vec();
        // Generated text that is not valid UTF-8 cannot come back verbatim; keep the prompt only.
        if let Ok(g) = std::str::from_utf8(&gen_bytes) {
            text.push_str(g);
            ids.extend_from_slice(generated);
        }
        let mut ends = Vec::with_capacity(ids.len());
        let mut at = 0;
        for &id in &ids {
            at += tok.token_bytes(id).len();
            ends.push(at);
        }
        debug_assert_eq!(at, text.len());
        s.text = text;
        s.ids = ids;
        s.ends = ends;
        s.prompt_len = prompt_text.len();
        s.images = images.to_vec();
        s.busy = false;
        self.leave(session_id);
        self.enter(session_id);
    }

    /// The session of the weights path `ablit` whose text shares the longest prefix with `text` (at least MIN_SHARED
    /// bytes), with the images inside it the same, and how many bytes it shares. The deepest level the prompt shares
    /// with some session holds it: every session sharing at least that level's length is in that bucket (a hash that
    /// only collides shares less, and the next level down is tried).
    fn best(&self, text: &str, ablit: bool, images: &[ImageKey]) -> Option<(u64, usize)> {
        let pk = prefix_keys(text.as_bytes());
        let mut top: Option<(u64, usize)> = None;
        for level in (0..pk.len()).rev() {
            for &id in self.index.get(&pk[level]).map_or(&[][..], |v| &v[..]) {
                let s = &self.sessions[&id];
                if s.ablit != ablit {
                    continue;
                }
                let n = common_prefix(text.as_bytes(), s.text.as_bytes());
                if !images_agree(&s.images, images, n) {
                    continue;
                }
                if top.map_or(true, |(_, bn)| n > bn) {
                    top = Some((id, n));
                }
            }
            if top.map_or(false, |(_, n)| n >= level_len(level)) {
                break;
            }
        }
        top
    }

    /// Session `id`'s text into the index at every level it reaches.
    fn enter(&mut self, id: u64) {
        let Some(s) = self.sessions.get_mut(&id) else { return };
        s.keys = prefix_keys(s.text.as_bytes());
        for &k in &s.keys {
            self.index.entry(k).or_default().push(id);
        }
    }

    /// Session `id` out of the index (before its text changes or it goes).
    fn leave(&mut self, id: u64) {
        let Some(s) = self.sessions.get_mut(&id) else { return };
        for k in std::mem::take(&mut s.keys) {
            if let Some(v) = self.index.get_mut(&k) {
                v.retain(|&x| x != id);
                if v.is_empty() {
                    self.index.remove(&k);
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

    /// Drop least recently used idle sessions over the limits; returns their ids.
    fn evict(&mut self, keep: u64) -> Vec<u64> {
        let mut out = Vec::new();
        loop {
            let total: usize = self.sessions.values().map(|s| s.ids.len()).sum();
            if self.sessions.len() <= self.max_sessions && total <= self.max_tokens {
                return out;
            }
            let victim = self.sessions.iter().filter(|(&id, s)| id != keep && !s.busy).min_by_key(|(_, s)| s.last_used).map(|(&id, _)| id);
            match victim {
                Some(id) => {
                    self.remove(id);
                    out.push(id);
                }
                None => return out,
            }
        }
    }

    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    /// The table's limits from what the engine can keep (HELLO keep_positions: its pool and disk tier), so that a
    /// conversation is matched for as long as the engine holds its state; never under 256 sessions and 16M ids.
    pub fn keep_for(&mut self, keep_positions: u64) {
        self.max_tokens = (keep_positions as usize).max(16 << 20);
        self.max_sessions = (keep_positions as usize / 4096).max(256);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKENIZER: &str = "/data/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/tokenizer.json";

    /// A session of `text` straight into the table and its index (no tokenizer: the index reads only the text).
    fn put(t: &mut SessionTable, id: u64, text: &str, ablit: bool) {
        t.sessions.insert(id, Session { text: text.into(), ids: Vec::new(), ends: Vec::new(), prompt_len: text.len(), busy: false,
                                        last_used: 0, t_used: 0, ablit, images: Vec::new(), keys: Vec::new() });
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
            put_fresh(&mut t, 7, "<｜begin▁of▁sentence｜>a conversation long enough to be kept", false);
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
        let tok = crate::tokenizer::synthetic::tokenizer();
        let mut t = SessionTable::new(8, 1 << 20);
        let c0 = t.changes();
        let p = "<｜begin▁of▁sentence｜><｜System｜>You are a careful assistant who answers briefly and well.<｜User｜>hello<｜Assistant｜></think>";
        let (plan, _) = t.plan(p, &tok, 1, false);
        assert!(t.changes() > c0);
        let c1 = t.changes();
        t.finish(plan.session_id, p, &plan.ids, &tok.encode("Hi."), &tok);
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
        put_fresh(&mut t, 3, "<｜begin▁of▁sentence｜>the first table's conversation", false);
        assert_eq!(t.save(&path, 3600), Ok(1));
        let first = std::fs::read(&path).unwrap();
        // a crash in the middle of the next save: half a file under the temporary name
        let tmp = dir.join("front.table.new");
        std::fs::write(&tmp, &first[..first.len() / 2]).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), first, "the saved file is as it was");
        // the next save replaces the leftover and the file whole
        put_fresh(&mut t, 4, "<｜begin▁of▁sentence｜>and a second conversation", false);
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
        let p1 = "<｜begin▁of▁sentence｜><｜System｜>You are a careful assistant who answers briefly and well.<｜User｜>hello<｜Assistant｜></think>";
        let (plan, _) = t.plan(p1, &tok, 1, false);
        assert_eq!(t.snapshot(3600).1, 0, "a session with no finished turn has nothing to save");
        let gen = tok.encode("Hi there.");
        t.finish(plan.session_id, p1, &plan.ids, &gen, &tok);
        assert_eq!(t.save(&path, 3600), Ok(1));
        // the next turn of the conversation is in its request: the session is busy, and still saved
        let p2 = format!("{p1}Hi there.<｜User｜>next<｜Assistant｜></think>");
        let (plan2, _) = t.plan(&p2, &tok, 1, false);
        assert_eq!(plan2.session_id, plan.session_id);
        assert!(t.sessions[&plan.session_id].busy);
        assert_eq!(t.save(&path, 3600), Ok(1));
        let mut u = SessionTable::new(8, 1 << 20);
        assert_eq!(u.load(&path, &tok, 3600), Ok(1));
        let s = &u.sessions[&plan.session_id];
        assert_eq!(s.text, format!("{p1}Hi there."), "its earlier history, not the turn in flight");
        assert_eq!(s.ids.len(), plan.ids.len() + gen.len());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_load_leaves_the_file_so_a_second_load_finds_the_same_table() {
        let tok = syn();
        let dir = table_dir("twice");
        let path = dir.join("front.table");
        let mut t = SessionTable::new(8, 1 << 20);
        let p1 = "<｜begin▁of▁sentence｜><｜System｜>You are a careful assistant who answers briefly and well.<｜User｜>hello<｜Assistant｜></think>";
        let (plan, _) = t.plan(p1, &tok, 1, false);
        t.finish(plan.session_id, p1, &plan.ids, &tok.encode("Hi there."), &tok);
        assert_eq!(t.save(&path, 3600), Ok(1));
        let bytes = std::fs::read(&path).unwrap();
        let mut a = SessionTable::new(8, 1 << 20);
        assert_eq!(a.load(&path, &tok, 3600), Ok(1));
        assert_eq!(std::fs::read(&path).unwrap(), bytes, "the file is as it was");
        // a crash right after the start: the next start loads the same table
        let mut b = SessionTable::new(8, 1 << 20);
        assert_eq!(b.load(&path, &tok, 3600), Ok(1));
        assert_eq!(b.sessions[&plan.session_id].text, a.sessions[&plan.session_id].text);
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
        put_fresh(&mut t, 3, "<｜begin▁of▁sentence｜>one conversation", false);
        put_fresh(&mut t, 9, "<｜begin▁of▁sentence｜>another, ablated", true);
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
    fn the_index_finds_the_longest_shared_prefix() {
        // conversations that share beginnings of every length (a common system prompt, then branches), against every
        // session read whole: the index's best match shares as many bytes as the best of all (ids may differ on ties)
        let mut seed = 7u64;
        let mut rnd = |n: u64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) % n
        };
        let alpha = |r: u64| (b'a' + (r % 3) as u8) as char;  // three letters: long accidental agreements
        let sys: String = (0..300).map(|i| alpha(i * 7)).collect();
        let mut t = SessionTable::new(1 << 20, 1 << 30);
        let mut texts: Vec<(u64, String, bool)> = Vec::new();
        for id in 1..=400u64 {
            let base = if texts.is_empty() || rnd(4) == 0 { sys.clone() } else {
                let (_, b, _) = &texts[rnd(texts.len() as u64) as usize];
                b[..rnd(b.len() as u64 + 1) as usize].to_string()
            };
            let tail: String = (0..rnd(6000)).map(|_| alpha(rnd(3))).collect();
            let text = base + &tail;
            let ablit = rnd(5) == 0;
            put(&mut t, id, &text, ablit);
            texts.push((id, text, ablit));
        }
        // half of them change their text (a finished turn): out of the index, the new text in
        for k in 0..200 {
            let (id, old, ablit) = texts[k * 2].clone();
            let text = old.clone() + &(0..rnd(3000)).map(|_| alpha(rnd(3))).collect::<String>();
            t.leave(id);
            t.sessions.get_mut(&id).unwrap().text = text.clone();
            t.enter(id);
            texts[k * 2] = (id, text, ablit);
        }
        for q in 0..2000 {
            let (_, b, _) = &texts[rnd(texts.len() as u64) as usize];
            let p = b[..rnd(b.len() as u64 + 1) as usize].to_string() + &(0..rnd(800)).map(|_| alpha(rnd(3))).collect::<String>();
            let ablit = q % 7 == 0;
            let brute = texts.iter().filter(|x| x.2 == ablit).map(|x| common_prefix(p.as_bytes(), x.1.as_bytes())).max().unwrap_or(0);
            let got = t.best(&p, ablit, &[]).map_or(0, |(_, n)| n);
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
        t.keep_for(0);  // an engine that keeps nothing (or an echo engine): the table's own limits, never less
        assert_eq!((t.max_sessions, t.max_tokens), (256, 16 << 20));
        t.keep_for(300_000_000);  // a pool and a 256 GiB disk tier: about 300M positions
        assert_eq!((t.max_sessions, t.max_tokens), (300_000_000 / 4096, 300_000_000));
    }

    #[test]
    fn reuses_the_models_own_ids() {
        let Ok(tok) = Tokenizer::from_file(TOKENIZER) else { return };
        let mut t = SessionTable::new(8, 1 << 20);
        let sys = "<｜begin▁of▁sentence｜><｜System｜>You are a careful assistant who answers briefly and well.";
        let p1 = format!("{sys}<｜User｜>hello<｜Assistant｜></think>");
        let (plan, _) = t.plan(&p1, &tok, 1, false);
        assert_eq!(plan.ids, tok.encode(&p1));
        // the model spelled "Hi there" in single letters, which BPE would never produce
        let mut gen: Vec<u32> = "Hi there".chars().map(|c| tok.encode(&c.to_string())[0]).collect();
        gen.push(tok.token_id("<｜end▁of▁sentence｜>").unwrap());
        t.finish(plan.session_id, &p1, &plan.ids, &gen, &tok);
        let p2 = format!("{p1}Hi there<｜end▁of▁sentence｜><｜User｜>next<｜Assistant｜></think>");
        let (plan2, _) = t.plan(&p2, &tok, 1, false);
        assert_eq!(plan2.session_id, plan.session_id);
        let mut want = plan.ids.clone();
        want.extend(&gen);
        want.extend(tok.encode("<｜User｜>next<｜Assistant｜></think>"));
        assert_eq!(plan2.ids, want);
        assert_eq!(plan2.reused, plan.ids.len() + gen.len());
        assert_eq!(tok.decode(&plan2.ids), p2);
        // the same system prompt in another conversation forks instead of taking over the session
        let p3 = format!("{sys}<｜User｜>something else entirely<｜Assistant｜></think>");
        let (plan3, _) = t.plan(&p3, &tok, 1, false);
        assert_ne!(plan3.session_id, plan.session_id);
        assert_eq!(plan3.base_session_id, plan.session_id);
        assert_eq!(tok.decode(&plan3.ids), p3);
    }

    #[test]
    fn the_plan_says_where_it_starts() {
        let Ok(tok) = Tokenizer::from_file(TOKENIZER) else { return };
        let mut t = SessionTable::new(8, 1 << 20);
        // pi-like: a long system text that ends with the time, then the tools and the first message
        let body = "You are an expert coding assistant. Read files, run commands and edit code.\n".repeat(40);
        let conv = |time: &str, msg: &str| {
            format!("<｜begin▁of▁sentence｜><｜System｜>{body}Current date and time: Monday, October 12, 2026 at {time}\n\n## Tools\n\nnone<｜User｜>{msg}<｜Assistant｜></think>")
        };
        let a = conv("09:14:08 AM", "fix the build");
        let (pa, _) = t.plan(&a, &tok, 1, false);
        assert_eq!(pa.describe(a.len()), format!("session {}, new: no session shares {MIN_SHARED} bytes", pa.session_id));
        t.finish(pa.session_id, &a, &pa.ids, &tok.encode("Done."), &tok);
        // another conversation, a minute later: a fork of the first, which has every id before the time
        let b = conv("09:15:31 AM", "write the tests");
        let (pb, _) = t.plan(&b, &tok, 1, false);
        assert_eq!(pb.base_session_id, pa.session_id);
        assert_eq!(pb.ids, tok.encode(&b), "the ids are a cold run's");
        let head = tok.encode(&b[..b.find("09:15").unwrap()]).len();
        assert!(pb.common + 2 >= head && pb.common <= head + 2, "{} ids in common, {head} before the time", pb.common);
        let d = pb.describe(b.len());
        assert!(d.starts_with(&format!("session {}, a fork of {}: it shares ", pb.session_id, pa.session_id)), "{d}");
        assert!(d.ends_with(&format!("of {} bytes, {} ids in common", b.len(), pb.common)), "{d}");
        // the next turn of the first: in place
        let a2 = format!("{a}Done.<｜end▁of▁sentence｜><｜User｜>thanks<｜Assistant｜></think>");
        let (pa2, _) = t.plan(&a2, &tok, 1, false);
        assert_eq!(pa2.session_id, pa.session_id);
        assert!(pa2.describe(a2.len()).starts_with(&format!("session {} in place: it shares ", pa.session_id)));
        t.finish(pa2.session_id, &a2, &pa2.ids, &tok.encode("Sure."), &tok);
        // a prompt that shares only the BOS and a few bytes: a new session, which names no base
        let c = "<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜></think>";
        let (pc, _) = t.plan(c, &tok, 1, false);
        assert_eq!(pc.base_session_id, 0);
        let n = common_prefix(c.as_bytes(), a.as_bytes());
        assert!(n > 0 && n < MIN_SHARED);
        let d = pc.describe(c.len());
        assert!(d.starts_with(&format!("session {}, new: the best match, session ", pc.session_id)), "{d}");
        assert!(d.ends_with(&format!(", shares {n} bytes (under 64)")), "{d}");
    }

    #[test]
    fn a_saved_table_finds_its_sessions_again() {
        let Ok(tok) = Tokenizer::from_file(TOKENIZER) else { return };
        let dir = std::env::temp_dir().join(format!("dsfront-table-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("front.table");
        let mut t = SessionTable::new(8, 1 << 20);
        let p1 = "<｜begin▁of▁sentence｜><｜System｜>You are a careful assistant who answers briefly and well.<｜User｜>hello<｜Assistant｜></think>";
        let (plan, _) = t.plan(p1, &tok, 1, false);
        let gen = tok.encode("Hi there.");
        t.finish(plan.session_id, p1, &plan.ids, &gen, &tok);
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
        let p2 = format!("{p1}Hi there.<｜User｜>next<｜Assistant｜></think>");
        let (plan2, _) = u.plan(&p2, &tok, 1, false);
        assert_eq!(plan2.session_id, plan.session_id);
        assert_eq!(plan2.reused, plan.ids.len() + gen.len());
        assert_eq!(tok.decode(&plan2.ids), p2);
        // nothing there, a damaged file, a file open to others: nothing taken, the file gone
        assert_eq!(u.load(&path, &tok, 3600), Err(String::new()));
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

    const SYS: &str = "<｜begin▁of▁sentence｜><｜System｜>You are a careful assistant who answers briefly and well, and who never gives a long answer where a short one will do.";

    fn turn(prefix: &str, msg: &str) -> String {
        format!("{prefix}<｜User｜>{msg}<｜Assistant｜></think>")
    }

    #[test]
    fn sessions_are_kept_per_weights_path() {
        let tok = crate::tokenizer::synthetic::tokenizer();
        let mut t = SessionTable::new(8, 1 << 20);
        let p1 = turn(SYS, "hello");
        let (plain, _) = t.plan(&p1, &tok, 1, false);
        t.finish(plain.session_id, &p1, &plain.ids, &tok.encode("Hi.<｜end▁of▁sentence｜>"), &tok);
        // the same prompt on the other path: a new session that names no base, and reuses nothing
        let (abl, _) = t.plan(&p1, &tok, 1, true);
        assert_ne!(abl.session_id, plain.session_id);
        assert_eq!((abl.base_session_id, abl.reused, abl.matched), (0, 0, 0));
        assert_eq!(tok.decode(&abl.ids), p1);
        t.finish(abl.session_id, &p1, &abl.ids, &tok.encode("Hey.<｜end▁of▁sentence｜>"), &tok);
        // the next turn of each conversation continues its own session, in place
        let n1 = format!("{p1}Hi.<｜end▁of▁sentence｜>");
        let (c_plain, _) = t.plan(&turn(&n1, "more"), &tok, 1, false);
        assert_eq!((c_plain.session_id, c_plain.base_session_id), (plain.session_id, 0));
        let n2 = format!("{p1}Hey.<｜end▁of▁sentence｜>");
        let (c_abl, _) = t.plan(&turn(&n2, "more"), &tok, 1, true);
        assert_eq!((c_abl.session_id, c_abl.base_session_id), (abl.session_id, 0));
        // a fork takes its base from the request's own path
        t.finish(c_abl.session_id, &turn(&n2, "more"), &c_abl.ids, &tok.encode("Ok.<｜end▁of▁sentence｜>"), &tok);
        t.finish(c_plain.session_id, &turn(&n1, "more"), &c_plain.ids, &tok.encode("Ok.<｜end▁of▁sentence｜>"), &tok);
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
        let (plain, _) = t.plan(&p1, &tok, 1, false);
        let gen = tok.encode("Hi.<｜end▁of▁sentence｜>");
        t.finish(plain.session_id, &p1, &plain.ids, &gen, &tok);
        let p2 = turn("<｜begin▁of▁sentence｜><｜System｜>Another conversation altogether, on the ablit path, which is as long as the other one.", "hello");
        let (abl, _) = t.plan(&p2, &tok, 1, true);
        t.finish(abl.session_id, &p2, &abl.ids, &gen, &tok);
        assert_eq!(t.save(&path, 3600), Ok(2));
        let mut u = SessionTable::new(8, 1 << 20);
        assert_eq!(u.load(&path, &tok, 3600), Ok(2));
        // each is found again by a request of its own path, and by that one only
        let next = |p: &str| format!("{p}Hi.<｜end▁of▁sentence｜><｜User｜>again<｜Assistant｜></think>");
        let (a, _) = u.plan(&next(&p2), &tok, 1, true);
        assert_eq!((a.session_id, a.base_session_id), (abl.session_id, 0));
        let (b, _) = u.plan(&next(&p1), &tok, 1, false);
        assert_eq!((b.session_id, b.base_session_id), (plain.session_id, 0));
        let (c, _) = u.plan(&next(&p1), &tok, 1, true);
        assert_eq!((c.reused, c.base_session_id), (0, 0));
        // a table of the first layout loads, every session on the plain path
        let ids = tok.encode(&p1);
        write_v1_table(&path, &[(77, &p1, &ids)]);
        let mut v = SessionTable::new(8, 1 << 20);
        assert_eq!(v.load(&path, &tok, 3600), Ok(1));
        let (on, _) = v.plan(&turn(&p1, "more"), &tok, 1, true);
        assert_eq!(on.reused, 0, "an older table's session is not on the ablit path");
        let mut w = SessionTable::new(8, 1 << 20);
        write_v1_table(&path, &[(77, &p1, &ids)]);
        assert_eq!(w.load(&path, &tok, 3600), Ok(1));
        let (off, _) = w.plan(&turn(&p1, "more"), &tok, 1, false);
        assert_eq!((off.session_id, off.reused), (77, ids.len()));
        std::fs::remove_dir_all(&dir).unwrap();
    }

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
        let mut t = SessionTable::new(8, 1 << 20);
        let first = format!("{SYS}<｜User｜>what is on {PH}?<｜Assistant｜></think>");
        let (x1, k1) = with_images(&first, 3, &[11]);
        let (p1, _) = t.plan_images(&x1, &tok, 1, false, &k1);
        assert_eq!((p1.base_session_id, p1.reused), (0, 0));
        assert_eq!(tok.decode(&p1.ids), x1, "the ids spell the expanded text");
        assert_eq!(p1.ids.iter().filter(|&&i| i == tok.token_id(PH).unwrap()).count(), 3);
        let gen = tok.encode("A cat.<｜end▁of▁sentence｜>");
        t.finish_images(p1.session_id, &x1, &p1.ids, &gen, &tok, &k1);
        // the client re-sends its history with the same image: the next turn continues the session in place
        let hist = format!("{first}A cat.<｜end▁of▁sentence｜>");
        let (x2, k2) = with_images(&turn(&hist, "and its colour?"), 3, &[11]);
        let (p2, _) = t.plan_images(&x2, &tok, 1, false, &k2);
        assert_eq!((p2.session_id, p2.base_session_id), (p1.session_id, 0));
        assert_eq!(p2.reused, p1.ids.len() + gen.len());
        assert_eq!(tok.decode(&p2.ids), x2);
        // the same text with another image at that place: the session is no candidate, a cold start
        let (x3, k3) = with_images(&turn(&hist, "and its colour?"), 3, &[12]);
        assert_eq!(x3, x2);
        let (p3, _) = t.plan_images(&x3, &tok, 1, false, &k3);
        assert_ne!(p3.session_id, p1.session_id);
        assert_eq!((p3.base_session_id, p3.reused, p3.matched), (0, 0, 0));
        assert_eq!(p3.ids, p2.ids, "the cold ids are the continued session's");
    }

    #[test]
    fn an_image_after_the_shared_text_keeps_the_earlier_turns() {
        let tok = crate::tokenizer::synthetic::tokenizer_with_images();
        let mut t = SessionTable::new(8, 1 << 20);
        let p1 = turn(SYS, "hello");
        let (plain, _) = t.plan(&p1, &tok, 1, false);
        let gen = tok.encode("Hi.<｜end▁of▁sentence｜>");
        t.finish(plain.session_id, &p1, &plain.ids, &gen, &tok);
        // the second turn brings the first image: it lies after everything the session holds
        let (x2, k2) = with_images(&turn(&format!("{p1}Hi.<｜end▁of▁sentence｜>"), &format!("see {PH}")), 4, &[7]);
        let (p2, _) = t.plan_images(&x2, &tok, 1, false, &k2);
        assert_eq!((p2.session_id, p2.base_session_id), (plain.session_id, 0));
        assert_eq!(p2.reused, plain.ids.len() + gen.len());
        assert_eq!(tok.decode(&p2.ids), x2);
        t.finish_images(p2.session_id, &x2, &p2.ids, &tok.encode("A dog.<｜end▁of▁sentence｜>"), &tok, &k2);
        // a different conversation with the same system prompt and no image forks the session: its
        // shared text ends before the image
        let other = turn(SYS, "something else entirely");
        let (f, _) = t.plan(&other, &tok, 1, false);
        assert_eq!(f.base_session_id, plain.session_id);
        // a conversation that holds the image too, at the same place, forks it with the image
        let (x4, k4) = with_images(&turn(&format!("{p1}Hi.<｜end▁of▁sentence｜>"), &format!("see {PH} now")), 4, &[7]);
        let (p4, _) = t.plan_images(&x4, &tok, 1, false, &k4);
        assert_eq!(p4.base_session_id, plain.session_id);
        assert!(p4.reused >= plain.ids.len() + gen.len());
    }

    #[test]
    fn image_sessions_are_kept_per_weights_path() {
        let tok = crate::tokenizer::synthetic::tokenizer_with_images();
        let mut t = SessionTable::new(8, 1 << 20);
        let raw = format!("{SYS}<｜User｜>what is on {PH}?<｜Assistant｜></think>");
        let (x, k) = with_images(&raw, 3, &[5]);
        let (plain, _) = t.plan_images(&x, &tok, 1, false, &k);
        t.finish_images(plain.session_id, &x, &plain.ids, &tok.encode("Ok.<｜end▁of▁sentence｜>"), &tok, &k);
        let (abl, _) = t.plan_images(&x, &tok, 1, true, &k);
        assert_ne!(abl.session_id, plain.session_id);
        assert_eq!((abl.base_session_id, abl.reused, abl.matched), (0, 0, 0));
        t.finish_images(abl.session_id, &x, &abl.ids, &tok.encode("Ok.<｜end▁of▁sentence｜>"), &tok, &k);
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
        let (p1, _) = t.plan_images(&x1, &tok, 1, false, &k1);
        let gen = tok.encode("A cat.<｜end▁of▁sentence｜>");
        t.finish_images(p1.session_id, &x1, &p1.ids, &gen, &tok, &k1);
        assert_eq!(t.save(&path, 3600), Ok(1));
        let mut u = SessionTable::new(8, 1 << 20);
        assert_eq!(u.load(&path, &tok, 3600), Ok(1));
        let hist = format!("{first}A cat.<｜end▁of▁sentence｜>");
        let (x2, k2) = with_images(&turn(&hist, "and its colour?"), 3, &[11]);
        let (p2, _) = u.plan_images(&x2, &tok, 1, false, &k2);
        assert_eq!((p2.session_id, p2.reused), (p1.session_id, p1.ids.len() + gen.len()));
        let (x3, k3) = with_images(&turn(&hist, "and its colour?"), 3, &[12]);
        let (p3, _) = u.plan_images(&x3, &tok, 1, false, &k3);
        assert_eq!((p3.base_session_id, p3.reused), (0, 0));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
