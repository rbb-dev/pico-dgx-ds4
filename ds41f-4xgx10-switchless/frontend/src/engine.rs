//! Client side of the rank-0 socket: one connection, many requests multiplexed by req_id.
//! Reconnects with backoff; requests in flight when the connection drops end with an error.

use crate::dashboard::{ReqLive, ReqMeta, Telemetry};
use crate::protocol::{read_frame, Done, DoneReason, Frame, HelloEngine, Submit, Tokens, VERSION, VERSION_PREFIX};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;
use tokio::sync::{mpsc, watch};

#[derive(Debug)]
pub enum Event {
    Tokens(Tokens),
    Done(Done),
}

struct Pending {
    tx: mpsc::UnboundedSender<Event>,
    n_top: u32,
    live: Arc<ReqLive>,
}

#[derive(Default)]
struct Conn {
    tx: Option<mpsc::UnboundedSender<Vec<u8>>>,
    pending: HashMap<u64, Pending>,
}

struct Inner {
    path: PathBuf,
    conn: Mutex<Conn>,
    next_req: AtomicU64,
    info: watch::Sender<Option<HelloEngine>>,
    telemetry: Arc<Telemetry>,
    /// Whether the next connection says `VERSION_PREFIX`: true until a connection that said it ended without the engine's answer
    /// (an engine of the older version closes at it), true again whenever the engine cannot be reached (if `allow_prefix`).
    try_prefix: AtomicBool,
    /// False: every connection says `VERSION` and every request carries its whole prompt (`--no-prefix`).
    allow_prefix: bool,
}

/// How long a connection that said `VERSION_PREFIX` waits for the engine's second HELLO before it serves with the version-4
/// frames (an engine that ignores the unknown version, as a stand-in may, never answers).
const SECOND_HELLO_WAIT: Duration = Duration::from_millis(300);

#[derive(Clone)]
pub struct EngineClient {
    inner: Arc<Inner>,
}

impl EngineClient {
    /// Start connecting to the engine socket (in the background, forever).
    pub fn start(path: PathBuf) -> EngineClient {
        EngineClient::start_with(path, true)
    }

    /// `start`, with `prefix` false to say the older version at every connection: requests then carry their whole prompt.
    pub fn start_with(path: PathBuf, prefix: bool) -> EngineClient {
        let (info, _) = watch::channel(None);
        let inner = Arc::new(Inner {
            path,
            conn: Mutex::new(Conn::default()),
            next_req: AtomicU64::new(1),
            info,
            telemetry: Arc::new(Telemetry::new()),
            try_prefix: AtomicBool::new(prefix),
            allow_prefix: prefix,
        });
        tokio::spawn(run(inner.clone()));
        EngineClient { inner }
    }

    /// The engine's HELLO while connected.
    pub fn info(&self) -> Option<HelloEngine> {
        self.inner.info.borrow().clone()
    }

    /// The engine's HELLO as it changes: Some on every connection, None while none is open.
    pub fn watch_info(&self) -> watch::Receiver<Option<HelloEngine>> {
        self.inner.info.subscribe()
    }

    pub async fn wait_connected(&self, timeout: Duration) -> Option<HelloEngine> {
        let mut rx = self.inner.info.subscribe();
        let _ = tokio::time::timeout(timeout, rx.wait_for(|i| i.is_some())).await;
        self.info()
    }

    /// What the dashboard knows of the engine and its requests (dashboard.rs).
    pub fn telemetry(&self) -> &Arc<Telemetry> {
        &self.inner.telemetry
    }

    /// Send a request (its req_id is assigned here). Events arrive on the receiver; the last one is
    /// always Done. `meta` is what the dashboard shows of it.
    pub fn submit(&self, mut s: Submit, meta: ReqMeta) -> Result<(u64, mpsc::UnboundedReceiver<Event>), String> {
        s.req_id = self.inner.next_req.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = mpsc::unbounded_channel();
        let mut c = self.inner.conn.lock().unwrap();
        let wtx = c.tx.clone().ok_or("engine not connected")?;
        let req_id = s.req_id;
        let live = self.inner.telemetry.register(req_id, meta);
        c.pending.insert(req_id, Pending { tx, n_top: if s.flags & crate::protocol::FLAG_LOGPROBS != 0 { s.n_top_logprobs } else { 0 }, live });
        // encoded from the request itself, which is dropped with it: a request with images carries their
        // rows (10 KiB each), and a clone here was one more copy of them
        if wtx.send(Frame::Submit(s).encode()).is_err() {
            c.pending.remove(&req_id);
            self.inner.telemetry.finish(&Done {
                req_id,
                reason: DoneReason::Error,
                prompt_tokens: 0,
                cached_tokens: 0,
                completion_tokens: 0,
                queue_us: 0,
                prefill_us: 0,
                decode_us: 0,
                error: String::new(),
                tail: None,
            });
            return Err("engine not connected".into());
        }
        Ok((req_id, rx))
    }

    /// Ask rank 0 for a STATS frame (the answer lands in the telemetry).
    pub fn request_stats(&self) {
        self.send(Frame::StatsReq);
    }

    /// Interleaved prefill's settings to rank 0, applied at once (the dashboard's live settings).
    pub fn configure(&self, c: crate::protocol::ConfigFrame) {
        self.send(Frame::Config(c));
    }

    pub fn cancel(&self, req_id: u64) {
        self.send(Frame::Cancel { req_id });
    }

    pub fn release(&self, session_id: u64) {
        self.send(Frame::Release { session_id });
    }

    fn send(&self, f: Frame) {
        if let Some(tx) = &self.inner.conn.lock().unwrap().tx {
            let _ = tx.send(f.encode());
        }
    }
}

async fn run(inner: Arc<Inner>) {
    let mut backoff = Duration::from_millis(100);
    let mut logged_down = false;
    loop {
        match UnixStream::connect(&inner.path).await {
            Ok(stream) => {
                backoff = Duration::from_millis(100);
                logged_down = false;
                match connection(&inner, stream).await {
                    Ok(()) => eprintln!("engine: connection closed"),
                    Err(e) => eprintln!("engine: connection lost: {e}"),
                }
                inner.info.send_replace(None);
                inner.telemetry.connected(false);
                let pending = {
                    let mut c = inner.conn.lock().unwrap();
                    c.tx = None;
                    std::mem::take(&mut c.pending)
                };
                inner.telemetry.clear_live();
                for (req_id, p) in pending {
                    let _ = p.tx.send(Event::Done(Done {
                        req_id,
                        reason: DoneReason::Error,
                        prompt_tokens: 0,
                        cached_tokens: 0,
                        completion_tokens: 0,
                        queue_us: 0,
                        prefill_us: 0,
                        decode_us: 0,
                        error: "engine connection lost".into(),
                        tail: None,
                    }));
                }
            }
            Err(e) => {
                // an engine that comes back may be a newer one
                inner.try_prefix.store(inner.allow_prefix, Ordering::Relaxed);
                if !logged_down {
                    eprintln!("engine: cannot connect to {}: {e}; retrying", inner.path.display());
                    logged_down = true;
                }
            }
        }
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(Duration::from_secs(2));
    }
}

async fn connection(inner: &Arc<Inner>, stream: UnixStream) -> Result<(), String> {
    let (mut rd, mut wr) = stream.into_split();
    let say = if inner.try_prefix.load(Ordering::Relaxed) { VERSION_PREFIX } else { VERSION };
    wr.write_all(&Frame::Hello { version: say }.encode()).await.map_err(|e| e.to_string())?;
    // The engine speaks first, in version 4; to a front-end that said version 5 it then answers with a second HELLO.
    let (ty, p) = read_frame(&mut rd).await.map_err(|e| e.to_string())?.ok_or("closed before HELLO")?;
    let greeting = match Frame::decode(ty, &p, |_| 0)? {
        Frame::HelloEngine(h) => h,
        f => return Err(format!("expected HELLO, got {f:?}")),
    };
    if greeting.version != VERSION && greeting.version != VERSION_PREFIX {
        return Err(format!("engine speaks protocol version {}, we speak {VERSION}", greeting.version));
    }
    eprintln!(
        "engine: connected to {} ({}; context {}, vocab {}, eos {}, {} sessions)",
        inner.path.display(),
        greeting.name,
        greeting.max_context,
        greeting.vocab_size,
        greeting.eos_id,
        greeting.max_sessions
    );
    let (tx, mut rx) = mpsc::unbounded_channel::<Vec<u8>>();
    let publish = |h: HelloEngine| {
        inner.conn.lock().unwrap().tx = Some(tx.clone());
        inner.info.send_replace(Some(h));
        inner.telemetry.connected(true);
    };
    let mut published = say == VERSION || greeting.version == VERSION_PREFIX;
    let mut got_second = greeting.version == VERSION_PREFIX;
    if published {
        publish(greeting.clone());
    }
    let writer = tokio::spawn(async move {
        while let Some(b) = rx.recv().await {
            if wr.write_all(&b).await.is_err() {
                break;
            }
        }
    });
    let timer = tokio::time::sleep(SECOND_HELLO_WAIT);
    tokio::pin!(timer);
    let result = loop {
        let mut next = std::pin::pin!(read_frame(&mut rd));
        let read = loop {
            tokio::select! {
                r = &mut next => break r,
                _ = &mut timer, if !published => {
                    published = true;
                    eprintln!("engine: no answer to version {say}; sending whole prompts (version {VERSION} frames)");
                    publish(greeting.clone());
                }
            }
        };
        let (ty, p) = match read {
            Ok(Some(f)) => f,
            Ok(None) => break Ok(()),
            Err(e) => break Err(e.to_string()),
        };
        let frame = {
            let c = inner.conn.lock().unwrap();
            Frame::decode(ty, &p, |id| c.pending.get(&id).map_or(0, |p| p.n_top))
        };
        match frame {
            Ok(Frame::HelloEngine(h)) if h.version == VERSION_PREFIX && !got_second => {
                got_second = true;
                published = true;
                eprintln!("engine: version {VERSION_PREFIX}: requests carry the suffix of their prompt");
                publish(h);
            }
            Ok(Frame::Tokens(t)) => {
                let c = inner.conn.lock().unwrap();
                if let Some(p) = c.pending.get(&t.req_id) {
                    inner.telemetry.tokens(&p.live, t.ids.len());
                    let _ = p.tx.send(Event::Tokens(t));
                }
            }
            Ok(Frame::Progress { req_id, computed, total }) => {
                let c = inner.conn.lock().unwrap();
                if let Some(p) = c.pending.get(&req_id) {
                    inner.telemetry.progress(&p.live, computed, total);
                }
            }
            Ok(Frame::Stats(s)) => inner.telemetry.set_stats(s),
            Ok(Frame::Done(d)) => {
                let p = inner.conn.lock().unwrap().pending.remove(&d.req_id);
                if let Some(p) = p {
                    inner.telemetry.finish(&d);
                    let _ = p.tx.send(Event::Done(d));
                }
            }
            Ok(f) => break Err(format!("unexpected frame from engine: {f:?}")),
            Err(e) => break Err(e),
        }
    };
    writer.abort();
    if say == VERSION_PREFIX && !got_second {
        // an engine that closes at the version it does not know: the next connection says the old one
        inner.try_prefix.store(false, Ordering::Relaxed);
    }
    result
}
