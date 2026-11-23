//! A request that carries only its suffix: the real router against the mock engine on a local socket, with a real TCP client. What
//! the engine saw of each request is the mock's record (`mock::Seen`); what the client got is the HTTP answer.

use super::*;
use crate::mock::{MockConfig, Seen};
use crate::protocol::{read_frame, Frame, HelloEngine, VERSION, VERSION_PREFIX};
use std::path::PathBuf;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpStream, UnixListener};

struct Rig {
    app: Arc<App>,
    addr: std::net::SocketAddr,
    seen: Arc<Mutex<Vec<Seen>>>,
    path: PathBuf,
    server: tokio::task::JoinHandle<()>,
    mock: tokio::task::JoinHandle<std::io::Result<()>>,
}

impl Drop for Rig {
    fn drop(&mut self) {
        self.server.abort();
        self.mock.abort();
        let _ = std::fs::remove_file(&self.path);
    }
}

/// The front-end over a local TCP port, connected to a mock engine whose newest protocol version is `mock_version`.
async fn rig(tag: &str, mock_version: u32) -> Rig {
    rig_with(tag, mock_version, true).await
}

/// `rig`, with `prefix` false as `--no-prefix` is: the front-end says version 4 whatever the engine speaks.
async fn rig_with(tag: &str, mock_version: u32, prefix: bool) -> Rig {
    let tok = Arc::new(crate::tokenizer::synthetic::tokenizer());
    let path = std::env::temp_dir().join(format!("dsfront-prefix-{}-{tag}.sock", std::process::id()));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let cfg = MockConfig {
        socket: path.clone(),
        tok: tok.clone(),
        delay: Duration::ZERO,
        prefill_tok_per_s: 0.0,
        max_context: 1 << 20,
        version: mock_version,
        seen: Some(seen.clone()),
    };
    let mock = tokio::spawn(crate::mock::run(cfg));
    let engine = EngineClient::start_with(path.clone(), prefix);
    assert!(engine.wait_connected(Duration::from_secs(10)).await.is_some(), "the front-end did not connect to the mock engine");
    let app = App::new(super::tests::cfg(), tok, engine);
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    let routes = router(app.clone());
    let server = tokio::spawn(async move { axum::serve(l, routes).await.unwrap() });
    Rig { app, addr, seen, path, server, mock }
}

/// One non-streaming chat request; the answer's JSON.
async fn chat(addr: std::net::SocketAddr, messages: Value) -> Value {
    let body = json!({"messages": messages, "max_tokens": 300}).to_string();
    let mut s = TcpStream::connect(addr).await.unwrap();
    let head = format!("POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\nconnection: close\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n", body.len());
    s.write_all(head.as_bytes()).await.unwrap();
    s.write_all(body.as_bytes()).await.unwrap();
    let mut got = Vec::new();
    tokio::time::timeout(Duration::from_secs(20), s.read_to_end(&mut got)).await.expect("no answer within 20 s").unwrap();
    let text = String::from_utf8_lossy(&got).to_string();
    let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
    assert!(head.starts_with("HTTP/1.1 200"), "{text}");
    serde_json::from_str(body).unwrap_or_else(|e| panic!("{e}: {body}"))
}

fn reply_of(v: &Value) -> String {
    v["choices"][0]["message"]["content"].as_str().unwrap_or_default().to_string()
}

fn n(v: &Value, path: &[&str]) -> u64 {
    path.iter().fold(v, |v, k| &v[*k]).as_u64().unwrap_or_else(|| panic!("no {path:?} in {v}"))
}

/// One SUBMIT the mock saw: its session, base, prefix (entry and cut) and prompt length.
type SeenSubmit = (u64, u64, Option<(u64, u32)>, usize);

/// The SUBMITs the mock saw, in order.
fn submits(r: &Rig) -> Vec<SeenSubmit> {
    r.seen.lock().unwrap().iter().filter_map(|s| if let Seen::Submit { session, base, prefix, n_prompt, .. } = s { Some((*session, *base, *prefix, *n_prompt)) } else { None }).collect()
}

/// The front-end never holds two connections to the engine: each one is closed before the next opens.
fn never_two_open(seen: &[Seen]) -> bool {
    let mut open = 0i32;
    for s in seen {
        match s {
            Seen::Open => open += 1,
            Seen::Closed => open -= 1,
            _ => {}
        }
        if open > 1 {
            return false;
        }
    }
    true
}

const U1: &str = "please tell me about the build and the tests of this project, in a few words";
const U2: &str = "and the second question about it";
const U3: &str = "one more turn on the same conversation";

fn turn_two(reply1: &str) -> Value {
    json!([{"role": "user", "content": U1}, {"role": "assistant", "content": reply1}, {"role": "user", "content": U2}])
}

/// Against an engine that speaks version 5 the front-end reads its second HELLO, and from the second turn on a request carries the text
/// after the cut and names the entry, so the engine's output is made from the right prompt.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_second_turn_of_a_conversation_carries_only_its_suffix() {
    let r = rig("suffix", VERSION_PREFIX).await;
    assert_eq!(r.app.engine.info().unwrap().version, VERSION_PREFIX, "the second HELLO answered version 5");
    let a = chat(r.addr, json!([{"role": "user", "content": U1}])).await;
    assert_eq!(reply_of(&a), format!("Mock reply to: {U1}"));
    let b = chat(r.addr, turn_two(&reply_of(&a))).await;
    assert_eq!(reply_of(&b), format!("Mock reply to: {U2}"), "the engine built the right prompt from the prefix and the suffix");
    let s = submits(&r);
    assert_eq!(s.len(), 2);
    assert!(s[0].2.is_none() && s[0].3 as u64 == n(&a, &["usage", "prompt_tokens"]), "a first turn carries its whole prompt");
    let (epoch, n_prefix) = s[1].2.expect("the second turn names its prefix");
    assert_ne!(epoch, 0);
    assert_eq!(n_prefix as u64, n(&a, &["usage", "prompt_tokens"]) + n(&a, &["usage", "completion_tokens"]) - 1, "the cut is before the reply's last id");
    assert_eq!((s[1].0, s[1].1), (s[0].0, 0), "the same session, in place");
    assert!((s[1].3 as u32) < n_prefix, "the frame carries the suffix, not the prompt: {} ids after {n_prefix}", s[1].3);
    assert_eq!(n(&b, &["usage", "prompt_tokens"]), n_prefix as u64 + s[1].3 as u64, "the usage counts the whole prompt");
    assert_eq!(n(&b, &["usage", "prompt_tokens_details", "cached_tokens"]), n_prefix as u64);
    // a third turn goes on from the second's DONE
    let c = chat(r.addr, json!([{"role": "user", "content": U1}, {"role": "assistant", "content": reply_of(&a)}, {"role": "user", "content": U2},
                                 {"role": "assistant", "content": reply_of(&b)}, {"role": "user", "content": U3}])).await;
    assert_eq!(reply_of(&c), format!("Mock reply to: {U3}"));
    let s = submits(&r);
    let (e3, n3) = s[2].2.expect("the third turn names its prefix");
    assert_eq!(n3 as u64, n(&b, &["usage", "prompt_tokens"]) + n(&b, &["usage", "completion_tokens"]) - 1);
    assert_eq!(e3, epoch, "the epoch stays while the history only grows");
    assert!(never_two_open(&r.seen.lock().unwrap()));
}

/// The engine no longer holds the entry (its session was released between the turns): DONE says so, the front-end tokenises the whole
/// prompt, sends it again with no prefix, and the answer is the one a cold request gives. The session's cuts come from that request, so
/// the turn after it names a prefix again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_prefix_the_engine_no_longer_holds_is_sent_again_whole() {
    let r = rig("gone", VERSION_PREFIX).await;
    let a = chat(r.addr, json!([{"role": "user", "content": U1}])).await;
    let sid = submits(&r)[0].0;
    r.app.engine.release(sid);
    let b = chat(r.addr, turn_two(&reply_of(&a))).await;
    assert_eq!(reply_of(&b), format!("Mock reply to: {U2}"), "the answer is the cold one");
    let s = submits(&r);
    assert_eq!(s.len(), 3, "the prefix request, then the whole one: {s:?}");
    assert!(s[1].2.is_some(), "the second request named a prefix");
    assert!(s[2].2.is_none() && s[2].0 == s[1].0, "the third is the same session, whole");
    assert_eq!(s[2].3 as u64, n(&b, &["usage", "prompt_tokens"]), "it carries the whole prompt");
    assert!(s[2].3 > s[1].3, "the whole prompt is longer than the suffix");
    let c = chat(r.addr, json!([{"role": "user", "content": U1}, {"role": "assistant", "content": reply_of(&a)}, {"role": "user", "content": U2},
                                 {"role": "assistant", "content": reply_of(&b)}, {"role": "user", "content": U3}])).await;
    assert_eq!(reply_of(&c), format!("Mock reply to: {U3}"));
    let s = submits(&r);
    assert_eq!(s.len(), 4);
    let (_, n4) = s[3].2.expect("after the whole request the next turn names a prefix again");
    assert_eq!(n4 as u64, n(&b, &["usage", "prompt_tokens"]) + n(&b, &["usage", "completion_tokens"]) - 1);
}

/// Against an engine of version 4 the connection says 5, is closed, and the next one says 4: every request carries its whole
/// prompt and no prefix, and the engine is never held by two connections.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn against_an_engine_of_the_older_version_it_keeps_the_older_frames() {
    let r = rig("older", VERSION).await;
    assert_eq!(r.app.engine.info().unwrap().version, VERSION);
    let a = chat(r.addr, json!([{"role": "user", "content": U1}])).await;
    let b = chat(r.addr, turn_two(&reply_of(&a))).await;
    assert_eq!(reply_of(&b), format!("Mock reply to: {U2}"));
    let s = submits(&r);
    assert_eq!(s.len(), 2);
    assert!(s.iter().all(|x| x.2.is_none()), "no prefix on a connection of version 4: {s:?}");
    assert_eq!(s[1].3 as u64, n(&b, &["usage", "prompt_tokens"]), "the whole prompt");
    let seen = r.seen.lock().unwrap().clone();
    assert_eq!(&seen[..5], &[Seen::Open, Seen::Hello(VERSION_PREFIX), Seen::Closed, Seen::Open, Seen::Hello(VERSION)], "{seen:?}");
    assert!(never_two_open(&seen));
}

/// An engine that ignores a HELLO it does not know (it neither closes nor answers) is served with the version-4 frames once the second
/// HELLO has not come within its wait.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_engine_that_never_answers_version_5_is_served_with_version_4() {
    let path = std::env::temp_dir().join(format!("dsfront-prefix-{}-ignore.sock", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).unwrap();
    let hellos = Arc::new(Mutex::new(Vec::new()));
    let h2 = hellos.clone();
    tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else { return };
            let (mut rd, mut wr) = stream.into_split();
            let hello = HelloEngine { version: VERSION, max_context: 1 << 20, vocab_size: 300, eos_id: 1, max_sessions: 4, keep_positions: 1 << 20, name: "ignores".into() };
            let _ = wr.write_all(&Frame::HelloEngine(hello).encode()).await;
            while let Ok(Some((ty, p))) = read_frame(&mut rd).await {
                if let Ok(Frame::Hello { version }) = Frame::decode(ty, &p, |_| 0) {
                    h2.lock().unwrap().push(version);
                }
            }
        }
    });
    let engine = EngineClient::start(path.clone());
    let t = Instant::now();
    let info = engine.wait_connected(Duration::from_secs(10)).await.expect("connected");
    assert_eq!(info.version, VERSION);
    assert!(t.elapsed() >= Duration::from_millis(250), "it waited for the second HELLO: {:?}", t.elapsed());
    assert_eq!(*hellos.lock().unwrap(), vec![VERSION_PREFIX], "it said version 5 once and did not reconnect");
    let _ = std::fs::remove_file(&path);
}

/// `--no-prefix` (the switch-off arm of a timing run): against an engine that speaks version 5, the front-end says version 4, one
/// connection, and every request carries its whole prompt.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn with_no_prefix_every_request_carries_its_whole_prompt() {
    let r = rig_with("noprefix", VERSION_PREFIX, false).await;
    assert_eq!(r.app.engine.info().unwrap().version, VERSION);
    let a = chat(r.addr, json!([{"role": "user", "content": U1}])).await;
    let b = chat(r.addr, turn_two(&reply_of(&a))).await;
    assert_eq!(reply_of(&b), format!("Mock reply to: {U2}"));
    let s = submits(&r);
    assert!(s.iter().all(|x| x.2.is_none()) && s[1].3 as u64 == n(&b, &["usage", "prompt_tokens"]), "{s:?}");
    assert_eq!(&r.seen.lock().unwrap()[..2], &[Seen::Open, Seen::Hello(VERSION)]);
}

/// A saved table of `n` conversations, written by a front-end that served them: the directory and the file.
async fn saved_table(tag: &str, n: usize) -> PathBuf {
    let r = rig(tag, VERSION_PREFIX).await;
    for i in 0..n {
        chat(r.addr, json!([{"role": "user", "content": format!("conversation number {i}: {U1}")}])).await;
    }
    let dir = std::env::temp_dir().join(format!("dsfront-limits-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir(&dir).unwrap();
    let file = dir.join("table");
    let said = r.app.save_sessions(&file, 3600);
    assert!(said.contains(&format!("{n} sessions saved")), "{said}");
    file
}

/// More sessions than the table's floor of 256 are saved; a restart loads them and the engine's HELLO is known: all are kept, with
/// no request after the start.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_saved_table_over_256_sessions_survives_a_restart_with_no_request() {
    let file = saved_table("limits-a", 300).await;
    let r = rig("limits-a2", VERSION_PREFIX).await;
    let said = r.app.load_sessions(&file, 3600);
    assert!(said.contains("300 sessions"), "{said}");
    tokio::time::sleep(Duration::from_millis(500)).await;
    r.app.govern(0);
    assert_eq!(r.app.sessions.lock().unwrap().len(), 300, "the periodic passes trimmed the table below the engine's limits");
    let _ = std::fs::remove_dir_all(file.parent().unwrap());
}

/// The same, with the table loaded before the engine connects (the order of the start): the HELLO that comes later sets the limits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_table_loaded_before_the_engines_hello_keeps_every_session() {
    let file = saved_table("limits-b", 300).await;
    let tok = Arc::new(crate::tokenizer::synthetic::tokenizer());
    let path = std::env::temp_dir().join(format!("dsfront-prefix-{}-limits-b2.sock", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let engine = EngineClient::start_with(path.clone(), true);
    let app = App::new(super::tests::cfg(), tok.clone(), engine);
    let said = app.load_sessions(&file, 3600);
    assert!(said.contains("300 sessions"), "{said}");
    assert_eq!(app.sessions.lock().unwrap().len(), 300, "the load trimmed the table before any limit was known");
    let cfg = MockConfig { socket: path.clone(), tok, delay: Duration::ZERO, prefill_tok_per_s: 0.0, max_context: 1 << 20, version: VERSION_PREFIX, seen: None };
    let mock = tokio::spawn(crate::mock::run(cfg));
    assert!(app.engine.wait_connected(Duration::from_secs(10)).await.is_some());
    tokio::time::sleep(Duration::from_millis(500)).await;
    app.govern(0);
    assert_eq!(app.sessions.lock().unwrap().len(), 300, "the engine's HELLO trimmed the table below its limits");
    mock.abort();
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir_all(file.parent().unwrap());
}

/// One GET to the front-end's port: the status line and head, lower-cased.
async fn get_head(addr: std::net::SocketAddr, path: &str) -> String {
    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(format!("GET {path} HTTP/1.1\r\nhost: 127.0.0.1\r\nconnection: close\r\n\r\n").as_bytes()).await.unwrap();
    let mut got = Vec::new();
    tokio::time::timeout(Duration::from_secs(10), s.read_to_end(&mut got)).await.expect("no answer within 10 s").unwrap();
    String::from_utf8_lossy(&got).split("\r\n\r\n").next().unwrap_or("").to_lowercase()
}

/// The dashboard page and its API answer with the framing headers and set no script policy.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_dashboard_cannot_be_framed() {
    let r = rig("frame", VERSION_PREFIX).await;
    for path in ["/dashboard", "/dashboard/", "/dashboard/api/v1/status"] {
        let head = get_head(r.addr, path).await;
        assert!(head.starts_with("http/1.1 200"), "{path}: {head}");
        assert!(head.contains("x-frame-options: deny"), "{path}: {head}");
        assert!(head.contains("content-security-policy: frame-ancestors 'none'"), "{path}: {head}");
        assert!(!head.contains("script-src"), "{path}: {head}");
    }
}
