//! A client that leaves ends its request on the engine: the real router (`stream()` and `chat()`) against a
//! fake engine on a local socket, a real TCP client that is closed while the engine says nothing, and the
//! cancel frame the engine side must then receive.

use super::*;
use crate::protocol::{read_frame, Frame, HelloEngine, VERSION};
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpStream, UnixListener};

/// How long a streaming request may take to be cancelled after its client closed. hyper sees the closed
/// connection at once; the fallback is the keep-alive write failing (KEEP_ALIVE, twice: the first write to a
/// closed peer is answered with a reset, the second fails), plus slack.
const STREAM_BOUND: Duration = Duration::from_secs(12);
/// A non-streaming request is dropped with its handler future when hyper sees the connection close.
const CHAT_BOUND: Duration = Duration::from_secs(3);

/// A rank-0 stand-in that never answers a request: it reports the req_id of every SUBMIT and CANCEL it reads.
struct FakeEngine {
    path: PathBuf,
    submits: mpsc::UnboundedReceiver<u64>,
    cancels: mpsc::UnboundedReceiver<u64>,
}

impl Drop for FakeEngine {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn fake_engine(tag: &str, vocab: usize) -> FakeEngine {
    let path = std::env::temp_dir().join(format!("dsfront-cancel-{}-{tag}.sock", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).unwrap();
    let (stx, submits) = mpsc::unbounded_channel();
    let (ctx, cancels) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else { return };
            let (mut rd, mut wr) = stream.into_split();
            let hello = HelloEngine {
                version: VERSION,
                max_context: 1 << 20,
                vocab_size: vocab as u32,
                eos_id: 1,
                max_sessions: 64,
                name: "fake".into(),
            };
            if wr.write_all(&Frame::HelloEngine(hello).encode()).await.is_err() {
                continue;
            }
            // `wr` stays open until the front-end closes: dropping it would end the connection
            while let Ok(Some((ty, p))) = read_frame(&mut rd).await {
                match Frame::decode(ty, &p, |_| 0) {
                    Ok(Frame::Submit(s)) => {
                        let _ = stx.send(s.req_id);
                    }
                    Ok(Frame::Cancel { req_id }) => {
                        let _ = ctx.send(req_id);
                    }
                    _ => {}
                }
            }
        }
    });
    FakeEngine { path, submits, cancels }
}

/// The front-end over a local TCP port, connected to a fresh fake engine.
async fn rig(tag: &str) -> (Arc<App>, std::net::SocketAddr, FakeEngine, tokio::task::JoinHandle<()>) {
    let tok = Arc::new(crate::tokenizer::synthetic::tokenizer());
    let fake = fake_engine(tag, tok.vocab_size());
    let engine = EngineClient::start(fake.path.clone());
    assert!(engine.wait_connected(Duration::from_secs(5)).await.is_some(), "the front-end did not connect to the fake engine");
    let app = App::new(super::tests::cfg(), tok, engine);
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    let routes = router(app.clone());
    let server = tokio::spawn(async move { axum::serve(l, routes).await.unwrap() });
    (app, addr, fake, server)
}

async fn post(addr: std::net::SocketAddr, stream: bool) -> TcpStream {
    let body = json!({"messages": [{"role": "user", "content": "hi"}], "stream": stream}).to_string();
    let mut s = TcpStream::connect(addr).await.unwrap();
    let head = format!("POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n", body.len());
    s.write_all(head.as_bytes()).await.unwrap();
    s.write_all(body.as_bytes()).await.unwrap();
    s
}

async fn next_id(rx: &mut mpsc::UnboundedReceiver<u64>, within: Duration, what: &str) -> u64 {
    tokio::time::timeout(within, rx.recv()).await.unwrap_or_else(|_| panic!("{what}: nothing within {within:?}")).expect("fake engine ended")
}

/// Open a request, wait until the engine has it, close the client, and require the engine to be sent the
/// cancel for that very request id within `bound`; then the gate must count the request out.
async fn client_leaves(tag: &str, stream: bool, bound: Duration) {
    let (app, addr, mut fake, server) = rig(tag).await;
    let client = post(addr, stream).await;
    let req_id = next_id(&mut fake.submits, Duration::from_secs(5), "submit").await;
    assert!(app.gate.status(true).contains("in_flight 1"), "{}", app.gate.status(true));
    drop(client);
    let t = Instant::now();
    let got = next_id(&mut fake.cancels, bound, "cancel after the client closed").await;
    assert_eq!(got, req_id, "the cancel names another request");
    eprintln!("{tag}: cancel after {:?}", t.elapsed());
    for _ in 0..50 {
        if app.gate.status(true).contains("in_flight 0") {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(app.gate.status(true).contains("in_flight 0"), "the request stays counted: {}", app.gate.status(true));
    server.abort();
}

/// Control: a cancel sent by the front-end's engine client reaches the fake engine's channel, so a
/// silent channel in the tests below means no cancel was sent.
#[tokio::test]
async fn the_fake_engine_sees_a_cancel() {
    let (app, _addr, mut fake, server) = rig("null").await;
    app.engine.cancel(4242);
    assert_eq!(next_id(&mut fake.cancels, Duration::from_secs(2), "direct cancel").await, 4242);
    server.abort();
}

/// A streaming request whose engine says nothing: the client closing must cancel it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn closed_streaming_client_cancels_its_request() {
    client_leaves("stream", true, STREAM_BOUND).await;
}

/// The same for a non-streaming request, which lives in the handler future.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn closed_non_streaming_client_cancels_its_request() {
    client_leaves("chat", false, CHAT_BOUND).await;
}
