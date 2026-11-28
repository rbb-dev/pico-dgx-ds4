//! Tool calls on the wire: the real router against the mock engine on a local socket, with a real TCP
//! client. A streamed call is named in its first chunk, its arguments follow in fragments, one id
//! stays with an index; a reply that is not streamed is the same as it always was.

use super::*;
use crate::mock::MockConfig;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

struct Rig {
    addr: std::net::SocketAddr,
    path: std::path::PathBuf,
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

async fn rig(tag: &str, encoding: Encoding) -> Rig {
    rig_with(tag, encoding, false).await
}

/// `guard`: the repetition guard is on, as in a served front-end.
async fn rig_with(tag: &str, encoding: Encoding, guard: bool) -> Rig {
    let tok = Arc::new(crate::tokenizer::synthetic::tokenizer());
    let path = std::env::temp_dir().join(format!("dsfront-toolstream-{}-{tag}.sock", std::process::id()));
    let cfg = MockConfig {
        socket: path.clone(),
        tok: tok.clone(),
        delay: Duration::ZERO,
        prefill_tok_per_s: 0.0,
        max_context: 1 << 20,
        version: crate::protocol::VERSION_PREFIX,
        seen: None,
    };
    let mock = tokio::spawn(crate::mock::run(cfg));
    let engine = EngineClient::start_with(path.clone(), true);
    assert!(engine.wait_connected(Duration::from_secs(10)).await.is_some(), "the front-end did not connect to the mock engine");
    let mut c = super::tests::cfg();
    c.encoding = encoding;
    if guard {
        c.loop_guard = Some(Default::default());
    }
    let app = App::new(c, tok, engine);
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    let routes = router(app.clone());
    let server = tokio::spawn(async move { axum::serve(l, routes).await.unwrap() });
    Rig { addr, path, server, mock }
}

fn tools(namespaced: bool) -> Value {
    if namespaced {
        json!([{"type": "function", "namespace": {"name": "web", "description": "Web tools."},
                "function": {"name": "search", "description": "Search", "parameters": {"type": "object", "properties": {}}}}])
    } else {
        json!([{"type": "function", "function": {"name": "write", "description": "Write", "parameters": {"type": "object", "properties": {}}}}])
    }
}

/// The request's raw answer: its status line and body.
async fn post(addr: std::net::SocketAddr, body: &Value) -> String {
    let body = body.to_string();
    let mut s = TcpStream::connect(addr).await.unwrap();
    let head = format!("POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\nconnection: close\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n", body.len());
    s.write_all(head.as_bytes()).await.unwrap();
    s.write_all(body.as_bytes()).await.unwrap();
    let mut got = Vec::new();
    tokio::time::timeout(Duration::from_secs(30), s.read_to_end(&mut got)).await.expect("no answer within 30 s").unwrap();
    let text = String::from_utf8_lossy(&got).to_string();
    assert!(text.starts_with("HTTP/1.1 200"), "{text}");
    text
}

fn request(user: &str, ns: bool, stream: bool, max_tokens: Option<u32>) -> Value {
    let mut r = json!({"messages": [{"role": "user", "content": user}], "tools": tools(ns), "stream": stream});
    if let Some(m) = max_tokens {
        r["max_tokens"] = json!(m);
    }
    r
}

/// A non-streaming answer's JSON.
async fn whole(addr: std::net::SocketAddr, user: &str, ns: bool, max_tokens: Option<u32>) -> Value {
    let text = post(addr, &request(user, ns, false, max_tokens)).await;
    let body = text.split_once("\r\n\r\n").unwrap().1;
    serde_json::from_str(body).unwrap_or_else(|e| panic!("{e}: {body}"))
}

/// A streaming answer's chunks, in order.
async fn streamed(addr: std::net::SocketAddr, user: &str, ns: bool, max_tokens: Option<u32>) -> Vec<Value> {
    let text = post(addr, &request(user, ns, true, max_tokens)).await;
    text.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).unwrap_or_else(|e| panic!("{e}: {d}")))
        .collect()
}

fn user_for(calls: &[(&str, Value)]) -> String {
    calls.iter().map(|(n, a)| format!("MOCK_TOOL {n} {a} ")).collect::<String>()
}

fn long_value() -> String {
    (0..200).map(|i| format!("line {i}: \"quoted\" \\ back < angle\n")).collect()
}

/// The tool-call entries of the chunks, in order, and the finish reason.
fn tool_entries(chunks: &[Value]) -> (Vec<Value>, Option<String>) {
    let mut e = Vec::new();
    let mut finish = None;
    for c in chunks {
        for ch in c["choices"].as_array().into_iter().flatten() {
            if let Some(t) = ch["delta"]["tool_calls"].as_array() {
                assert_eq!(t.len(), 1, "one entry a chunk: {c}");
                e.push(t[0].clone());
            }
            if let Some(f) = ch["finish_reason"].as_str() {
                finish = Some(f.to_string());
            }
        }
    }
    (e, finish)
}

/// Per index: (the id of the entry that opens the call, its name, its namespace, the arguments joined).
/// Checks the shape the OpenAI clients rely on: the first entry of an index carries id, type, name and
/// empty arguments; the others only the index and a fragment; indexes never interleave.
fn assemble(entries: &[Value]) -> Vec<(String, String, Option<String>, String, usize)> {
    let mut calls: Vec<(String, String, Option<String>, String, usize)> = Vec::new();
    let mut seen_ids = std::collections::HashSet::new();
    for e in entries {
        let i = e["index"].as_u64().expect("an index on every entry") as usize;
        if i == calls.len() {
            assert!(e["id"].as_str().unwrap().starts_with("call_"), "{e}");
            assert!(seen_ids.insert(e["id"].as_str().unwrap().to_string()), "an id is used for two indexes: {e}");
            assert_eq!(e["type"], "function", "{e}");
            assert_eq!(e["function"]["arguments"], "", "{e}");
            calls.push((e["id"].as_str().unwrap().into(), e["function"]["name"].as_str().unwrap().into(), e["namespace"].as_str().map(String::from), String::new(), 0));
        } else {
            assert_eq!(i + 1, calls.len(), "entries of an earlier call after a later one started: {e}");
            assert!(e.get("id").is_none() && e.get("type").is_none() && e["function"].get("name").is_none(), "only a fragment: {e}");
            calls[i].3.push_str(e["function"]["arguments"].as_str().unwrap());
            calls[i].4 += 1;
        }
    }
    calls
}

/// T1, T2 on the wire: the first entry names the call, the fragments join to what the non-streamed
/// answer carries, the finish is tool_calls.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_call_is_named_first_and_its_fragments_join() {
    let r = rig("one", Encoding::V41).await;
    let user = user_for(&[("write", json!({"path": "/tmp/a", "content": long_value()}))]);
    let chunks = streamed(r.addr, &user, false, None).await;
    let (entries, finish) = tool_entries(&chunks);
    assert_eq!(finish.as_deref(), Some("tool_calls"));
    let calls = assemble(&entries);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].1, "write");
    assert!(calls[0].4 > 5, "the arguments came in {} fragments", calls[0].4);
    // the name chunk is the first of the call and the first chunk after the role chunk and the text
    assert_eq!(entries[0]["function"]["name"], "write");
    let w = whole(r.addr, &user, false, None).await;
    let c = &w["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(c["function"]["arguments"].as_str().unwrap(), calls[0].3);
    assert_eq!(c["function"]["name"], "write");
    assert_eq!(w["choices"][0]["finish_reason"], "tool_calls");
}

/// T5 on the wire: two calls, indexes 0 and 1, two ids, each id on its call's first entry only.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_calls_have_distinct_stable_ids() {
    let r = rig("two", Encoding::V41).await;
    let user = user_for(&[("write", json!({"path": "a", "content": long_value()})), ("write", json!({"path": "b", "n": 2}))]);
    let (entries, finish) = tool_entries(&streamed(r.addr, &user, false, None).await);
    assert_eq!(finish.as_deref(), Some("tool_calls"));
    let calls = assemble(&entries);
    assert_eq!(calls.len(), 2);
    assert_ne!(calls[0].0, calls[1].0);
    let w = whole(r.addr, &user, false, None).await;
    let wc = w["choices"][0]["message"]["tool_calls"].as_array().unwrap();
    assert_eq!(wc.len(), 2);
    for i in 0..2 {
        assert_eq!(wc[i]["function"]["arguments"].as_str().unwrap(), calls[i].3);
    }
}

/// T6 on the wire: under 1017 the name and namespace of a streamed call are those of the whole call.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_namespaced_call_streams_name_and_namespace() {
    let r = rig("ns", Encoding::V41_1017).await;
    let user = user_for(&[("web::search", json!({"q": "streams"}))]);
    let (entries, _) = tool_entries(&streamed(r.addr, &user, true, None).await);
    let calls = assemble(&entries);
    assert_eq!(calls.len(), 1);
    let w = whole(r.addr, &user, true, None).await;
    let c = &w["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(c["function"]["name"], "search");
    assert_eq!(c["namespace"], "web");
    assert_eq!((calls[0].1.as_str(), calls[0].2.as_deref()), ("search", Some("web")));
    assert_eq!(c["function"]["arguments"].as_str().unwrap(), calls[0].3);
}

/// T3 on the wire: the engine stops on the length limit inside a value: the call is there, its
/// arguments close as JSON, and the finish reason is length (not the unfinished block as text).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_call_cut_by_the_length_limit_ends_with_length() {
    let r = rig("cut", Encoding::V41).await;
    let user = user_for(&[("write", json!({"path": "/tmp/a", "content": long_value()}))]);
    let full = whole(r.addr, &user, false, None).await["usage"]["completion_tokens"].as_u64().unwrap() as u32;
    let chunks = streamed(r.addr, &user, false, Some(full / 2)).await;
    let (entries, finish) = tool_entries(&chunks);
    assert_eq!(finish.as_deref(), Some("length"));
    let calls = assemble(&entries);
    assert_eq!(calls.len(), 1, "the call that was named is still there");
    let args: Value = serde_json::from_str(&calls[0].3).unwrap_or_else(|e| panic!("{e}: {}", calls[0].3));
    assert!(args["content"].as_str().unwrap().starts_with("line 0:"));
}

/// T4: what a reply that is not streamed carries, for a set of calls and a cut-off one: equal to the
/// golden of whole replies (ids replaced; the answers' own ids and times are random).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn whole_replies_are_unchanged() {
    let r = rig("golden", Encoding::V41).await;
    let cases: Vec<Vec<(&str, Value)>> = vec![
        vec![("write", json!({"path": "a.txt", "n": 3}))],
        vec![("write", json!({}))],
        vec![("write", json!({"s": ""}))],
        vec![("write", json!({"content": "line \"one\"\nline two\n\ttab \\ back <b>x</b> a < b"}))],
        vec![("write", json!({"a": 1.5, "b": true, "c": null, "d": [1, 2, {"x": "y"}], "e": {"k": "v"}, "n": -7}))],
        vec![("write", json!({"t": "héllo 世界 🎉"}))],
        vec![("write", json!({"path": "/tmp/big.txt", "content": long_value()}))],
        vec![("write", json!({"path": "a"})), ("write", json!({"q": "rust \"streams\"", "k": 5})), ("write", json!({}))],
    ];
    let mut got = Vec::new();
    let norm = |w: &Value| {
        let mut m = w["choices"][0]["message"].clone();
        for c in m["tool_calls"].as_array_mut().into_iter().flatten() {
            assert!(c["id"].as_str().unwrap().starts_with("call_"));
            c["id"] = json!("call_ID");
        }
        json!({"finish_reason": w["choices"][0]["finish_reason"], "message": m})
    };
    for c in &cases {
        got.push(norm(&whole(r.addr, &user_for(c), false, None).await));
    }
    // cut off inside the value: the block comes back as text
    let user = user_for(&[("write", json!({"path": "/tmp/a", "content": long_value()}))]);
    let full = whole(r.addr, &user, false, None).await["usage"]["completion_tokens"].as_u64().unwrap() as u32;
    got.push(norm(&whole(r.addr, &user, false, Some(full / 2)).await));
    got.push(norm(&whole(r.addr, &user, false, Some(full - 3)).await));
    let actual = serde_json::to_string_pretty(&Value::Array(got)).unwrap();
    let golden = include_str!("server_toolstream_golden.json");
    assert_eq!(actual.trim(), golden.trim(), "the answers now:\n{actual}");
}

/// A reply written by hand: a call with a path and a content value.
fn raw_call(second_name: &str, value: &str) -> String {
    format!("MOCK_RAW Go.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">a.txt</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"{second_name}\" string=\"true\">{value}</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>")
}

/// A streamed request with more fields (a stop list).
async fn streamed_extra(addr: std::net::SocketAddr, user: &str, extra: Value) -> Vec<Value> {
    let mut req = request(user, false, true, None);
    for (k, v) in extra.as_object().unwrap() {
        req[k] = v.clone();
    }
    let text = post(addr, &req).await;
    text.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).unwrap())
        .collect()
}

/// What an incomplete call must look like to a client: named, its arguments closed JSON, and the reply
/// ending "length", so a client does not run it as a finished call.
fn incomplete_call(chunks: &[Value]) {
    let (entries, finish) = tool_entries(chunks);
    assert_eq!(finish.as_deref(), Some("length"), "an incomplete call ends the reply with length");
    let calls = assemble(&entries);
    assert_eq!(calls.len(), 1, "the call that was named is there");
    serde_json::from_str::<Value>(&calls[0].3).unwrap_or_else(|e| panic!("{e}: {}", calls[0].3));
}

/// (a) A call that was named and then fails the grammar (a repeated parameter name), the reply ending
/// by EOS: the engine's reason is Stop, and the reply still ends "length" (not "stop" with a closed,
/// short call in it).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_named_call_that_fails_the_grammar_ends_with_length() {
    let r = rig("bad", Encoding::V41).await;
    incomplete_call(&streamed(r.addr, &raw_call("path", "second"), false, None).await);
}

/// (b) A stop string hit inside a named call's value.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stop_string_inside_a_call_ends_with_length() {
    let r = rig("stop", Encoding::V41).await;
    let chunks = streamed_extra(r.addr, &raw_call("content", "first half STOPHERE second half"), json!({"stop": ["STOPHERE"]})).await;
    incomplete_call(&chunks);
    let (entries, _) = tool_entries(&chunks);
    assert_eq!(assemble(&entries)[0].3, "{\"path\": \"a.txt\", \"content\": \"first half \"}");
}

/// (c) The repetition guard firing inside a call's value (eight identical lines).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_loop_guard_inside_a_call_ends_with_length() {
    let r = rig_with("loop", Encoding::V41, true).await;
    let value = "this line repeats itself\n".repeat(60);
    incomplete_call(&streamed(r.addr, &raw_call("content", &value), false, None).await);
}

/// (d) A complete call followed by EOS is still "tool_calls", by the mock's own script and by hand.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_complete_call_still_ends_with_tool_calls() {
    let r = rig("done", Encoding::V41).await;
    for user in [raw_call("content", "all of it"), user_for(&[("write", json!({"path": "a", "content": "all of it"}))])] {
        let (entries, finish) = tool_entries(&streamed(r.addr, &user, false, None).await);
        assert_eq!(finish.as_deref(), Some("tool_calls"));
        assert_eq!(assemble(&entries).len(), 1);
    }
}
