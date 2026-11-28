//! Tool calls streamed as they are written: the call is named as soon as its name has parsed, its
//! arguments follow in fragments that join to exactly the JSON the whole-call parser builds, a call
//! cut off mid-value ends as closed JSON, and a reply that is sent whole is unchanged.

use super::*;

/// What a streamed tool call looks like to a client, whatever the parser's deltas are called.
#[derive(Debug, Clone, PartialEq)]
enum Ev {
    Start(usize, String),
    Args(usize, String),
    End(usize),
}

// The adapter between the parser's deltas and `Ev`.
fn events(d: &Delta) -> Vec<Ev> {
    match d {
        Delta::ToolCallStart { index, name } => vec![Ev::Start(*index, name.clone())],
        Delta::ToolCallArgs { index, fragment } => vec![Ev::Args(*index, fragment.clone())],
        Delta::ToolCallEnd { index } => vec![Ev::End(*index)],
        _ => vec![],
    }
}

fn truncated(p: &StreamParser) -> bool {
    p.calls_truncated()
}

fn whole_mode(p: &mut StreamParser) {
    p.hold_calls();
}
// end of the adapter

/// One call as a client assembles it.
#[derive(Debug, Clone, PartialEq)]
struct Call {
    index: usize,
    name: String,
    args: String,
    fragments: usize,
}

/// Everything a run of the parser produced.
struct Run {
    /// every event with the number of characters fed when it came out
    events: Vec<(usize, Ev)>,
    reasoning: String,
    content: String,
    truncated: bool,
}

impl Run {
    /// The calls, with the order the client relies on checked: a call's fragments follow its start
    /// and end before the next call starts, so two calls never interleave.
    fn calls(&self) -> Vec<Call> {
        let mut calls: Vec<Call> = Vec::new();
        let mut open: Option<Call> = None;
        for (_, e) in &self.events {
            match e {
                Ev::Start(i, name) => {
                    assert!(open.is_none(), "a call starts inside another: {:?}", self.events);
                    assert_eq!(*i, calls.len(), "indexes count up from 0");
                    open = Some(Call { index: *i, name: name.clone(), args: String::new(), fragments: 0 });
                }
                Ev::Args(i, f) => {
                    let c = open.as_mut().expect("a fragment outside a call");
                    assert_eq!(*i, c.index, "fragments of two calls interleave");
                    c.args.push_str(f);
                    c.fragments += 1;
                }
                Ev::End(i) => {
                    let c = open.take().expect("an end outside a call");
                    assert_eq!(*i, c.index);
                    calls.push(c);
                }
            }
        }
        assert!(open.is_none(), "a call never ended");
        calls
    }

    fn pairs(&self) -> Vec<(String, String)> {
        self.calls().into_iter().map(|c| (c.name, c.args)).collect()
    }
}

/// Feed `text` in pieces of the given sizes (in characters, cycling), then finish.
fn feed(text: &str, mode: ThinkingMode, sizes: &[usize], whole: bool) -> Run {
    let mut p = StreamParser::new(mode);
    if whole {
        whole_mode(&mut p);
    }
    let mut out = Vec::new();
    let (mut run, mut fed) = (Run { events: vec![], reasoning: String::new(), content: String::new(), truncated: false }, 0usize);
    let chars: Vec<char> = text.chars().collect();
    let mut at = 0;
    let mut k = 0;
    let take = |out: &mut Vec<Delta>, run: &mut Run, fed: usize| {
        for d in out.drain(..) {
            match &d {
                Delta::Reasoning(s) => run.reasoning.push_str(s),
                Delta::Content(s) => run.content.push_str(s),
                _ => {}
            }
            for e in events(&d) {
                run.events.push((fed, e));
            }
        }
    };
    while at < chars.len() {
        let n = sizes[k % sizes.len()].max(1).min(chars.len() - at);
        k += 1;
        p.push(&chars[at..at + n].iter().collect::<String>(), &mut out);
        at += n;
        fed = at;
        take(&mut out, &mut run, fed);
    }
    p.finish(&mut out);
    take(&mut out, &mut run, fed);
    run.truncated = truncated(&p);
    run
}

/// The pieces of `text` split between word characters and everything else, as a rough stand-in for tokens.
fn token_sizes(text: &str) -> Vec<usize> {
    let mut sizes = Vec::new();
    let mut run = 0;
    let mut word = false;
    for c in text.chars() {
        let w = c.is_alphanumeric();
        if run > 0 && (w != word || !w) {
            sizes.push(run);
            run = 0;
        }
        word = w;
        run += 1;
    }
    if run > 0 {
        sizes.push(run);
    }
    sizes
}

/// Chunk sizes from a fixed pseudo-random sequence.
fn random_sizes(seed: u64, max: u64) -> Vec<usize> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..4096)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            1 + (x % max) as usize
        })
        .collect()
}

fn block(calls: &[(&str, Value)]) -> String {
    let c: Vec<(Value, Value)> = calls.iter().map(|(n, a)| (json!(n), a.clone())).collect();
    format!("Calling.{}{EOS}", render_tool_calls(&c))
}

fn long_text() -> String {
    (0..120).map(|i| format!("line {i}: \"quoted\" \\ back < angle </p> 世界 🎉\n")).collect()
}

/// The cases: arguments of every kind, one call and several, names with a namespace, and the texts
/// of the existing parser tests and of the cases the streaming fork was checked with.
fn corpus() -> Vec<String> {
    let mut v = vec![
        block(&[("read", json!({"path": "a.txt", "n": 3}))]),
        block(&[("ls", json!({}))]),
        block(&[("w", json!({"s": ""}))]),
        block(&[("w", json!({"content": "line \"one\"\nline two\n\ttab \\ back <b>x</b> a < b <"}))]),
        block(&[("w", json!({"x": "ends with a newline\n"}))]),
        block(&[("w", json!({"x": "ends with an angle <"}))]),
        block(&[("w", json!({"a": 1.5, "b": true, "c": null, "d": [1, 2, {"x": "y"}], "e": {"k": "v"}, "n": -7}))]),
        block(&[("w", json!({"t": "héllo 世界 🎉 \u{7f} \u{1} end"}))]),
        block(&[("write", json!({"path": "/tmp/big.txt", "content": long_text()}))]),
        block(&[("read", json!({"path": "a"})), ("web::search", json!({"q": "rust \"streams\"", "k": 5}))]),
        block(&[("a", json!({})), ("b", json!({"x": 1})), ("c", json!({"y": "z\nz", "x": false}))]),
        "Let me look.</think>Checking.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">a.txt</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"n\" string=\"false\">3</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"ls\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜>".to_string(),
        "Hi.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">/tmp/a.html</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"n\" string=\"false\">7</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>".to_string(),
        "Hi.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"w\">\n<｜DSML｜ parameter name=\"v\" string=\"true\">a\n</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>".to_string(),
    ];
    // the reference also takes a value that ends `<` and a newline before the end tag's slash
    v.push("Hi.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"w\">\n<｜DSML｜ parameter name=\"v\" string=\"true\">abc<\n/｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>".to_string());
    v
}

/// The chunkings every case is fed in: one character, one token, and fixed pseudo-random sizes.
fn chunkings(text: &str) -> Vec<(String, Vec<usize>)> {
    let mut v = vec![("char".to_string(), vec![1]), ("token".to_string(), token_sizes(text))];
    for seed in 1..=6u64 {
        v.push((format!("random{seed}"), random_sizes(seed, if seed % 2 == 0 { 5 } else { 40 })));
    }
    v.push(("whole".to_string(), vec![usize::MAX / 2]));
    v
}

fn mode_of(text: &str) -> ThinkingMode {
    if text.contains("</think>") {
        ThinkingMode::Thinking
    } else {
        ThinkingMode::Chat
    }
}

/// T1: a call is named while its arguments are still being written, before its closing tag.
#[test]
fn t1_the_name_comes_before_the_closing_tag() {
    let text = "Hi.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">/tmp/a.html</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"n\" string=\"false\">7</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    let closing = text.find("</｜DSML｜ invoke").unwrap();
    let before = text[..closing].chars().count();
    let run = feed(text, ThinkingMode::Chat, &[1], false);
    let start = run.events.iter().find(|(_, e)| matches!(e, Ev::Start(0, _))).expect("the call is never named");
    assert_eq!(start.1, Ev::Start(0, "write".into()));
    assert!(start.0 < before, "named only after {} characters, the closing tag starts at {before}", start.0);
    let first_arg = run.events.iter().find(|(_, e)| matches!(e, Ev::Args(..))).unwrap();
    assert!(first_arg.0 < before, "no argument before the closing tag");
    // the name is out before the first parameter's value has arrived
    assert!(start.0 <= text[..text.find("/tmp/a.html").unwrap()].chars().count());
    assert_eq!(run.calls(), vec![Call { index: 0, name: "write".into(), args: "{\"path\": \"/tmp/a.html\", \"n\": 7}".into(), fragments: run.calls()[0].fragments }]);
}

/// T2: the fragments join to the arguments the whole-call parser builds, byte for byte, however the
/// text is cut; reasoning and content are those of the whole parse too.
#[test]
fn t2_fragments_join_to_the_whole_call() {
    let mut cases = 0;
    for text in corpus() {
        let mode = mode_of(&text);
        let want = parse_message_from_completion_text(&text, mode, Encoding::V41).expect(&text);
        assert!(!want.tool_calls.is_empty());
        for (label, sizes) in chunkings(&text) {
            let run = feed(&text, mode, &sizes, false);
            assert_eq!(run.pairs(), want.tool_calls, "chunking {label}, text {text:?}");
            assert_eq!(run.reasoning, want.reasoning_content, "chunking {label}");
            assert_eq!(run.content, want.content, "chunking {label}");
            assert!(!run.truncated);
            for c in run.calls() {
                serde_json::from_str::<Value>(&c.args).expect("closed JSON");
            }
            cases += 1;
        }
    }
    assert!(cases > 100);
    // a long value comes in pieces, not at its end
    let text = corpus().into_iter().find(|t| t.contains("/tmp/big.txt")).unwrap();
    let run = feed(&text, ThinkingMode::Chat, &token_sizes(&text), false);
    assert!(run.calls()[0].fragments > 20, "the arguments come in {} fragments", run.calls()[0].fragments);
}

/// T3: a call cut off by the length limit: the call that was named is closed, its arguments are
/// valid JSON, and the parser says it was cut off (the server then reports length), where raw DSML
/// text would give a client nothing to parse.
#[test]
fn t3_a_call_cut_by_the_length_limit_is_closed_json() {
    let text = "Hi.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">a.txt</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"content\" string=\"true\">line \"one\"\nline two";
    for (label, sizes) in chunkings(text) {
        let run = feed(text, ThinkingMode::Chat, &sizes, false);
        let calls = run.calls();
        assert_eq!(calls.len(), 1, "chunking {label}");
        assert!(run.truncated, "chunking {label}");
        let args: Value = serde_json::from_str(&calls[0].args).unwrap_or_else(|e| panic!("{label}: {e}: {}", calls[0].args));
        assert_eq!(args["path"], "a.txt");
        assert_eq!(args["content"], "line \"one\"\nline two");
        assert_eq!(run.content, "Hi.");
    }
    // cut off inside a number, between parameters, and right after the name
    for cut in [
        "Hi.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"w\">\n<｜DSML｜ parameter name=\"n\" string=\"false\">12",
        "Hi.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"w\">\n<｜DSML｜ parameter name=\"n\" string=\"false\">1</｜DSML｜ parameter>\n",
        "Hi.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"w\">\n<｜DSML｜ parameter name=\"s\" string=\"true\">abc<",
    ] {
        let run = feed(cut, ThinkingMode::Chat, &[1], false);
        let calls = run.calls();
        assert_eq!(calls.len(), 1, "{cut:?}");
        serde_json::from_str::<Value>(&calls[0].args).unwrap_or_else(|e| panic!("{e}: {}", calls[0].args));
        assert!(run.truncated);
    }
    // cut off before the call is named: nothing to close, the block stays text, as before
    let early = "Hi.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"wri";
    let run = feed(early, ThinkingMode::Chat, &[1], false);
    assert!(run.calls().is_empty());
    assert!(!run.truncated);
    assert_eq!(run.content, early);
}

/// T4 (the parser's part): a reply sent whole carries each call whole, as it always did: with
/// `hold_calls` the deltas of a call come out together once it is complete, equal to the whole
/// parse for every case; a call that is not complete is the block's raw text, never a closed call.
#[test]
fn t4_a_whole_reply_is_unchanged() {
    for text in corpus() {
        let mode = mode_of(&text);
        let want = parse_message_from_completion_text(&text, mode, Encoding::V41).unwrap();
        for (label, sizes) in chunkings(&text) {
            let run = feed(&text, mode, &sizes, true);
            assert_eq!(run.pairs(), want.tool_calls, "{label}");
            assert_eq!(run.content, want.content, "{label}");
            // the call's events come out in one burst: none is left waiting for later text
            for call_events in run.events.chunk_by(|a, b| a.0 == b.0) {
                assert!(matches!(call_events[0].1, Ev::Start(..)), "a burst starts with a name: {call_events:?}");
            }
        }
    }
    let cut = "Hi.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n<｜DSML｜ parameter name=\"content\" string=\"true\">line two";
    for (label, sizes) in chunkings(cut) {
        let run = feed(cut, ThinkingMode::Chat, &sizes, true);
        assert!(run.calls().is_empty(), "{label}");
        assert!(!run.truncated, "{label}");
        assert_eq!(run.content, cut, "{label}: the raw text, as a whole reply gives it");
    }
}

/// T5: two calls in a reply: indexes 0 and 1 in order, each call's fragments between its own start
/// and end (`Run::calls` fails on any interleaving).
#[test]
fn t5_two_calls_do_not_interleave() {
    let text = block(&[("read", json!({"path": "a", "n": 1})), ("write", json!({"path": "b", "content": long_text()}))]);
    for (label, sizes) in chunkings(&text) {
        let run = feed(&text, ThinkingMode::Chat, &sizes, false);
        let calls = run.calls();
        assert_eq!(calls.iter().map(|c| (c.index, c.name.as_str())).collect::<Vec<_>>(), vec![(0, "read"), (1, "write")], "{label}");
        // the second call is named once the first has ended
        let end0 = run.events.iter().position(|(_, e)| *e == Ev::End(0)).unwrap();
        let start1 = run.events.iter().position(|(_, e)| matches!(e, Ev::Start(1, _))).unwrap();
        assert!(end0 < start1);
    }
}

/// T6: a namespaced call is named as it is written, `web::search`, streamed and whole alike; the
/// server splits it into namespace and name the same way for both.
#[test]
fn t6_a_namespaced_name_is_the_same_streamed() {
    let text = block(&[("web::search", json!({"q": "x"}))]);
    let want = parse_message_from_completion_text(&text, ThinkingMode::Chat, Encoding::V41_1017).unwrap();
    let run = feed(&text, ThinkingMode::Chat, &[1], false);
    assert_eq!(run.pairs(), want.tool_calls);
    assert_eq!(run.calls()[0].name, "web::search");
    assert_eq!(api_tool_name(Encoding::V41_1017, &run.calls()[0].name), (Some("web"), "search"));
}

/// A parameter whose header is malformed does not hold the call open: at its end tag the call is
/// closed, and the text that follows is content. (A reply sent whole passes the block on as text.)
#[test]
fn a_malformed_parameter_header_ends_the_call_at_its_end_tag() {
    let text = "Hi.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"w\">\n<｜DSML｜ parameter name=\"a\" string=\"true\">1</｜DSML｜ parameter>\n<｜DSML｜ parameter nom=\"b\">2</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    let run = feed(text, ThinkingMode::Chat, &[1], false);
    let calls = run.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].args, "{\"a\": \"1\"}");
    let end = run.events.iter().find(|(_, e)| *e == Ev::End(0)).unwrap().0;
    let closing = text[..text.find("</｜DSML｜ invoke").unwrap()].chars().count();
    assert!(end < closing, "the call ended after {end} characters, the invoke closes at {closing}");
    assert!(run.content.starts_with("Hi.") && run.content.contains("nom="));
    let whole = feed(text, ThinkingMode::Chat, &[1], true);
    assert!(whole.calls().is_empty());
    assert!(whole.content.contains("string=\"true\">1<") && whole.content.contains("nom="));
}
