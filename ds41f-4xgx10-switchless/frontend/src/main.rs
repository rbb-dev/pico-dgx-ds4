use dsfront::tokenizer::Tokenizer;
use std::io::BufRead;
use std::time::Instant;

const DEFAULT_TOKENIZER: &str = "/data/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/tokenizer.json";

fn usage() -> ! {
    eprintln!(
        "usage:
  dsfront serve [options]                             the OpenAI-compatible server
      --listen ADDR             127.0.0.1:18890; repeat it for more (loopback or specific addresses;
                                never 0.0.0.0 or ::)
      --engine-socket PATH      /tmp/dsgb10-engine.sock
      --api-key-file PATH       require this bearer key (the file's content, trimmed)
      --model-name NAME         deepseek-v4.1-flash-1017; also served as NAME-ablit, which runs a
                                request on the ablated weights path (NAME and any other `model` run on
                                the plain path)
      --no-ablit-alias          serve NAME only
      --max-tokens N            default and cap for max_tokens: 1048576
      --effort N                reasoning effort budget when a request names none: 75
      --encoding E              the chat encoding: 1017 (DeepSeek-V4.1-Flash-1017's: tool namespaces, the
                                reasoning kept across a turn's tool calls; the default) or v4.1
                                (DeepSeek-V4.1-Flash's encoding.py)
      --thinking                thinking on when a request does not say (the default)
      --no-thinking             thinking off when a request does not say (the later
                                of --thinking and --no-thinking wins); requests can override it
      --temperature T --top-p P sampling defaults: 1.0 and 0.95
      --no-loop-guard           do not stop repetition loops
      --dump-dir DIR            write each request, prompt and ids to DIR, and at its end the output's
                                ids (N.out.json; debugging and bench traces)
      --hold-s N                while the engine is down (the ring lent, or a restart), a request
                                waits this long (900), then is answered 503 with Retry-After: 60
      --admin-socket PATH       commands from service.sh (mode 600): status, hold IDLE_S, release
      --table-file PATH         the session table, saved here while serving and at a stop, and taken back
                                at the next start (mode 600, in a directory of mode 700; it holds hashes and
                                numbers, never a conversation's text), so that the engine's kept sessions are
                                found again
      --no-prefix               say protocol version 4 to the engine: every request carries its whole prompt
                                (the engine's own frames before a request could carry only its suffix)
      --table-save-s N          while serving, save the table every N seconds when it has changed
                                (30; 0 = only at the stop); written whole or not at all, off the
                                request path
      --table-days N            sessions unused for longer are not kept: 7
      --front-table-mib N       the most memory the session table may hold, counted by what its buffers allocate:
                                0 = from --front-mib (the cap / 1.1, less 107 and --front-image-mib; the default),
                                or no limit when --front-mib is 0; the oldest idle sessions go first, beside the
                                engine's position limit
      --front-mib N             the most memory this process may hold as the kernel counts it, read every 10 s:
                                840 by default, the same number as the engine's --front-mib; 0 = not watched; over it, idle sessions go, oldest first (not
                                below half of what the table held when it went over, and not at all while the
                                memory does not fall as they go), and image requests are answered 429 with
                                Retry-After: 5 until it is under
      --front-image-mib N       the most the image requests in flight count for against --front-table-mib: 400 (their rows, or
                                with --vision engine the pixels they hold, counted as the media worker's frames arrive;
                                one request over it is refused, 413)
      --node-stats HOST:PORT,... each rank's sampler, in rank order (docs/dashboard.md): the dashboard at
                                /dashboard polls them while it is open; without them its NODE HEALTH is empty
      --dashboard-host NAME     a host name the dashboard also answers (repeat it), e.g. a proxy's; it always
                                answers localhost and the listen addresses, and refuses every other Host
      --fan-tool PATH           the fan tool (dgx-fan-fleet, docs/dashboard.md): the dashboard then has a COOLING AND CLOCK box
                                that sets the Sparks' fan profile and GPU clock limit; without it the box is absent
      --vision-addr HOST:PORT   with --vision container: the image encoder (tools/vision) that turns images into
                                embedding rows (docs/vision.md); without it, under that flag, images are refused
      --vision engine|container where images are encoded: container (the rollback) asks the encoder at --vision-addr for
                                embedding rows and sends them in the request; engine asks the media worker at --media-addr
                                for each image's padded pixels and sends those, and the engine encodes them (docs/vision.md)
      --media-addr HOST:PORT    the media worker (tools/vision/media_worker.py), which reads and pads
                                the images and decodes video, and holds no model; required with an explicit --vision engine; without --media-addr images are refused
      --max-body-mb N           the largest request body taken, in MiB: 64; a larger one is answered
                                413. Raise it for requests that carry many large images or video
                                inline; a body costs several times its size in memory while its
                                request is prepared (docs/vision.md)
      --max-image-rows N        the most image tokens (embedding rows) the images of one request may
                                take together: 40000; more is answered 400. Memory on rank 0's box
                                follows rows, not the number of images (docs/vision.md); at most 110000
      --max-image-rows-in-flight N
                                the most image tokens the image requests being answered may take
                                together: 80000 (two full requests). A request with images that does
                                not fit is answered 429 with Retry-After: 5 at once; text requests are
                                never counted. Not below --max-image-rows
  dsfront admin --socket PATH <command>               one command to a server's admin socket; prints
                                                      the answer (docs/engine.md, \"Launch and stop\")
  dsfront mock-engine [options]                       a rank-0 stand-in (docs/protocol.md)
      --socket PATH             /tmp/dsgb10-engine.sock
      --delay-ms N              per generated token: 5
      --prefill-tps N           simulated prefill tokens/s: 20000
      --mock-version N          the newest protocol version it speaks (4 or 5): 5
      --max-context N           1048576
  dsfront mock-node-stats --port P --rank R           a rank's sampler stand-in on 127.0.0.1:P (docs/dashboard.md)
  dsfront tokcheck <cases.jsonl>                      compare against reference ids ({{\"text\",\"ids\"}} per line)
  dsfront enccheck <cases.jsonl> [--encoding E]       compare the chat encoding with encoding.py's (tools/gen_encoding_cases.py)
  dsfront tokenize <file>                             print ids of a file's text
  dsfront tokbench <file>                             tokenisation throughput
  every command that tokenises takes --tokenizer PATH ({DEFAULT_TOKENIZER})"
    );
    std::process::exit(2)
}

fn parse_opt<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    match opt(args, name) {
        None => default,
        Some(v) => v.parse().unwrap_or_else(|_| {
            eprintln!("bad value for {name}: {v}");
            std::process::exit(2)
        }),
    }
}

/// The `--encoding` option, else `default` (serve: the served one; enccheck: v4.1, the encoding its case files without a flag are made with).
fn encoding_opt(args: &[String], default: dsfront::encoding::Encoding) -> dsfront::encoding::Encoding {
    opt(args, "--encoding").map_or(default, |v| {
        dsfront::encoding::Encoding::parse(&v).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2)
        })
    })
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread().enable_all().build().expect("tokio runtime")
}

fn serve(args: &[String]) {
    use dsfront::server::{router, App, ServerConfig};
    let mut listen: Vec<std::net::SocketAddr> = opts(args, "--listen")
        .iter()
        .map(|v| {
            v.parse().unwrap_or_else(|_| {
                eprintln!("bad value for --listen: {v}");
                std::process::exit(2)
            })
        })
        .collect();
    if listen.is_empty() {
        listen.push("127.0.0.1:18890".parse().unwrap());
    }
    for a in &listen {
        if unspecified(a.ip()) {
            eprintln!("refusing to listen on {a}: bind to loopback or specific addresses");
            std::process::exit(2);
        }
    }
    let api_key = opt(args, "--api-key-file").map(|p| {
        let k = std::fs::read_to_string(&p).unwrap_or_else(|e| {
            eprintln!("{p}: {e}");
            std::process::exit(1)
        });
        let k = k.trim().to_string();
        if k.is_empty() {
            eprintln!("{p} is empty");
            std::process::exit(1);
        }
        k
    });
    if api_key.is_none() {
        eprintln!("warning: no --api-key-file, requests are not authenticated");
    }
    let vision = vision_opt(args).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let mut dashboard_hosts: Vec<String> = vec!["localhost".into()];
    dashboard_hosts.extend(listen.iter().map(|a| a.ip().to_string()));
    dashboard_hosts.extend(opts(args, "--dashboard-host").into_iter().map(|h| h.to_ascii_lowercase()));
    let mut cfg = ServerConfig {
        dashboard_hosts,
        model_name: opt(args, "--model-name").unwrap_or_else(|| dsfront::served::MODEL_NAME.into()),
        ablit_alias: !args.iter().any(|a| a == "--no-ablit-alias"),
        api_key,
        max_tokens_cap: parse_opt(args, "--max-tokens", dsfront::served::MAX_TOKENS),
        default_effort: parse_opt(args, "--effort", dsfront::served::EFFORT).clamp(1, 100),
        encoding: encoding_opt(args, dsfront::served::ENCODING),
        default_thinking: dsfront::served::default_thinking(args),
        default_temperature: parse_opt(args, "--temperature", dsfront::served::TEMPERATURE),
        default_top_p: parse_opt(args, "--top-p", dsfront::served::TOP_P),
        loop_guard: if args.iter().any(|a| a == "--no-loop-guard") { None } else { Some(Default::default()) },
        tokenize_threads: parse_opt(args, "--tokenize-threads", 8usize).max(1),
        dump_dir: opt(args, "--dump-dir").map(Into::into),
        hold: std::time::Duration::from_secs(parse_opt(args, "--hold-s", dsfront::served::HOLD_S)),
        vision_addr: opt(args, "--vision-addr"),
        vision,
        media_addr: opt(args, "--media-addr"),
        max_image_rows: max_image_rows(args),
        max_body_bytes: max_body_bytes(args),
        max_image_rows_in_flight: max_image_rows_in_flight(args),
        fan_tool: opt(args, "--fan-tool").map(Into::into),
        front_table_mib: parse_opt(args, "--front-table-mib", dsfront::server::DEFAULT_FRONT_TABLE_MIB),
        front_mib: parse_opt(args, "--front-mib", dsfront::served::FRONT_MIB),
        front_image_mib: parse_opt(args, "--front-image-mib", dsfront::server::DEFAULT_FRONT_IMAGE_MIB),
    };
    // the one memory cap bounds the table too, unless the table's own bound is given
    if cfg.front_table_mib == 0 {
        cfg.front_table_mib = dsfront::server::table_mib_from_front(cfg.front_mib, cfg.front_image_mib);
    }
    let node_stats: Vec<String> = opt(args, "--node-stats").map_or(Vec::new(), |v| v.split(',').map(|a| a.trim().to_string()).filter(|a| !a.is_empty()).collect());
    let admin = opt(args, "--admin-socket");
    let table = opt(args, "--table-file").map(std::path::PathBuf::from);
    let table_age_s = (parse_opt(args, "--table-days", 7f64) * 86400.0) as u64;
    let table_save_s = parse_opt(args, "--table-save-s", 30u64);
    dsfront::session::cap_heap_arenas(2);
    let tok = std::sync::Arc::new(load_tokenizer(args));
    let socket = opt(args, "--engine-socket").unwrap_or_else(|| dsfront::protocol::DEFAULT_SOCKET.into());
    let prefix = !args.iter().any(|a| a == "--no-prefix");
    runtime().block_on(async move {
        let engine = dsfront::engine::EngineClient::start_with(socket.into(), prefix);
        let app = App::new(cfg, tok, engine);
        dsfront::dashboard::spawn(app.clone(), node_stats);
        if let Some(t) = &table {
            eprintln!("{}", app.load_sessions(t, table_age_s));
        }
        let saver = app.clone();
        if let Some(t) = &table {
            app.spawn_table_saver(t.clone(), table_age_s, std::time::Duration::from_secs(table_save_s));
        }
        app.spawn_front_governor(dsfront::server::FRONT_WATCH_EVERY);
        app.spawn_front_status(std::time::Duration::from_secs(60));
        if let Some(path) = admin {
            let (gate, engine) = (app.gate.clone(), app.engine.clone());
            tokio::spawn(async move {
                if let Err(e) = dsfront::gate::admin(gate, engine, path.clone().into()).await {
                    eprintln!("admin socket {path}: {e}");
                    std::process::exit(1);
                }
            });
        }
        // every address bound before any is served: one that cannot be bound stops the start
        let mut listeners = Vec::new();
        for a in &listen {
            listeners.push(tokio::net::TcpListener::bind(a).await.unwrap_or_else(|e| {
                eprintln!("cannot listen on {a}: {e}");
                std::process::exit(1)
            }));
        }
        let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
        let gate = app.gate.clone();
        tokio::spawn(async move {
            let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("signal");
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = term.recv() => {}
            }
            eprintln!("shutting down");
            gate.close(); // requests waiting at the gate are answered now, not at the hold limit
            let _ = stop_tx.send(true);
        });
        let routes = router(app);
        let mut servers = tokio::task::JoinSet::new();
        for (a, l) in listen.iter().zip(listeners) {
            eprintln!("serving on http://{a}/v1");
            let mut stop = stop_rx.clone();
            let shutdown = async move {
                let _ = stop.wait_for(|s| *s).await;
            };
            servers.spawn(std::future::IntoFuture::into_future(axum::serve(l, routes.clone().into_make_service_with_connect_info::<std::net::SocketAddr>()).with_graceful_shutdown(shutdown)));
        }
        while let Some(r) = servers.join_next().await {
            r.expect("server task").expect("server");
        }
        if let Some(t) = &table {
            eprintln!("{}", saver.save_sessions(t, table_age_s));
        }
    });
}

/// 0.0.0.0, ::, and :: with 0.0.0.0 mapped into it: every address of the box
fn unspecified(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => v4.is_unspecified(),
        std::net::IpAddr::V6(v6) => v6.is_unspecified() || v6.to_ipv4_mapped().is_some_and(|v4| v4.is_unspecified()),
    }
}

/// One command to a server's admin socket (gate.rs); prints the one-line answer. Exit 2 when there
/// is no answer.
fn admin(args: &[String]) {
    use std::io::{Read, Write};
    let sock = opt(args, "--socket").unwrap_or_else(|| usage());
    let mut words = Vec::new();
    let mut i = 2;
    while i < args.len() {
        if args[i] == "--socket" {
            i += 2;
            continue;
        }
        words.push(args[i].as_str());
        i += 1;
    }
    if words.is_empty() {
        usage();
    }
    let fail = |e: std::io::Error| -> ! {
        eprintln!("{sock}: {e}");
        std::process::exit(2)
    };
    let mut s = std::os::unix::net::UnixStream::connect(&sock).unwrap_or_else(|e| fail(e));
    s.set_read_timeout(Some(std::time::Duration::from_secs(10))).unwrap_or_else(|e| fail(e));
    writeln!(s, "{}", words.join(" ")).unwrap_or_else(|e| fail(e));
    let mut reply = String::new();
    s.read_to_string(&mut reply).unwrap_or_else(|e| fail(e));
    if reply.trim().is_empty() {
        eprintln!("{sock}: no answer");
        std::process::exit(2);
    }
    print!("{reply}");
}

fn mock_engine(args: &[String]) {
    let cfg = dsfront::mock::MockConfig {
        socket: opt(args, "--socket").unwrap_or_else(|| dsfront::protocol::DEFAULT_SOCKET.into()).into(),
        tok: std::sync::Arc::new(load_tokenizer(args)),
        delay: std::time::Duration::from_millis(parse_opt(args, "--delay-ms", 5u64)),
        prefill_tok_per_s: parse_opt(args, "--prefill-tps", 20000f64),
        max_context: parse_opt(args, "--max-context", 1048576u32),
        version: parse_opt(args, "--mock-version", dsfront::protocol::VERSION_PREFIX),
        seen: None,
    };
    if let Err(e) = runtime().block_on(dsfront::mock::run(cfg)) {
        eprintln!("mock engine: {e}");
        std::process::exit(1);
    }
}

fn mock_node_stats(args: &[String]) {
    let (port, rank) = (parse_opt(args, "--port", 9760u16), parse_opt(args, "--rank", 0u32));
    if let Err(e) = runtime().block_on(dsfront::dashboard::mock_node_stats(port, rank)) {
        eprintln!("mock node sampler: {e}");
        std::process::exit(1);
    }
}

fn opt(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

/// Every value of a repeatable option.
fn opts(args: &[String], name: &str) -> Vec<String> {
    args.iter().enumerate().filter(|(_, a)| *a == name).filter_map(|(i, _)| args.get(i + 1).cloned()).collect()
}

/// --max-body-mb: the body limit in bytes (64 MiB unless raised).
fn max_body_bytes(args: &[String]) -> usize {
    let default_mb = dsfront::server::DEFAULT_MAX_BODY_BYTES >> 20;
    let mb = parse_opt(args, "--max-body-mb", default_mb);
    if !(1..=4096).contains(&mb) {
        eprintln!("bad value for --max-body-mb: {mb} (1 to 4096)");
        std::process::exit(2);
    }
    mb << 20
}

/// --vision: engine (the default: the engine encodes the media worker's pixels) or container (the rollback: the encoder at
/// --vision-addr answers rows). An explicit engine needs --media-addr; without it and without --vision, images are refused.
fn vision_opt(args: &[String]) -> Result<dsfront::vision::VisionKind, String> {
    use dsfront::vision::VisionKind;
    let kind = match opt(args, "--vision") {
        None => VisionKind::Engine,
        Some(v) => VisionKind::parse(&v).ok_or_else(|| format!("bad value for --vision: {v} (container or engine)"))?,
    };
    if kind == VisionKind::Engine && opt(args, "--vision").is_some() && opt(args, "--media-addr").is_none() {
        return Err("--vision engine needs --media-addr HOST:PORT (the media worker)".into());
    }
    Ok(kind)
}

/// --max-image-rows: 1 to what the engine's frame has room for (100 images of 1100 rows).
fn max_image_rows(args: &[String]) -> usize {
    use dsfront::vision::{DEFAULT_MAX_IMAGE_ROWS, MAX_TOTAL_IMAGE_ROWS};
    let n = parse_opt(args, "--max-image-rows", DEFAULT_MAX_IMAGE_ROWS);
    if !(1..=MAX_TOTAL_IMAGE_ROWS).contains(&n) {
        eprintln!("bad value for --max-image-rows: {n} (1 to {MAX_TOTAL_IMAGE_ROWS})");
        std::process::exit(2);
    }
    n
}

/// --max-image-rows-in-flight: from --max-image-rows (below it a full request could never run) up.
fn max_image_rows_in_flight(args: &[String]) -> usize {
    let one = max_image_rows(args);
    let n = parse_opt(args, "--max-image-rows-in-flight", dsfront::vision::DEFAULT_MAX_IMAGE_ROWS_IN_FLIGHT.max(one));
    if n < one {
        eprintln!("bad value for --max-image-rows-in-flight: {n} (not below --max-image-rows, {one})");
        std::process::exit(2);
    }
    n
}

#[cfg(test)]
mod limit_tests {
    use super::{max_body_bytes, max_image_rows, max_image_rows_in_flight, vision_opt};

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn request_limits() {
        assert_eq!(max_body_bytes(&args("dsfront serve")), 64 << 20, "the body limit is 64 MiB unless raised");
        assert_eq!(max_body_bytes(&args("dsfront serve --max-body-mb 256")), 256 << 20);
        assert_eq!(max_image_rows(&args("dsfront serve")), 40_000);
        assert_eq!(max_image_rows(&args("dsfront serve --max-image-rows 99400")), 99_400);
        assert_eq!(max_image_rows_in_flight(&args("dsfront serve")), 80_000, "two full requests unless set");
        assert_eq!(max_image_rows_in_flight(&args("dsfront serve --max-image-rows-in-flight 40000")), 40_000);
        assert_eq!(max_image_rows_in_flight(&args("dsfront serve --max-image-rows-in-flight 200000")), 200_000);
        // a raised request cap raises the default with it: one full request always fits
        assert_eq!(max_image_rows_in_flight(&args("dsfront serve --max-image-rows 99400")), 99_400);
    }

    #[test]
    fn the_vision_flag() {
        use dsfront::vision::VisionKind;
        assert_eq!(vision_opt(&args("dsfront serve")), Ok(VisionKind::Engine), "the engine's encoder unless the rollback is asked");
        assert_eq!(vision_opt(&args("dsfront serve --media-addr 127.0.0.1:18951")), Ok(VisionKind::Engine));
        assert_eq!(vision_opt(&args("dsfront serve --vision container")), Ok(VisionKind::Container));
        assert_eq!(vision_opt(&args("dsfront serve --vision engine --media-addr 127.0.0.1:18951")), Ok(VisionKind::Engine));
        assert!(vision_opt(&args("dsfront serve --vision engine")).unwrap_err().contains("--media-addr"), "the engine's images come from the media worker");
        assert!(vision_opt(&args("dsfront serve --vision gpu")).unwrap_err().contains("container or engine"));
    }
}

fn load_tokenizer(args: &[String]) -> Tokenizer {
    let path = opt(args, "--tokenizer").unwrap_or_else(|| DEFAULT_TOKENIZER.to_string());
    let t0 = Instant::now();
    let tok = Tokenizer::from_file(&path).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1)
    });
    eprintln!("tokenizer loaded in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
    tok
}

fn tokcheck(args: &[String]) {
    let path = args.get(2).unwrap_or_else(|| usage());
    let tok = load_tokenizer(args);
    let f = std::fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(1)
    });
    let (mut n, mut bad, mut bad_decode, mut bytes, mut toks) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut secs = 0f64;
    for line in std::io::BufReader::new(f).lines() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line).expect("bad json line");
        let text = v["text"].as_str().expect("text");
        let want: Vec<u32> = v["ids"].as_array().expect("ids").iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let t0 = Instant::now();
        let got = tok.encode(text);
        secs += t0.elapsed().as_secs_f64();
        n += 1;
        bytes += text.len();
        toks += want.len();
        if got != want {
            bad += 1;
            if bad <= 10 {
                let k = got.iter().zip(want.iter()).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
                let lo = k.saturating_sub(3);
                eprintln!(
                    "MISMATCH case {n} ({} bytes): first diff at token {k}\n  text: {:?}\n  want[{lo}..]: {:?}\n  got [{lo}..]: {:?}",
                    text.len(),
                    text.chars().take(200).collect::<String>(),
                    &want[lo..(k + 4).min(want.len())],
                    &got[lo..(k + 4).min(got.len())]
                );
            }
        }
        if tok.decode(&want) != text {
            bad_decode += 1;
            if bad_decode <= 5 {
                eprintln!("DECODE MISMATCH case {n}: {:?}", text.chars().take(120).collect::<String>());
            }
        }
    }
    println!(
        "{n} cases, {bad} encode mismatches, {bad_decode} decode mismatches; {toks} tokens, {:.1} MB in {:.2} s = {:.2} M tok/s",
        bytes as f64 / 1e6,
        secs,
        toks as f64 / secs / 1e6
    );
    if bad > 0 || bad_decode > 0 {
        std::process::exit(1);
    }
}

/// Compare encoding.rs with encoding.py's output (tools/gen_encoding_cases.py).
fn enccheck(args: &[String]) {
    use dsfront::encoding::{encode_messages, parse_message_from_completion_text, Delta, EncodeOptions, StreamParser, ThinkingMode};
    use serde_json::Value;
    let path = args.get(2).unwrap_or_else(|| usage());
    let f = std::fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(1)
    });
    let enc = encoding_opt(args, dsfront::encoding::Encoding::V41);
    let (mut n_enc, mut n_parse, mut n_stream, mut bad) = (0usize, 0usize, 0usize, 0usize);
    let mut rng = 0x9E3779B97F4A7C15u64;
    let mut report = |name: &str, what: String| {
        bad += 1;
        if bad <= 12 {
            eprintln!("MISMATCH {name}: {what}");
        }
    };
    let t0 = Instant::now();
    for line in std::io::BufReader::new(f).lines() {
        let line = line.unwrap();
        let c: Value = serde_json::from_str(&line).expect("bad json line");
        let name = c["name"].as_str().unwrap_or("?").to_string();
        let mode = ThinkingMode::parse(c["thinking_mode"].as_str().unwrap()).unwrap();
        match c["kind"].as_str() {
            Some("encode") => {
                n_enc += 1;
                let msgs = c["messages"].as_array().cloned().unwrap_or_default();
                let ctx = c["context"].as_array().cloned();
                let mut o = EncodeOptions::new(mode);
                o.context = ctx.as_deref();
                o.drop_thinking = c["drop_thinking"].as_bool().unwrap();
                o.add_default_bos_token = c["add_bos"].as_bool().unwrap();
                o.reasoning_effort = c.get("reasoning_effort");
                o.encoding = enc;
                let got = encode_messages(&msgs, &o);
                match (got, c.get("prompt")) {
                    (Ok((p, imgs)), Some(Value::String(want))) => {
                        if &p != want {
                            let k = p.chars().zip(want.chars()).position(|(a, b)| a != b).unwrap_or(p.len().min(want.len()));
                            let at = |s: &str| s.chars().skip(k.saturating_sub(20)).take(60).collect::<String>();
                            report(&name, format!("prompt differs at char {k}:\n  want {:?}\n  got  {:?}", at(want), at(&p)));
                        } else if Value::Array(imgs) != c["images"] {
                            report(&name, "image records differ".into());
                        }
                    }
                    (Err(e), Some(Value::String(_))) => report(&name, format!("port raised {e:?}, encoding.py did not")),
                    (Ok((p, _)), None) => {
                        report(&name, format!("encoding.py raised {}, port gave {:?}", c["error"], p.chars().take(80).collect::<String>()))
                    }
                    _ => {}
                }
            }
            Some("parse") => {
                n_parse += 1;
                let text = c["text"].as_str().unwrap();
                let got = parse_message_from_completion_text(text, mode, enc);
                let want = c.get("result");
                match (&got, want) {
                    (Ok(m), Some(w)) => {
                        let calls: Vec<(String, String)> = w["tool_calls"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|t| {
                                let name = t["function"]["name"].as_str().unwrap();
                                // 1017 reports a namespace beside the function; the port keeps it in the name
                                let name = match t.get("namespace").and_then(Value::as_str) {
                                    Some(ns) => format!("{ns}::{name}"),
                                    None => name.to_string(),
                                };
                                (name, t["function"]["arguments"].as_str().unwrap().to_string())
                            })
                            .collect();
                        if m.content != w["content"].as_str().unwrap()
                            || m.reasoning_content != w["reasoning_content"].as_str().unwrap()
                            || m.tool_calls != calls
                        {
                            report(&name, format!("parse differs: got {m:?}\n  want {w}"));
                        }
                    }
                    (Err(e), Some(_)) => report(&name, format!("port raised {e:?}, encoding.py did not")),
                    (Ok(m), None) => report(&name, format!("encoding.py raised {}, port gave {m:?}", c["error"])),
                    _ => {}
                }
                // The streaming parser, fed in random pieces, must agree wherever the reference parses.
                let chars: Vec<char> = text.chars().collect();
                let mut p = StreamParser::new(mode);
                let mut out = Vec::new();
                let mut i = 0;
                while i < chars.len() {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    let k = 1 + (rng % 9) as usize;
                    let piece: String = chars[i..(i + k).min(chars.len())].iter().collect();
                    p.push(&piece, &mut out);
                    i += k;
                }
                p.finish(&mut out);
                if let Ok(m) = got {
                    n_stream += 1;
                    let (mut r, mut ct, mut calls) = (String::new(), String::new(), Vec::new());
                    for d in out {
                        match d {
                            Delta::Reasoning(s) => r.push_str(&s),
                            Delta::Content(s) => ct.push_str(&s),
                            Delta::ToolCall(_, n, a) => calls.push((n, a)),
                        }
                    }
                    if r != m.reasoning_content || ct != m.content || calls != m.tool_calls {
                        report(&name, format!("stream differs: reasoning {r:?} content {ct:?} calls {calls:?}\n  want {m:?}"));
                    }
                }
            }
            _ => panic!("unknown case kind"),
        }
    }
    println!(
        "{n_enc} encode cases, {n_parse} parse cases ({n_stream} also streamed): {bad} mismatches in {:.2} s",
        t0.elapsed().as_secs_f64()
    );
    if bad > 0 {
        std::process::exit(1);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("tokcheck") => tokcheck(&args),
        Some("enccheck") => enccheck(&args),
        Some("serve") => serve(&args),
        Some("mock-engine") => mock_engine(&args),
        Some("mock-node-stats") => mock_node_stats(&args),
        Some("admin") => admin(&args),
        Some("tokenize") => {
            let path = args.get(2).unwrap_or_else(|| usage());
            let tok = load_tokenizer(&args);
            let text = std::fs::read_to_string(path).expect("read");
            println!("{:?}", tok.encode(&text));
        }
        Some("tokbench") => {
            let path = args.get(2).unwrap_or_else(|| usage());
            let tok = load_tokenizer(&args);
            let text = std::fs::read_to_string(path).expect("read");
            let reference = tok.encode(&text);
            for threads in [1usize, 4, 16] {
                let t0 = Instant::now();
                let ids = tok.encode_parallel(&text, threads);
                let s = t0.elapsed().as_secs_f64();
                println!(
                    "threads {threads}: {} tokens from {:.1} MB in {:.3} s = {:.2} M tok/s{}",
                    ids.len(),
                    text.len() as f64 / 1e6,
                    s,
                    ids.len() as f64 / s / 1e6,
                    if ids == reference { "" } else { "  (DIFFERS from the single-pass encoding)" }
                );
            }
        }
        _ => usage(),
    }
}

#[cfg(test)]
mod tests {
    use super::{encoding_opt, opt, parse_opt};
    use dsfront::served;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    /// `dsfront serve` with none of the served settings given reads every one of them as the served value, and each can
    /// be given another value.
    #[test]
    fn serve_without_settings_is_the_served_configuration() {
        let a = args("dsfront serve");
        assert_eq!(opt(&a, "--model-name").unwrap_or_else(|| served::MODEL_NAME.into()), "deepseek-v4.1-flash-1017");
        assert_eq!(parse_opt(&a, "--max-tokens", served::MAX_TOKENS), 1_048_576u32);
        assert_eq!(parse_opt(&a, "--effort", served::EFFORT), 75i64);
        assert_eq!(parse_opt(&a, "--temperature", served::TEMPERATURE), 1.0f32);
        assert_eq!(parse_opt(&a, "--top-p", served::TOP_P), 0.95f32);
        assert_eq!(parse_opt(&a, "--front-mib", served::FRONT_MIB), 840usize);
        assert_eq!(encoding_opt(&a, served::ENCODING), dsfront::encoding::Encoding::parse("1017").unwrap());
        assert!(served::default_thinking(&a));
        let b = args("dsfront serve --max-tokens 32768 --effort 50 --encoding v4.1 --no-thinking");
        assert_eq!(parse_opt(&b, "--max-tokens", served::MAX_TOKENS), 32768u32);
        assert_eq!(parse_opt(&b, "--effort", served::EFFORT), 50i64);
        assert_eq!(encoding_opt(&b, served::ENCODING), dsfront::encoding::Encoding::V41);
        assert!(!served::default_thinking(&b));
    }

    /// enccheck keeps v4.1 as its default: its case files without a flag are made with that encoding.
    #[test]
    fn enccheck_default_is_v41() {
        assert_eq!(encoding_opt(&args("dsfront enccheck f"), dsfront::encoding::Encoding::V41), dsfront::encoding::Encoding::V41);
    }
}
