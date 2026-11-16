#!/usr/bin/env python3
"""End to end checks of the dashboard and the -ablit name: dsfront against its mock engine, four mock node samplers and a
fake fan tool, all started here (stdlib only; no engine, no GPU).

    dashboard_smoke.py [--bin target/release/dsfront] [--tokenizer tokenizer.json]

Without --tokenizer a byte-level one is written by make_test_tokenizer.py beside this script. Covers: both served
names in /v1/models, the `[ablit] ` prefix arriving only on the ablated requests (the -ablit name; never on the plain or
any other name), the echoed model name, sessions kept apart by weights path, the dashboard's sign-in (the API key on
every data route; the page loads without it), every dashboard route with the keys the page reads, the page itself, no
switch route, a request's states
through the requests route (queued, prefill with the prompt pass, decode), STATS asked only while the dashboard is
watched, the series, node health and the memory ledger, and a sanity timing of a stream with and without the page
polling.
"""
import http.client
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]


def opt(name, default):
    return args[args.index(name) + 1] if name in args else default


BIN = opt("--bin", os.path.join(HERE, "..", "target", "release", "dsfront"))
if not os.path.exists(BIN):
    BIN = os.path.join(HERE, "..", "target", "debug", "dsfront")
TMP = tempfile.mkdtemp(prefix="dsfront-smoke-")
os.chmod(TMP, 0o700)
TOK = opt("--tokenizer", None)
if not TOK:
    TOK = os.path.join(TMP, "tokenizer.json")
    subprocess.run([sys.executable, os.path.join(HERE, "make_test_tokenizer.py"), TOK], check=True)
NAME = "smoke-model"
KEY = "smoke-key-%d" % os.getpid()
FAILS = []
PROCS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def start(argv, log):
    f = open(os.path.join(TMP, log), "w")
    p = subprocess.Popen([BIN] + argv, stdout=f, stderr=f)
    PROCS.append(p)
    return p


PORT = free_port()
SOCK = os.path.join(TMP, "engine.sock")
NODE_PORTS = [free_port() for _ in range(4)]


def http_(method, path, body=None, headers=None, timeout=60):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    h = dict(headers or {})
    h.setdefault("Authorization", "Bearer " + KEY)  # the key every client sends; None leaves it out
    h = {k: v for k, v in h.items() if v is not None}
    data = None
    if body is not None:
        data = body if isinstance(body, (bytes, str)) else json.dumps(body)
        h.setdefault("Content-Type", "application/json")
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    raw = r.read()
    c.close()
    return r.status, raw, r


def poll_get(path):
    """A background poll: a connection the host refuses under load (ephemeral ports) is not a finding."""
    try:
        http_("GET", path)
    except OSError:
        pass


def get_json(path):
    st, raw, _ = http_("GET", path)
    return st, json.loads(raw) if raw else None


def chat(model, content, user=None, extra=None):
    body = {"model": model, "messages": [{"role": "user", "content": content}]}
    if user:
        body["user"] = user
    body.update(extra or {})
    st, raw, _ = http_("POST", "/v1/chat/completions", body)
    return st, json.loads(raw)


def text_of(d):
    return d["choices"][0]["message"]["content"]


def stream_chat(model, content, user=None, gaps=None):
    """Content of a streamed answer; the arrival gaps of its chunks go to `gaps`."""
    body = {"model": model, "stream": True, "messages": [{"role": "user", "content": content}]}
    if user:
        body["user"] = user
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=120)
    c.request("POST", "/v1/chat/completions", body=json.dumps(body), headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    r = c.getresponse()
    out, last = [], None
    for line in r:
        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
            continue
        d = json.loads(line[6:])
        if d.get("choices") and d["choices"][0]["delta"].get("content"):
            now = time.monotonic()
            if last is not None and gaps is not None:
                gaps.append(now - last)
            last = now
            out.append(d["choices"][0]["delta"]["content"])
    c.close()
    return "".join(out)


def finish():
    for p in PROCS:
        p.terminate()
    for p in PROCS:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED; logs in {TMP}")
        for f in FAILS:
            print("  " + f)
        sys.exit(1)
    shutil.rmtree(TMP, ignore_errors=True)
    print("\nall checks passed")


def wait_for(cond, timeout, step=0.1):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            v = cond()
        except OSError:
            v = None
        if v:
            return v
        time.sleep(step)
    return None


def main():
    start(["mock-engine", "--socket", SOCK, "--tokenizer", TOK, "--delay-ms", "5", "--prefill-tps", "4000"], "mock.log")
    for r, p in enumerate(NODE_PORTS):
        start(["mock-node-stats", "--port", str(p), "--rank", str(r)], f"node{r}.log")
    time.sleep(0.5)
    table = os.path.join(TMP, "front.table")
    nodes = ",".join(f"127.0.0.1:{p}" for p in NODE_PORTS)
    keyfile = os.path.join(TMP, "api-key")
    with open(keyfile, "w") as f:
        f.write(KEY + "\n")
    start(["serve", "--engine-socket", SOCK, "--tokenizer", TOK, "--listen", f"127.0.0.1:{PORT}", "--model-name", NAME,
           "--table-file", table, "--node-stats", nodes, "--api-key-file", keyfile], "serve.log")
    up = wait_for(lambda: http_("GET", "/health")[0] == 200, 20)
    check(bool(up), "dsfront serves and is connected to the mock engine")
    if not up:
        return

    # ---- the names
    st, models = get_json("/v1/models")
    ids = [m["id"] for m in models["data"]]
    check(st == 200 and ids == [NAME, NAME + "-ablit"], f"/v1/models lists both names: {ids}")
    msg = "hello there, this is a conversation long enough to be matched by its text alone, over sixty-four bytes."
    st, a = chat(NAME, msg)
    check(st == 200 and a["model"] == NAME and text_of(a) == "Mock reply to: " + msg, "the plain name: no prefix, model echoed")
    st, b = chat(NAME + "-ablit", msg)
    check(st == 200 and b["model"] == NAME + "-ablit" and text_of(b) == "[ablit] Mock reply to: " + msg, "the -ablit name: prefix, model echoed")
    st, o = chat("something-else", msg)
    check(o["model"] == NAME and text_of(o) == "Mock reply to: " + msg, "any other model: the plain name, on the default path")
    check(stream_chat(NAME + "-ablit", "stream me") == "[ablit] Mock reply to: stream me", "a streamed ablit request carries the prefix")
    check(stream_chat(NAME, "stream me") == "Mock reply to: stream me", "a streamed plain request does not")

    # ---- sessions are kept apart by weights path: the mock engine refuses a session of the other path, the
    # front-end must never send one
    base = "A shared opening for the sessions test, long enough to be matched by text, well over sixty-four bytes: "
    st, p1 = chat(NAME, base + "first")
    st, c1 = chat(NAME, base + "first")
    check(st == 200 and c1["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, "a plain request continues a plain session")
    for m in (NAME + "-ablit", NAME, NAME + "-ablit"):
        st, r = chat(m, base + "first")
        check(st == 200 and "error" not in r, f"the same text on {m} is served (no session crosses the paths)")
    st, r = chat(NAME + "-ablit", base + "first")
    check(st == 200 and r["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, "an ablit request continues an ablit session")

    # ---- the page and the routes. Nothing asked the engine for STATS before this first dashboard request.
    time.sleep(2.2)
    st, raw, resp = http_("GET", "/dashboard")
    html = raw.decode()
    check(st == 200 and resp.getheader("content-type", "").startswith("text/html") and "DSFRONT" in html and "/api/v1/" in html, "GET /dashboard is the page")
    for label in ("'Total'", "'Users generating'", "'Time per token'", "'Speed now'", "'This prompt'", "'Median'",
                  "'Slowest 1 %'", "'CONVERSATION MEMORY'", "'SSD CACHE'", "'Save points'", "'Found on SSD'",
                  "'Tokens not recomputed'", "'Average wait'", "const SPARKS = 4;"):
        check(label in html, f"the page has the label {label}")
    # NODE HEALTH's table: one row a Spark, the units in the column heads
    for label in ("'Hot', '°C'", "'Wear', '%'", "'Clock', 'MHz'", "'Free / total', 'GiB'", "class: 'ntab'"):
        check(label in html, f"the page has {label}")
    # the page's script parses (a stray quote breaks every tile); needs node where the smoke runs
    js = html.split("<script>", 1)[1].split("</script>", 1)[0] if "<script>" in html else ""
    if shutil.which("node"):
        jp = os.path.join(TMP, "page.js")
        with open(jp, "w") as f:
            f.write(js)
        r = subprocess.run(["node", "--check", jp], capture_output=True, text=True)
        check(bool(js) and r.returncode == 0, f"the page's script parses {r.stderr.strip()[:200]}")
    else:
        print("  SKIP the page's script parse: node is not installed here")
    check("const filling = x.state === 'prefill'" in html and "firstWaiting" not in html,
          "LINES takes each request's own state (two long prompts can be read at once)")
    for label in ("'All streams'", "'Token gap'", "'p99', 'p99'", "'SESSIONS'", "'SSD TIER'", "'Disk hits'"):
        check(label not in html, f"the old label {label} is gone")
    st, raw, _ = http_("GET", "/dashboard/")
    check(st == 200 and raw.decode() == html, "GET /dashboard/ is the same page")
    # ---- the sign-in: the page loads without the key and asks for it; every data route needs it
    st, raw, _ = http_("GET", "/dashboard", headers={"Authorization": None})
    check(st == 200 and "API key" in raw.decode(), "the page loads without the key and has the sign-in")
    for route in ("status", "requests", "ledger", "series?name=decode_tps&seconds=900&step=10"):
        check(http_("GET", "/dashboard/api/v1/" + route, headers={"Authorization": None})[0] == 401, f"GET {route} without the key is 401")
    check(http_("GET", "/dashboard/api/v1/status", headers={"Authorization": "Bearer wrong-" + KEY})[0] == 401, "a wrong key is 401")
    check(http_("GET", "/v1/models", headers={"Authorization": None})[0] == 401, "/v1 needs the same key")
    st, s0 = get_json("/dashboard/api/v1/status")
    check(st == 200 and s0["sessions"] is None, "no STATS were asked before the dashboard was first watched (sessions is null)")
    s = wait_for(lambda: (lambda x: x if x["sessions"] else None)(get_json("/dashboard/api/v1/status")[1]), 5)
    check(bool(s), "STATS arrive once the dashboard is watched")
    s = s or s0
    check(s["ablit"] == {"alias": NAME + "-ablit"} and s["model_name"] == NAME, "status.ablit names the -ablit name only (no switch)")
    hl = s["health"]
    live = hl["live"]
    check(hl["max_batch_size"] == 64, "health.max_batch_size is the HELLO max_sessions")
    check(all(k in live for k in ("connections", "waiting", "decode_tokens_per_second", "prefill_tokens_per_second", "prompt_pass")), "health.live keys")
    check(hl["spill"]["state"] == "on", "health.spill.state")
    pc = hl["prompt_cache"]
    t0 = pc["spill"][0]
    check(all(k in t0 for k in ("stored_bytes", "cap_bytes", "points", "writes_committed", "written_bytes", "writes_pending")), "prompt_cache.spill[0] keys")
    check(all(k in pc for k in ("restores_begun", "disk_hits", "restored_tokens", "restore_waits", "restore_wait_ns", "restore_wait_max_ns")), "prompt_cache counters")
    ss = s["sessions"]
    want = 2 * 1048576 + 14 * 65536
    check(ss["slots"] == 16 and ss["slots_big"] == 2 and ss["positions"] == want and ss["usable_window"] == 1048576, f"sessions map the slots: {ss['positions']}")
    check(ss["store_cap_bytes"] == want * 890 and "bytes_per_token" not in ss and ss["sessions_resident"] >= 0, "sessions carry the store as its own bytes")

    # node health, polled while watched
    def nodes_up():
        nh = get_json("/dashboard/api/v1/status")[1]["node_health"]
        return nh if len(nh["ranks"]) == 4 and all(r.get("age_ms") is not None for r in nh["ranks"]) else None

    nh = wait_for(nodes_up, 8)
    check(bool(nh), "node_health: four ranks, each with its age")
    if nh:
        r = nh["ranks"][1]
        keys = ["rank", "mem_available_bytes", "mem_total_bytes", "zones_c", "gpu", "load1", "cpus", "cpu_busy_pct", "nvme", "rails", "memory", "age_ms", "at_ms"]
        check(all(k in r for k in keys) and r["rank"] == 1, "a rank's sample keeps the sampler's keys and gets rank and age_ms")
        check(all(k in r["gpu"] for k in ("temp_c", "util_pct", "clock_mhz", "clock_max_mhz", "reasons", "power_w")), "gpu block")
        check(all(k in r["nvme"] for k in ("temp_c", "wear_pct", "root_free_bytes", "root_total_bytes", "available_spare_pct", "spare_threshold_pct",
                                          "media_errors", "critical_warning", "data_units_written", "data_unit_bytes", "power_on_hours", "unsafe_shutdowns")), "nvme block")
        check(len(r["rails"]) == 2 and all(k in r["rails"][0] for k in ("rx_bytes", "tx_bytes", "errors", "discards")), "rails")
        check(nh["zone_names"] == ["cpu-thermal", "soc-thermal"], "node_health.zone_names")
    st, led = get_json("/dashboard/api/v1/ledger")
    check(st == 200 and [x["rank"] for x in led["ranks"]] == [0, 1, 2, 3] and len(led["entries"]) == 12, "ledger: four ranks, three entries each")
    check({e["feature_id"] for e in led["entries"]} == {"weights", "store", "work"} and all(x["device_bytes"] > 0 for x in led["ranks"]), "ledger feature ids")

    # ---- a request's states through the requests route. The prompt takes 5 s to prefill at 4,000 tokens/s.
    long_prompt = "word " * 4000
    seen = {"states": [], "pass": None, "rate": 0, "key": None, "ablit": None, "row": None}
    done = threading.Event()

    def long_req():
        stream_chat(NAME + "-ablit", long_prompt + "MOCK_LONG 200", user="alice")
        done.set()

    th = threading.Thread(target=long_req)
    th.start()
    while not done.is_set():
        _, rq = get_json("/dashboard/api/v1/requests")
        _, sj = get_json("/dashboard/api/v1/status")
        for x in rq["in_flight"]:
            if not seen["states"] or seen["states"][-1] != x["state"]:
                seen["states"].append(x["state"])
            seen["key"], seen["ablit"] = x["key_label"], x["ablit"]
            if x["state"] == "prefill":
                seen["row"] = x
            last = x
        pp = sj["health"]["live"]["prompt_pass"]
        if pp:
            seen["pass"] = pp
        seen["rate"] = max(seen["rate"], sj["health"]["live"]["prefill_tokens_per_second"])
        time.sleep(0.15)
    th.join()
    st = seen["states"]
    check(st[:1] == ["queued"] or st[:1] == ["prefill"], f"states seen {st}")
    check("prefill" in st and st[-1] == "decode", "the request went through prefill to decode")
    check(seen["key"] == "alice" and seen["ablit"] is True, "key_label is the request's user field, ablit is its flag")
    pp = seen["pass"]
    check(pp is not None and 0 < pp["computed"] <= pp["total"] and pp["seconds"] > 0, f"prompt_pass while prefilling: {pp}")
    check(seen["rate"] > 500, f"the prefill rate showed while the prompt was read: {seen['rate']} tok/s")
    rw = seen["row"] or {}
    check(0 < rw.get("computed", 0) <= rw.get("to_compute", 0) and rw.get("cached_tokens") == rw.get("prompt_tokens", -1) - rw.get("to_compute", 0),
          f"a prefilling row carries its own progress and cached part: {({k: rw.get(k) for k in ('computed', 'to_compute', 'cached_tokens', 'prompt_tokens')})}")
    _, rq = get_json("/dashboard/api/v1/requests")
    f0 = rq["recent"][0]
    check(all(k in f0 for k in ("ttft_s", "prompt_tokens", "cached_tokens", "ablit")) and f0["ablit"] is True and f0["ttft_s"] > 0.5, f"recent[0]: {f0}")
    ix = rq["in_flight"]
    check(ix == [], "nothing in flight after the stream ended")
    k = {"id", "state", "prompt_tokens", "generated_tokens", "key_label", "age_s", "kv_tokens", "endpoint", "priority", "ablit"}
    check(k <= set(last), "in_flight keys")
    check(chat(NAME, "no user field")[0] == 200 and get_json("/dashboard/api/v1/requests")[1]["recent"][0]["ablit"] is False, "a request without user is labelled api and not ablit")

    # ---- the series
    for name in ("decode_tps", "prefill_tps"):
        st, ser = get_json(f"/dashboard/api/v1/series?name={name}&seconds=900&step=10")
        n_real = sum(1 for p in ser["points"] if p is not None)
        check(st == 200 and len(ser["points"]) == 90 and len(ser["peaks"]) == 90 and ser["step_s"] == 10 and abs(ser["last_ms"] - time.time() * 1000) < 20000
              and n_real >= 1, f"series {name}: 90 points, {n_real} with readings")
    _, ser = get_json("/dashboard/api/v1/series?name=decode_tps&seconds=900&step=10")
    check(max(p for p in ser["peaks"] if p is not None) > 20, "the decode series saw the stream (peak %.0f tok/s)" % max(p for p in ser["peaks"] if p is not None))
    _, ser = get_json("/dashboard/api/v1/series?name=prefill_tps&seconds=900&step=10")
    check(max(p for p in ser["peaks"] if p is not None) > 500, "the prefill series saw the prompt")
    check(http_("GET", "/dashboard/api/v1/series?name=nope")[0] == 400, "an unknown series name is 400")

    # ---- the served names: the plain name (and any other) runs plain, the -ablit name runs ablated
    check(http_("GET", "/dashboard/api/v1/status", headers={"Host": "rebound.example:%d" % PORT})[0] == 403,
          "a Host the dashboard does not answer is refused (a rebound name)")
    check(http_("POST", "/dashboard/api/v1/ablit", {"on": True})[0] in (404, 405), "there is no switch to change the plain name")
    check(not os.path.exists(os.path.join(TMP, "front.ablit")), "no switch file beside the table")
    st, d = chat(NAME, "plain")
    check(d["model"] == NAME and text_of(d) == "Mock reply to: plain", "the plain name runs plain")
    st, d = chat("other", "other name")
    check(text_of(d) == "Mock reply to: other name", "any other model runs plain")
    st, d = chat(NAME + "-ablit", "named")
    check(d["model"] == NAME + "-ablit" and text_of(d).startswith("[ablit] "), "the -ablit name always runs ablated")

    # ---- STATS only while watched: after the watch lapses a finished request does not reach the dashboard's numbers
    time.sleep(11.5)
    before = get_json("/dashboard/api/v1/status")[1]["health"]["prompt_cache"]["spill"][0]["writes_committed"]
    time.sleep(0.2)
    chat(NAME, "one request while nobody watches")
    # this first call after the lapse still sees the old numbers: the engine was not asked since
    stale = get_json("/dashboard/api/v1/status")[1]["health"]["prompt_cache"]["spill"][0]["writes_committed"]
    check(stale == before, f"STATS were not asked while the dashboard was not watched ({before} then {stale})")
    fresh = wait_for(lambda: (lambda x: x if x > before else None)(get_json("/dashboard/api/v1/status")[1]["health"]["prompt_cache"]["spill"][0]["writes_committed"]), 4)
    check(bool(fresh), "and arrive again once it is watched")

    # ---- sanity timing: a stream with and without the page polling (a check, not a measurement)
    def run_streams(n):
        gaps = []
        for _ in range(n):
            stream_chat(NAME, "MOCK_LONG 400", gaps=gaps)
        return gaps

    quiet = run_streams(3)
    stop = threading.Event()

    def page():
        tick = 0
        while not stop.is_set():
            tick += 1
            for p in ("status", "requests"):
                poll_get("/dashboard/api/v1/" + p)
            if tick % 5 == 1:
                for n in ("decode_tps", "prefill_tps"):
                    poll_get(f"/dashboard/api/v1/series?name={n}&seconds=900&step=10")
            if tick % 30 == 1:
                poll_get("/dashboard/api/v1/ledger")
            stop.wait(1.0)

    def hammer():
        while not stop.is_set():
            for p in ("status", "requests"):
                poll_get("/dashboard/api/v1/" + p)

    ths = [threading.Thread(target=page), threading.Thread(target=hammer)]
    [t.start() for t in ths]
    polled = run_streams(3)
    stop.set()
    [t.join() for t in ths]
    q, p = statistics.median(quiet) * 1000, statistics.median(polled) * 1000
    q99, p99 = sorted(quiet)[int(len(quiet) * .99)] * 1000, sorted(polled)[int(len(polled) * .99)] * 1000
    print(f"     token gaps over {len(quiet)} / {len(polled)} chunks (mock delay 5 ms; the page's own polling plus one more client in a tight loop)\n"
          f"     without the page: median {q:.2f} ms, p99 {q99:.2f} ms; with it: median {p:.2f} ms, p99 {p99:.2f} ms")
    check(p < q * 1.5 + 1.0, "polling the dashboard does not slow the stream (sanity only)")


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
    FAILS.append("the script raised an exception")
finally:
    finish()
