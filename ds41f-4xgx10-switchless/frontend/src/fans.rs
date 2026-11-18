//! The FANS box of the dashboard: the fan profile of the Sparks, set and read through one tool (`--fan-tool PATH`,
//! docs/dashboard.md). The tool writes a profile name into a small file on each Spark and reads each Spark's daemon status;
//! this front-end never touches a fan, a sensor or ssh itself, and a request that is not one of the fixed profile names
//! never reaches the tool. Without `--fan-tool` the box is absent: the routes answer `{"tool":false}` (GET) or 503 (POST).

use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;

/// The profiles the page offers, in the order it lists them: id, label, group.
pub const PROFILES: &[(&str, &str, &str)] = &[
    ("stock", "Stock (Dell EC curve)", "Curves"),
    ("medium", "Medium", "Curves"),
    ("performance", "Performance (default)", "Curves"),
    ("max", "Max", "Fixed"),
    ("hold-4500", "Hold 4,500 RPM", "Fixed"),
    ("hold-6300", "Hold 6,300 RPM", "Fixed"),
    ("hold-9000", "Hold 9,000 RPM", "Fixed"),
];

/// A status reading older than this (seconds, by the Spark's own clock) is stale.
pub const STALE_S: f64 = 10.0;
/// A status answer is reused this long: the tool (an ssh to each Spark) runs at most this often however many pages ask.
const CACHE: Duration = Duration::from_secs(4);
const STATUS_TIMEOUT: Duration = Duration::from_secs(20);
const SET_TIMEOUT: Duration = Duration::from_secs(25);
/// The tool's answer is one JSON object; a longer one is refused.
const MAX_OUT: usize = 1 << 20;

pub fn profile_list() -> Value {
    Value::Array(PROFILES.iter().map(|(id, label, group)| json!({"id": id, "label": label, "group": group})).collect())
}

/// A change from the page: exactly `{"profile": "<id>"}` with an id of `PROFILES`; anything else is refused with the list.
pub fn parse_fan_request(v: &Value) -> Result<&'static str, String> {
    let ids = || PROFILES.iter().map(|p| p.0).collect::<Vec<_>>().join(", ");
    let Some(o) = v.as_object() else { return Err(format!("a JSON object with one key, profile: one of {}", ids())) };
    if let Some(k) = o.keys().find(|k| k.as_str() != "profile") {
        return Err(format!("{k}: not a fan setting (profile is the only one)"));
    }
    let Some(p) = o.get("profile") else { return Err(format!("profile: one of {}", ids())) };
    let Some(p) = p.as_str() else { return Err(format!("profile: a name, one of {}", ids())) };
    PROFILES.iter().find(|x| x.0 == p).map(|x| x.0).ok_or_else(|| format!("profile {p:?} is not one of {}", ids()))
}

fn unix_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// The tool's `status --json` answer as the API gives it: one entry a Spark, the page's flags worked out here (stale, mixed,
/// the profile all four ask for), so the page and any script read one shape.
pub fn merge_status(tool: &Value, at_ms: u64) -> Result<Value, String> {
    let nodes = tool.get("nodes").and_then(Value::as_array).ok_or("the fan tool's answer has no nodes")?;
    let mut out = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    let mut asked: Vec<Option<String>> = Vec::new();
    for (i, n) in nodes.iter().enumerate() {
        let ok = n.get("ok").and_then(Value::as_bool).unwrap_or(false);
        let st = n.get("status").filter(|s| s.is_object());
        let get = |k: &str| st.and_then(|s| s.get(k)).cloned().unwrap_or(Value::Null);
        let fallback = st.and_then(|s| s.get("fallback")).and_then(Value::as_bool).unwrap_or(false);
        let age = n.get("age_s").and_then(Value::as_f64);
        let stale = !ok || st.is_none() || fallback || age.is_none_or(|a| a > STALE_S);
        if ok && !fallback {
            if let Some(p) = get("profile").as_str() {
                if !seen.iter().any(|s| s == p) {
                    seen.push(p.to_string());
                }
            }
        }
        if ok {
            asked.push(get("requested").as_str().filter(|s| !s.is_empty()).map(str::to_string));
        } else {
            asked.push(None);
        }
        out.push(json!({
            "rank": n.get("rank").cloned().unwrap_or(json!(i)), "host": n.get("host").cloned().unwrap_or(Value::Null),
            "ok": ok, "stale": stale, "fallback": fallback,
            "profile": get("profile"), "requested": get("requested"), "profile_error": get("profile_error"),
            "state": get("state"), "floor_rpm": get("floor_rpm"), "hottest_c": get("hottest_c"), "rpm": get("rpm"),
            "guard": get("zero_rpm_guard"), "age_s": age, "error": n.get("error").cloned().unwrap_or(Value::Null),
        }));
    }
    let wanted = match asked.first() {
        Some(Some(p)) if asked.iter().all(|a| a.as_deref() == Some(p.as_str())) => json!(p),
        _ => Value::Null,
    };
    Ok(json!({"at_ms": at_ms, "tool": true, "profiles": profile_list(), "wanted": wanted, "mixed": seen.len() > 1,
              "pinned": tool.get("pinned").cloned().unwrap_or(Value::Null), "nodes": out}))
}

/// What a `set` came to.
#[derive(Debug, PartialEq)]
pub enum SetOutcome {
    /// at least one Spark took it: the answer for the page (`sent`, one entry a Spark)
    Done(Value),
    /// the fan tool reports the fans and clock as pinned: its reason
    Pinned(String),
    /// the tool failed, or no Spark took it
    Failed(String),
}

/// The tool's exit code and output for `set --json PROFILE`: 0 every Spark took it, 1 some did not, 3 pinned.
pub fn interpret_set(code: Option<i32>, stdout: &str, profile: &str) -> SetOutcome {
    let v: Option<Value> = serde_json::from_str(stdout.trim()).ok();
    match code {
        Some(3) => SetOutcome::Pinned(v.and_then(|v| v.get("pinned").and_then(Value::as_str).map(str::to_string)).unwrap_or_else(|| "pinned".into())),
        Some(0) | Some(1) => {
            let Some(nodes) = v.as_ref().and_then(|v| v.get("nodes")).and_then(Value::as_array) else {
                return SetOutcome::Failed("the fan tool's answer has no nodes".into());
            };
            let took = nodes.iter().filter(|n| n.get("ok").and_then(Value::as_bool) == Some(true)).count();
            if took == 0 {
                return SetOutcome::Failed("no Spark took the profile".into());
            }
            let slim: Vec<Value> = nodes.iter().map(|n| json!({"rank": n.get("rank"), "ok": n.get("ok"), "error": n.get("error")})).collect();
            SetOutcome::Done(json!({"sent": {"profile": profile}, "nodes": slim}))
        }
        Some(c) => SetOutcome::Failed(format!("the fan tool exited {c}")),
        None => SetOutcome::Failed("the fan tool was stopped".into()),
    }
}

pub struct Fans {
    tool: Option<PathBuf>,
    cache: Mutex<Option<(Instant, Value)>>,
}

impl Fans {
    pub fn new(tool: Option<PathBuf>) -> Fans {
        Fans { tool, cache: Mutex::new(None) }
    }

    pub fn present(&self) -> bool {
        self.tool.is_some()
    }

    /// The Sparks' fans now (the tool's status, at most every few seconds however many pages ask); one run at a time.
    pub async fn status(&self) -> Result<Value, String> {
        let Some(tool) = &self.tool else { return Err("no fan tool".into()) };
        let mut cache = self.cache.lock().await;
        if let Some((at, v)) = cache.as_ref() {
            if at.elapsed() < CACHE {
                return Ok(v.clone());
            }
        }
        let (code, out) = run(tool, &["status", "--json"], STATUS_TIMEOUT).await?;
        // 1: no Spark answered (the answer still names each); any other code is the tool failing
        if !matches!(code, Some(0) | Some(1)) {
            return Err(format!("the fan tool exited {}", code.map_or("by signal".to_string(), |c| c.to_string())));
        }
        let v: Value = serde_json::from_str(out.trim()).map_err(|e| format!("the fan tool's answer: {e}"))?;
        let merged = merge_status(&v, unix_ms())?;
        *cache = Some((Instant::now(), merged.clone()));
        Ok(merged)
    }

    pub async fn set(&self, profile: &'static str) -> SetOutcome {
        let Some(tool) = &self.tool else { return SetOutcome::Failed("no fan tool".into()) };
        let mut cache = self.cache.lock().await;
        *cache = None;
        match run(tool, &["set", "--json", profile], SET_TIMEOUT).await {
            Ok((code, out)) => interpret_set(code, &out, profile),
            Err(e) => SetOutcome::Failed(e),
        }
    }
}

/// Run the tool with fixed arguments (no shell, no input) and read its output, bounded in time and size.
async fn run(tool: &PathBuf, args: &[&str], limit: Duration) -> Result<(Option<i32>, String), String> {
    use tokio::io::AsyncReadExt;
    let mut child = tokio::process::Command::new(tool)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("the fan tool did not start: {e}"))?;
    let mut stdout = child.stdout.take().ok_or("no output pipe")?;
    let work = async {
        let mut buf = Vec::new();
        (&mut stdout).take(MAX_OUT as u64 + 1).read_to_end(&mut buf).await.map_err(|e| e.to_string())?;
        if buf.len() > MAX_OUT {
            return Err("the fan tool's answer is too long".to_string());
        }
        let status = child.wait().await.map_err(|e| e.to_string())?;
        Ok((status.code(), String::from_utf8_lossy(&buf).into_owned()))
    };
    match tokio::time::timeout(limit, work).await {
        Ok(r) => r,
        Err(_) => Err("the fan tool did not answer in time".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_names_one_profile_of_the_fixed_list() {
        for (id, _, _) in PROFILES {
            assert_eq!(parse_fan_request(&json!({"profile": id})), Ok(*id), "{id}");
        }
        for bad in [json!({}), json!({"profile": ""}), json!({"profile": "hold-5000"}), json!({"profile": "Max"}), json!({"profile": "quiet"}),
                    json!({"profile": 3}), json!({"profile": null}), json!({"profile": ["max"]}), json!({"profile": "max", "gpu_clock": "2200"}),
                    json!({"profile": "max", "x": 1}), json!({"gpu_clock": "2200"}), json!(["max"]), json!("max"), json!(5), json!(null),
                    json!({"profile": "max;rm"}), json!({"profile": "../max"}), json!({"profile": "hold-13500"})] {
            assert!(parse_fan_request(&bad).is_err(), "{bad}");
        }
        let e = parse_fan_request(&json!({"profile": "quiet"})).unwrap_err();
        assert!(e.contains("stock") && e.contains("hold-9000"), "the list is in the message: {e}");
    }

    fn node(rank: u64, profile: &str, age: f64) -> Value {
        json!({"rank": rank, "host": format!("n{rank}"), "ok": true, "stale": false, "age_s": age, "error": null,
               "status": {"v": 1, "at": 1.0, "profile": profile, "requested": profile, "profile_error": null, "state": 5, "floor_rpm": 6300,
                          "hottest_c": 57.4, "filtered_c": 57.1, "rpm": [3620, 5400], "zero_rpm_guard": false, "driver": "0.1.4"}})
    }

    #[test]
    fn four_sparks_on_one_profile_are_not_mixed_and_name_what_is_wanted() {
        let tool = json!({"pinned": null, "nodes": [node(0, "medium", 1.0), node(1, "medium", 2.0), node(2, "medium", 0.5), node(3, "medium", 3.0)]});
        let m = merge_status(&tool, 1234).unwrap();
        assert_eq!((m["tool"].clone(), m["mixed"].clone(), m["wanted"].clone(), m["at_ms"].clone()), (json!(true), json!(false), json!("medium"), json!(1234)));
        assert_eq!(m["profiles"].as_array().unwrap().len(), 7);
        let n = &m["nodes"][1];
        assert_eq!((n["rank"].clone(), n["ok"].clone(), n["stale"].clone(), n["profile"].clone(), n["state"].clone(), n["floor_rpm"].clone()),
                   (json!(1), json!(true), json!(false), json!("medium"), json!(5), json!(6300)));
        assert_eq!((n["rpm"].clone(), n["hottest_c"].clone(), n["age_s"].clone(), n["guard"].clone()), (json!([3620, 5400]), json!(57.4), json!(2.0), json!(false)));
        assert!(m["pinned"].is_null());
    }

    #[test]
    fn a_different_profile_makes_the_set_mixed() {
        let tool = json!({"pinned": null, "nodes": [node(0, "medium", 1.0), node(1, "medium", 1.0), node(2, "max", 1.0), node(3, "medium", 1.0)]});
        let m = merge_status(&tool, 0).unwrap();
        assert_eq!(m["mixed"], json!(true));
        assert!(m["wanted"].is_null(), "no one profile is asked for by all");
    }

    #[test]
    fn an_old_reading_is_stale_and_a_fresh_one_is_not() {
        let tool = json!({"nodes": [node(0, "medium", 9.9), node(1, "medium", 10.5)]});
        let m = merge_status(&tool, 0).unwrap();
        assert_eq!((m["nodes"][0]["stale"].clone(), m["nodes"][1]["stale"].clone()), (json!(false), json!(true)));
    }

    #[test]
    fn a_spark_read_from_the_kernel_is_stale_and_does_not_count_for_mixed() {
        let mut dead = node(2, "x", 0.0);
        dead["status"]["profile"] = Value::Null;
        dead["status"]["fallback"] = json!(true);
        dead["status"]["requested"] = json!("medium");
        dead["age_s"] = Value::Null;
        let tool = json!({"nodes": [node(0, "medium", 1.0), node(1, "medium", 1.0), dead, node(3, "medium", 1.0)]});
        let m = merge_status(&tool, 0).unwrap();
        assert_eq!((m["nodes"][2]["stale"].clone(), m["nodes"][2]["fallback"].clone(), m["mixed"].clone()), (json!(true), json!(true), json!(false)));
        assert_eq!(m["wanted"], json!("medium"), "the file on the dead one still says medium");
    }

    #[test]
    fn an_unreachable_spark_is_not_ok_and_stale_with_its_error() {
        let down = json!({"rank": 3, "host": "n3", "ok": false, "stale": true, "age_s": null, "status": null, "error": "no answer: ssh: refused"});
        let m = merge_status(&json!({"nodes": [node(0, "max", 1.0), down]}), 0).unwrap();
        let n = &m["nodes"][1];
        assert_eq!((n["ok"].clone(), n["stale"].clone(), n["error"].clone(), n["profile"].clone()), (json!(false), json!(true), json!("no answer: ssh: refused"), Value::Null));
        assert!(m["wanted"].is_null());
    }

    #[test]
    fn the_pin_is_carried() {
        let m = merge_status(&json!({"pinned": "pinned for a test", "nodes": []}), 0).unwrap();
        assert_eq!(m["pinned"], json!("pinned for a test"));
    }

    #[test]
    fn a_tool_answer_without_nodes_is_refused() {
        assert!(merge_status(&json!({"x": 1}), 0).is_err());
        assert!(merge_status(&json!([]), 0).is_err());
    }

    #[test]
    fn set_answers_are_read_by_exit_code() {
        let ok = r#"{"profile":"max","pinned":null,"nodes":[{"rank":0,"ok":true,"error":null},{"rank":1,"ok":false,"error":"no answer"}]}"#;
        match interpret_set(Some(1), ok, "max") {
            SetOutcome::Done(v) => {
                assert_eq!(v["sent"], json!({"profile": "max"}));
                assert_eq!(v["nodes"][1], json!({"rank": 1, "ok": false, "error": "no answer"}));
            }
            o => panic!("{o:?}"),
        }
        assert!(matches!(interpret_set(Some(0), ok, "max"), SetOutcome::Done(_)));
        let none = r#"{"profile":"max","nodes":[{"rank":0,"ok":false,"error":"x"}]}"#;
        assert!(matches!(interpret_set(Some(1), none, "max"), SetOutcome::Failed(_)), "no Spark took it");
        assert_eq!(interpret_set(Some(3), r#"{"pinned":"window 7","nodes":[]}"#, "max"), SetOutcome::Pinned("window 7".into()));
        assert_eq!(interpret_set(Some(3), "garbage", "max"), SetOutcome::Pinned("pinned".into()));
        assert!(matches!(interpret_set(Some(2), ok, "max"), SetOutcome::Failed(_)));
        assert!(matches!(interpret_set(Some(0), "not json", "max"), SetOutcome::Failed(_)));
        assert!(matches!(interpret_set(None, ok, "max"), SetOutcome::Failed(_)));
    }

    #[tokio::test]
    async fn without_a_tool_nothing_runs() {
        let f = Fans::new(None);
        assert!(!f.present());
        assert!(f.status().await.is_err());
        assert!(matches!(f.set("max").await, SetOutcome::Failed(_)));
    }

    #[tokio::test]
    async fn a_tool_that_is_missing_is_an_error_not_a_panic() {
        let f = Fans::new(Some(PathBuf::from("/nonexistent/dgx-fan-fleet")));
        assert!(f.present());
        assert!(f.status().await.unwrap_err().contains("did not start"));
        assert!(matches!(f.set("max").await, SetOutcome::Failed(_)));
    }
}
