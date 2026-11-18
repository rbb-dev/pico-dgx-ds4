#!/usr/bin/env python3
"""A stand-in for dgx-fan-fleet (the fan tool dsfront runs with --fan-tool), for dashboard_smoke.py: the same three commands
(`status --json`, `set --json PROFILE`, `clock set --json MHZ`), four fake Sparks, no ssh and no fan.

State is a JSON file named by FAKE_FAN_STATE: {"profiles": [four names], "pinned": null | "reason", "down": [ranks],
"stale": [ranks]}; every call's arguments are appended, one JSON list a line, to FAKE_FAN_LOG when it is set.
"""
import json
import os
import sys
import time

FLOOR = {"stock": 0, "medium": 5, "performance": 5, "max": 12, "hold-4500": 5, "hold-6300": 5, "hold-9000": 8}
RPM = [0, 2700, 3600, 4500, 5400, 6300, 7200, 8100, 9000, 10125, 11250, 12375, 13500]


def load():
    path = os.environ["FAKE_FAN_STATE"]
    try:
        with open(path) as f:
            return json.load(f)
    except OSError:
        return {"profiles": ["performance"] * 4, "pinned": None, "down": [], "stale": []}


def save(st):
    with open(os.environ["FAKE_FAN_STATE"], "w") as f:
        json.dump(st, f)


def main(argv):
    log = os.environ.get("FAKE_FAN_LOG")
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(argv) + "\n")
    st = load()
    if argv == ["status", "--json"]:
        nodes = []
        for r in range(4):
            if r in st["down"]:
                nodes.append({"rank": r, "host": f"spark{r + 1}", "ok": False, "stale": True, "age_s": None, "status": None,
                              "error": "no answer: ssh: connect: Connection refused"})
                continue
            p = st["profiles"][r]
            s = FLOOR.get(p, 5)
            nodes.append({"rank": r, "host": f"spark{r + 1}", "ok": True, "stale": r in st["stale"], "age_s": 30.0 if r in st["stale"] else 1.0,
                          "error": None,
                          "status": {"v": 1, "at": time.time(), "profile": p, "requested": p, "profile_error": None, "state": s,
                                     "floor_rpm": RPM[s], "hottest_c": 55.0 + r, "filtered_c": 55.0 + r, "rpm": [3000 + 100 * r, 4500],
                                     "zero_rpm_guard": False, "driver": "0.1.4"}})
        profs = {n["status"]["profile"] for n in nodes if n["ok"]}
        print(json.dumps({"at": time.time(), "pinned": st["pinned"], "mixed": len(profs) > 1,
                          "profile": next(iter(profs)) if len(profs) == 1 and not st["down"] else None, "nodes": nodes}))
        return 0 if any(n["ok"] for n in nodes) else 1
    if len(argv) == 3 and argv[:2] == ["set", "--json"]:
        p = argv[2]
        if st["pinned"]:
            print(json.dumps({"profile": p, "pinned": st["pinned"], "nodes": []}))
            return 3
        nodes = []
        for r in range(4):
            if r in st["down"]:
                nodes.append({"rank": r, "host": f"spark{r + 1}", "ok": False, "error": "no answer: ssh: connect: Connection refused"})
            else:
                st["profiles"][r] = p
                nodes.append({"rank": r, "host": f"spark{r + 1}", "ok": True, "error": None})
        save(st)
        print(json.dumps({"profile": p, "pinned": None, "nodes": nodes}))
        return 0 if all(n["ok"] for n in nodes) else 1
    print("fake_fan_tool: unexpected arguments " + json.dumps(argv), file=sys.stderr)
    return 2


sys.exit(main(sys.argv[1:]))
