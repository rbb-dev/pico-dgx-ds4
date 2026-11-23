#!/usr/bin/env python3
"""A tripwire for new allocation sites in the engine (not the byte account).

Lists every call of cudaMalloc*, cudaHostAlloc, cuMemCreate, cuMemAddressReserve, mmap, big_alloc, halloc, dalloc,
posix_memalign, aligned_alloc, ibv_reg_mr, cudaGraphInstantiate* and the display reservation's CREATE_DUMB ioctl in the
engine's sources, and compares them with engine/alloc_sites.txt: the call's text and its enclosing function, with the class
(Fixed, Side, Store numbers, Store regions) and the start-table line that counts it. A call not listed, or listed and gone,
fails and prints the diff to accept. Memory with no call site in engine/ (the CUDA context, NCCL) is not in it.

  alloc_audit.py [--src ENGINE_DIR] [--sites FILE]      compare; exit 1 on a difference (prints it)
  alloc_audit.py --init [--src ..] [--sites ..]         write the sites file from the sources, classes by the rules below
  alloc_audit.py --accept [--src ..] [--sites ..]       keep the classes of the lines that stay, add new ones as unclassified,
                                                        drop the gone ones: the diff accepted
  alloc_audit.py --selftest [--src ENGINE_DIR]          the tree passes; a call added in a scratch copy is reported; a call
                                                        removed from it is reported
Files under tests/ and tools/, build directories and *_test.* files are not the engine and are not scanned.
"""
import argparse
import collections
import os
import re
import shutil
import subprocess
import sys
import tempfile

CALLS = (r"cudaMalloc\w*|cudaHostAlloc|cuMemCreate|cuMemAddressReserve|mmap|big_alloc|halloc|dalloc|posix_memalign|"
         r"aligned_alloc|ibv_reg_mr|cudaGraphInstantiate\w*")
CALL_RE = re.compile(r"\b(?:%s)\s*(?:<[^<>()]*(?:<[^<>()]*>[^<>()]*)*>)?\s*\(" % CALLS)
DUMB_RE = re.compile(r"\b\w*CREATE_DUMB\w*")
STR_RE = re.compile(r'"(?:\\.|[^"\\])*"')
EXTS = (".cu", ".cpp", ".h", ".cuh", ".hpp")
SKIP_DIRS = {"tests", "tools", "build", "build-old"}

# class, start-table line by where the call is: (path prefix, class, line); the first match wins. Used by --init and for the
# new entries --accept adds (they are then to be classified by a person: `unclassified` until set).
RULES = [
    ("weights/", "Fixed", "weights"),
    ("model/vmm", "Store numbers", "the chunk pool"),
    ("model/state.cu", "Store numbers", "the chunk pool"),
    ("model/spill", "Side", "the state gather buffer"),
    ("model/window.cu", "Store regions", "windows' buffers"),
    ("model/ablit", "Fixed", "ablit tables"),
    ("model/prefill.cu", "Fixed", "prefill buffers"),
    ("model/spec.cu", "Fixed", "draft module buffers and graphs"),
    ("dspark/", "Fixed", "draft module buffers and graphs"),
    ("model/model.cu", "Fixed", "step buffers and graphs"),
    ("model/copy", "Fixed", "step buffers and graphs"),
    ("model/impl.h", "Fixed", "step buffers and graphs"),
    ("model/wave_probe", "Fixed", "diagnostics"),
    ("comm/", "Fixed", "ring buffers (registered)"),
    ("engram/display", "Side", "Engram display reservation"),
    ("engram/", "Side", "Engram cache"),
    ("vision/", "Fixed", "vision weights and work buffers"),
    ("kernels/", "Fixed", "kernel scratch"),
    ("server/", "Side", "server buffers"),
    ("common/", "Side", "the glibc heap"),
]


def scanned(src):
    for root, dirs, files in os.walk(src):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("build"))
        for f in sorted(files):
            if f.endswith(EXTS) and "_test." not in f:
                yield os.path.join(root, f)


def strip_code(line, in_block):
    """The line without strings and comments; in_block: inside a /* */ comment at its start."""
    line = STR_RE.sub('""', line)
    out, i = "", 0
    while i < len(line):
        if in_block:
            j = line.find("*/", i)
            if j < 0:
                return out, True
            i, in_block = j + 2, False
            continue
        if line.startswith("//", i):
            break
        if line.startswith("/*", i):
            in_block = True
            i += 2
            continue
        out += line[i]
        i += 1
    return out, in_block


FUNC_RE = re.compile(r"^(\s*)(?:[\w:<>,\*&~\[\]]+\s+)*?(~?[A-Za-z_]\w*(?:::~?[A-Za-z_]\w*)*)\s*\([^;]*\)\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?\{?\s*$")
NOT_FUNC = {"if", "for", "while", "switch", "catch", "return", "else", "do", "sizeof"}


def enclosing(lines, n):
    """The name of the function a line sits in: the nearest line above at indentation 0 or 2 that opens a function."""
    for k in range(n, -1, -1):
        ln = lines[k]
        if not ln.strip() or ln[0] in " \t" and len(ln) - len(ln.lstrip()) > 2:
            continue
        m = FUNC_RE.match(ln)
        if m and m.group(2) not in NOT_FUNC and (ln.rstrip().endswith("{") or (k + 1 < len(lines) and lines[k + 1].lstrip().startswith("{"))):
            return m.group(2)
    return "(file scope)"


def call_text(code, m):
    """The call's text from its name to the matching paren (or the line's end), spaces collapsed."""
    s = code[m.start():]
    depth, end = 0, len(s)
    for i, c in enumerate(s):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    return re.sub(r"\s+", " ", s[:end]).strip()


def sites(src):
    """[(rel path, function, text, line number)] in file order."""
    out = []
    for path in scanned(src):
        rel = os.path.relpath(path, src).replace(os.sep, "/")
        with open(path, errors="replace") as f:
            lines = f.read().split("\n")
        block = False
        for n, raw in enumerate(lines):
            code, block = strip_code(raw, block)
            for m in CALL_RE.finditer(code):
                out.append((rel, enclosing(lines, n), call_text(code, m), n + 1))
            for m in DUMB_RE.finditer(code):
                out.append((rel, enclosing(lines, n), m.group(0), n + 1))
    return out


def classify(rel):
    for prefix, cls, line in RULES:
        if rel.startswith(prefix):
            return cls, line
    return "unclassified", "unclassified"


def counted(found):
    c = collections.OrderedDict()
    for rel, fn, text, n in found:
        c.setdefault((rel, fn, text), []).append(n)
    return c


def read_sites(path):
    out = collections.OrderedDict()
    with open(path) as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#"):
                continue
            p = ln.rstrip("\n").split("\t")
            if len(p) != 6:
                raise SystemExit("alloc_audit: %s: a line of %d fields, 6 wanted: %r" % (path, len(p), ln))
            cls, line, rel, fn, cnt, text = p
            out[(rel, fn, text)] = (cls, line, int(cnt))
    return out


HEADER = """# Every allocation call site in engine/ (tools/alloc_audit.py checks this file against the sources: a call not listed, or
# listed and gone, fails). Fields, tab separated: class, start-table line that counts it, file, enclosing function, number of
# identical calls in it, the call's text. Classes: Fixed (before the sizing read), Side (a rank's own capped line taken after
# it), Store numbers, Store regions (whole groups lent). To accept a change: tools/alloc_audit.py --accept, then set the class
# and line of every `unclassified` entry it adds.
"""


def write_sites(path, found, old=None):
    c = counted(found)
    with open(path, "w") as f:
        f.write(HEADER)
        for (rel, fn, text), ns in c.items():
            if old is None:
                cls, line = classify(rel)
            else:
                cls, line = old[(rel, fn, text)][:2] if (rel, fn, text) in old else ("unclassified", "unclassified")
            f.write("\t".join([cls, line, rel, fn, str(len(ns)), text]) + "\n")


def compare(found, listed):
    c = counted(found)
    new, gone, count = [], [], []
    for key, ns in c.items():
        if key not in listed:
            new.append((key, ns))
        elif listed[key][2] != len(ns):
            count.append((key, listed[key][2], ns))
    for key, v in listed.items():
        if key not in c:
            gone.append((key, v))
    return new, gone, count


def report(new, gone, count):
    for (rel, fn, text), ns in new:
        print("NEW  %s:%s in %s: %s  (not in alloc_sites.txt)" % (rel, ",".join(map(str, ns)), fn, text))
    for (rel, fn, text), v in gone:
        print("GONE %s in %s: %s  (listed as %s / %s, not in the sources)" % (rel, fn, text, v[0], v[1]))
    for (rel, fn, text), was, ns in count:
        print("COUNT %s in %s: %s  (listed %d, found %d at %s)" % (rel, fn, text, was, len(ns), ",".join(map(str, ns))))
    if new or gone or count:
        print("alloc_audit: the sources and alloc_sites.txt differ: if the change is wanted, run tools/alloc_audit.py --accept and "
              "classify what it adds")
        return 1
    return 0


def selftest(src):
    here = os.path.abspath(__file__)
    sites_file = os.path.join(src, "alloc_sites.txt")
    bad = 0

    def ok(cond, what):
        nonlocal bad
        print("%s %s" % ("ok  " if cond else "FAIL", what))
        bad += 0 if cond else 1

    def run(root, *extra):
        p = subprocess.run([sys.executable, here, "--src", root, "--sites", os.path.join(root, "alloc_sites.txt")] + list(extra),
                           capture_output=True, text=True)
        return p.returncode, p.stdout + p.stderr

    rc, out = run(src)
    ok(rc == 0, "the tree and alloc_sites.txt agree")
    with tempfile.TemporaryDirectory(prefix="alloc_audit.") as tmp:
        root = os.path.join(tmp, "engine")
        shutil.copytree(src, root, ignore=shutil.ignore_patterns("build*", "*.o"))
        found = sites(root)
        ok(len(found) > 50, "the scan finds the engine's call sites (%d)" % len(found))
        # an added call site is reported
        target = os.path.join(root, "model", "prefill.cu")
        with open(target) as f:
            text = f.read()
        with open(target, "w") as f:
            f.write(text + "\nvoid audit_selftest_added(void** p) {\n  cudaMalloc(p, 1 << 20);\n}\n")
        rc, out = run(root)
        ok(rc == 1 and "NEW" in out and "audit_selftest_added" in out, "a call added in a scratch copy is reported")
        # accept keeps the classes and marks the new one
        rc, out = run(root, "--accept")
        ok(rc == 0, "--accept writes the diff")
        rc, out = run(root)
        ok(rc == 0, "after --accept the tree passes")
        with open(os.path.join(root, "alloc_sites.txt")) as f:
            acc = f.read()
        ok("audit_selftest_added" in acc, "the accepted call is listed")
        with open(target, "w") as f:
            f.write(text)
        rc, out = run(root)
        ok(rc == 1 and "GONE" in out and "audit_selftest_added" in out, "a listed call that is gone is reported")
        # one removed from the sources
        shutil.copyfile(sites_file, os.path.join(root, "alloc_sites.txt"))
        victim = next((s for s in found if s[0] == "model/window.cu"), found[0])
        vpath = os.path.join(root, victim[0])
        with open(vpath) as f:
            vlines = f.read().split("\n")
        vlines[victim[3] - 1] = "  // removed by the selftest"
        with open(vpath, "w") as f:
            f.write("\n".join(vlines))
        rc, out = run(root)
        ok(rc == 1 and ("GONE" in out or "COUNT" in out), "a call removed from the sources is reported")
        # a call inside a comment or a string is not a call site
        shutil.copyfile(sites_file, os.path.join(root, "alloc_sites.txt"))
        shutil.copyfile(os.path.join(src, victim[0]), vpath)
        with open(vpath, "a") as f:
            f.write('\n// cudaMalloc(&p, 4);\nconst char* audit_s = "mmap(0, 4)";\n')
        rc, out = run(root)
        ok(rc == 0, "a call in a comment or a string is not a call site")
    print("%s alloc_audit selftest: %d failed" % ("FAIL" if bad else "PASS", bad))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--src", default=default_src)
    ap.add_argument("--sites")
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--accept", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    src = os.path.abspath(a.src)
    sites_file = a.sites or os.path.join(src, "alloc_sites.txt")
    if a.selftest:
        return selftest(src)
    found = sites(src)
    if a.init:
        write_sites(sites_file, found)
        print("alloc_audit: %d call sites written to %s" % (len(found), sites_file))
        return 0
    listed = read_sites(sites_file) if os.path.exists(sites_file) else collections.OrderedDict()
    if a.accept:
        write_sites(sites_file, found, listed)
        new, gone, count = compare(found, listed)
        print("alloc_audit: accepted: %d new (unclassified), %d gone, %d recounted" % (len(new), len(gone), len(count)))
        return 0
    return report(*compare(found, listed))


if __name__ == "__main__":
    sys.exit(main())
