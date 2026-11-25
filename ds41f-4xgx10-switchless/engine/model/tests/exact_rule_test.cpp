// The exact shared steps' settings (model/exact_rule.h), on the host, no GPU. Holds:
//  H1  DSGB10_NORM_EXACT: unset / empty / 0 / 1 are values; 2, -1, "01x", " 1", "1 " and "on" are refused
//  H2  DSGB10_WIDE_EXACT: every value 0..95 without bit 32 is a value (and returns itself); 32..63 with bit 32, 96 and up, signs
//      and text are refused; 95 is the whole exact wide body
//  H3  DSGB10_WIDE_EXACT_MT: 1 and 2 only, default 1; GQ: 0, 2, 4 and each plus 16, default 0; R16_EXACT: 0..63
//  H4  env_error names the first setting outside its range, with its text, and says 0 for a good set; unset settings are good
//  H5  the defaults are the present behaviour: every parser of an unset setting returns 0 (MT: 1)
// Every row has its control: a parser that accepts everything, or one that refuses one value too many, compiled into this test must
// FAIL the same check (reported as "control rejected").
// Command: exact_rule_test (exit 0 = all pass)
#include <cstdio>
#include <cstring>
#include <string>

#include "model/exact_rule.h"

namespace {

int fails = 0, checks = 0;
bool quiet = false;  // the controls' expected failures are not printed
void check(bool ok, const char* what, long a = 0, long b = 0) {
  ++checks;
  if (ok) return;
  if (++fails <= 40 && !quiet) std::printf("FAIL: %s (%ld, %ld)\n", what, a, b);
}

// the wrong rules the controls run
int accept_all(const char* e) { return e && *e ? std::atoi(e) : 0; }  // no range at all
int off_by_one_wide(const char* e) {                                   // refuses 95
  const int v = model::exact::wide_exact(e);
  return v == 95 ? -1 : v;
}

// each check runs against the real parser and the control: the control must fail it
template <class F>
void both(const char* what, F&& body) {
  const int before = fails;
  body(+[](const char* e) { return model::exact::wide_exact(e); }, false);
  const bool real_ok = fails == before;
  const int mid = fails, cmid = checks;
  quiet = true;
  body(accept_all, true);
  body(off_by_one_wide, true);
  quiet = false;
  const bool control_failed = fails > mid;
  fails = mid, checks = cmid;  // the controls' failures are expected: not counted
  check(control_failed, what);
  if (real_ok && control_failed) std::printf("control rejected: %s\n", what);
}

}  // namespace

int main() {
  using namespace model::exact;

  // H1
  check(norm_exact(nullptr) == 0 && norm_exact("") == 0 && norm_exact("0") == 0 && norm_exact("1") == 1, "H1 norm values");
  for (const char* bad : {"2", "-1", "01x", " 1", "1 ", "on", "10", "1.0"}) check(norm_exact(bad) == -1, bad);

  // H2
  int wide_bad = 0;
  for (int v = 0; v <= 200; ++v) {
    char t[16];
    std::snprintf(t, sizeof t, "%d", v);
    const bool allowed = v <= 95 && !(v & 32);
    if (allowed ? wide_exact(t) != v : wide_exact(t) != -1) ++wide_bad;
  }
  check(wide_bad == 0, "H2 wide values 0..200 (allowed: 0..95 without bit 32)", wide_bad);
  check(wide_exact("95") == 95 && wide_exact("95") == (kWideFamilies | kWideRowGroup), "H2 95 is the whole exact wide body");
  for (const char* bad : {"-1", "96", "127", "32", "33", "x", "9 5", "0x5f", "+95"}) check(wide_exact(bad) == -1, bad);
  check(wide_exact(nullptr) == 0 && wide_exact("") == 0, "H2 unset is 0");

  // H3
  check(wide_mt(nullptr) == 1 && wide_mt("") == 1 && wide_mt("1") == 1 && wide_mt("2") == 2, "H3 mt values");
  for (const char* bad : {"0", "3", "-1", "x", "12"}) check(wide_mt(bad) == -1, bad);
  for (const char* good : {"0", "2", "4", "16", "18", "20"}) check(wide_gq(good) == std::atoi(good), good);
  for (const char* bad : {"1", "3", "5", "6", "8", "17", "19", "32", "-2", "x"}) check(wide_gq(bad) == -1, bad);
  int r16_bad = 0;
  for (int v = 0; v <= 100; ++v) {
    char t[16];
    std::snprintf(t, sizeof t, "%d", v);
    if (v <= 63 ? r16_exact(t) != v : r16_exact(t) != -1) ++r16_bad;
  }
  check(r16_bad == 0, "H3 r16 values 0..100 (allowed 0..63)", r16_bad);

  // H4
  char msg[200];
  check(env_error(nullptr, nullptr, nullptr, nullptr, nullptr, msg, sizeof msg) == 0, "H4 all unset is good");
  check(env_error("1", "95", "2", "4", "0", msg, sizeof msg) == 0, "H4 the served set is good");
  check(env_error("2", "96", "3", "1", "64", msg, sizeof msg) == 1 && std::strstr(msg, "DSGB10_NORM_EXACT=2"), "H4 names the first", 0, 0);
  check(env_error("1", "96", "3", "1", "64", msg, sizeof msg) == 1 && std::strstr(msg, "DSGB10_WIDE_EXACT=96"), "H4 wide", 0, 0);
  check(env_error("1", "95", "3", "1", "64", msg, sizeof msg) == 1 && std::strstr(msg, "DSGB10_WIDE_EXACT_MT=3"), "H4 mt", 0, 0);
  check(env_error("1", "95", "2", "1", "64", msg, sizeof msg) == 1 && std::strstr(msg, "DSGB10_WIDE_EXACT_GQ=1"), "H4 gq", 0, 0);
  check(env_error("1", "95", "2", "4", "64", msg, sizeof msg) == 1 && std::strstr(msg, "DSGB10_R16_EXACT=64"), "H4 r16", 0, 0);

  // H5
  check(norm_exact(nullptr) == 0 && wide_exact(nullptr) == 0 && wide_gq(nullptr) == 0 && r16_exact(nullptr) == 0 && wide_mt(nullptr) == 1,
        "H5 the defaults are the present behaviour");

  // the controls: the wrong rules must fail the checks above
  both("control: no range (accept_all) is caught by 96 and 32", [](int (*f)(const char*), bool) {
    check(f("96") == -1, "96 refused");
    check(f("32") == -1, "32 refused");
  });
  both("control: refusing 95 is caught by 95", [](int (*f)(const char*), bool) { check(f("95") == 95, "95 accepted"); });

  std::printf("exact_rule_test: %d checks, %d failed\n", checks, fails);
  return fails != 0;
}
