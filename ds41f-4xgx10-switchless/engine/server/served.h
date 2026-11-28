// The settings pico serves with, as the defaults of `--mode serve`. A start with no settings runs the measured configuration:
// every flag and environment variable below that the command line (or the start's environment) does not name takes the value
// here. Other modes (the golden checks, the benches) take none of them unless the start says --served: then they take every
// variable below and the flags the table marks (see "--served"). Every one can still be set to another value, and
// each switch is turned off with its off value (DSGB10_X=0, --no-draft-head4, --kv-split off, --pscan off, --engram-vram off).
// What describes one machine is not here and stays in the settings file: the weights, abliteration and vision files, the
// confidence network's file (DSGB10_CF_MLP; without it DSGB10_BATCH_CF and the one-session confidence gate stay idle), addresses.
// Built with -DPICO_OLD_RULES the tables are empty: a start with no settings takes the engine's bare defaults, and
// tests/served_test.cpp fails.
#pragma once
#include <cstring>
#include <string>
#include <vector>

namespace served {

struct Flag {
  const char* name;
  const char* value;  // null: a switch with no value (--draft-head4; turned off by --no-draft-head4)
  bool tests;         // taken by a test mode that says --served (see the table below)
};
struct Env {
  const char* name;
  const char* value;
};

// --served: a test mode (bench, decode, spectf, ...) runs the served engine's settings. Every variable of kEnv applies, the way
// it does to serve (a start's own environment or --setenv wins). Of the flags, per entry:
//  --max-ctx --slots --big-slots   no: the size of the session slots; a test names its own (serve sizes them beside the
//                                  chunk pool)
//  --pool-free-gib --prefill-rows-max --disk-gib --front-mib --media-mib --keep-gib --kv-split
//                                  no: the chunk pool, the second prefill set, the disk tier, the front-end's and media caps,
//                                  the kept conversations and the split imager exist only in serve (the parser zeroes or
//                                  ignores them in every other mode)
//  --engram-vram --engram-cache-mib  yes: the other modes make a fixed Engram row cache of this size, and its first slots sit
//                                  in the display reservation, as in serve
//  --pscan all --draft-head4 --draft-markov8  yes: model options that change the step graphs and the draft pass in every mode
//                                  (the drafters act where the mode drafts)
#ifndef PICO_OLD_RULES
inline constexpr Flag kFlags[] = {
    {"--max-ctx", "4194304", false},        {"--slots", "32", false},               {"--big-slots", "0", false},
    {"--pool-free-gib", "6.3", false},      {"--prefill-rows-max", "16384", false}, {"--engram-vram", "on", true},
    {"--disk-gib", "512", false},           {"--draft-head4", nullptr, true},       {"--draft-markov8", nullptr, true},
    {"--engram-cache-mib", "2032", true},   {"--front-mib", "840", false},          {"--media-mib", "512", false},
    {"--pscan", "all", true},               {"--kv-split", "tiered", false},        {"--keep-gib", "64", false},
};
inline constexpr Env kEnv[] = {
    {"DSGB10_COPY", "1"},          {"DSGB10_RING_RSBAL", "1"},       {"DSGB10_PF_TUNE", "hc,small,down"},
    {"DSGB10_BUDGET", "1"},        {"DSGB10_DRAFT1", "6"},           {"DSGB10_WIDE", "64"},
    {"DSGB10_WIDE_COST", "24,2.2,57,1.1"},                           {"DSGB10_HEAD4_C1", "1"},
    {"DSGB10_CF_ALPHA", "1.0"},    {"DSGB10_BATCH_CF", "1"},         {"DSGB10_BATCH_TRIM", "1"},
    {"DSGB10_BATCH_TRIM_COST", "19,2.9"},                            {"DSGB10_CERT", "1"},
    {"DSGB10_RING_DIRECT", "2"},   {"DSGB10_WIDE_SH", "1"},          {"DSGB10_WIDE_AR", "2"},
    {"DSGB10_HEADG", "2"},         {"DSGB10_COPY_ROWS", "64"},       {"DSGB10_REUSE_EXACT", "1"},
};
#else
inline constexpr Flag kFlags[1] = {{nullptr, nullptr, false}};
inline constexpr Env kEnv[1] = {{nullptr, nullptr}};
#endif

inline bool has(const std::vector<std::string>& args, const char* flag) {
  for (const std::string& a : args)
    if (a == flag) return true;
  return false;
}

// A default stays out when the start names the flag, names its off form, or names a flag it cannot be combined with:
//  --pool-free-gib: with --pool-tokens the pool is sized by positions (both given: the smaller; one given: that one)
//  --prefill-rows-max: with --prefill-rows the start chose its own chunk (the second set is its to ask for)
//  --draft-head4: with --draft-bf16 (a start that refuses the pair)
inline bool given(const std::vector<std::string>& args, const Flag& f) {
  if (has(args, f.name)) return true;
  if (!f.value && has(args, (std::string("--no-") + (f.name + 2)).c_str())) return true;
  const std::string n = f.name;
  if (n == "--pool-free-gib") return has(args, "--pool-tokens");
  if (n == "--prefill-rows-max") return has(args, "--prefill-rows");
  if (n == "--draft-head4") return has(args, "--draft-bf16");
  return false;
}

inline bool serves(const std::vector<std::string>& args) {
  for (size_t i = 0; i + 1 < args.size(); ++i)
    if (args[i] == "--mode") return args[i + 1] == "serve";
  return false;
}

// Whether the served environment applies: a serve start, or any start that says --served.
inline bool applies(const std::vector<std::string>& args) { return serves(args) || has(args, "--served"); }

// The arguments to put in front of a serve start's own: every served flag the start does not name. (A later flag of the
// start wins over an earlier one, but none repeats, so the prefix and the start's flags never meet.) Outside serve it is empty,
// or with --served the flags the table marks.
inline std::vector<std::string> flags_for(const std::vector<std::string>& args) {
  std::vector<std::string> out;
  const bool serve = serves(args);
  if (!serve && !has(args, "--served")) return out;
  for (const Flag& f : kFlags) {
    if (!f.name || given(args, f) || (!serve && !f.tests)) continue;
    out.push_back(f.name);
    if (f.value) out.push_back(f.value);
  }
  return out;
}

// The served environment: set(name, value) for every variable the process has not set itself (the caller's set keeps a
// variable that is already in the environment, --setenv's included).
template <class Set>
inline void env_for(bool serve_mode, Set set) {
  if (!serve_mode) return;
  for (const Env& e : kEnv)
    if (e.name) set(e.name, e.value);
}

}  // namespace served
