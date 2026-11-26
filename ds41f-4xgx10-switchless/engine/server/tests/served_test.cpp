// server/served.h against the list pico serves with (host only, no GPU): a `--mode serve` start with no settings takes every
// flag and environment variable of the list, a start that names one keeps its own, other modes take none, every name has a
// reader in the engine's source, and main.cpp applies the tables. Built with -DPICO_OLD_RULES the tables are empty and the
// cases fail (exit 1). Exit 0 when every case passes.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../served.h"

static int g_bad = 0, g_n = 0;
static void check(bool ok, const std::string& what) {
  ++g_n;
  if (!ok) ++g_bad, std::printf("FAIL: %s\n", what.c_str());
}

// The served ENGINE_ARGS and --setenv settings, as the settings file holds them; the machine's own (weights, abliteration,
// vision and confidence-network files) are not defaults and not in this list.
static const std::map<std::string, std::string> kFlagValues = {
    {"--max-ctx", "4194304"}, {"--slots", "32"},         {"--big-slots", "0"},           {"--pool-free-gib", "6.3"},
    {"--prefill-rows-max", "16384"}, {"--engram-vram", "on"}, {"--disk-gib", "512"},     {"--engram-cache-mib", "2032"},
    {"--front-mib", "840"},   {"--media-mib", "512"},    {"--pscan", "all"},             {"--kv-split", "tiered"},
    {"--keep-gib", "64"},
};
static const std::vector<std::string> kSwitches = {"--draft-head4", "--draft-markov8"};
static const std::map<std::string, std::string> kEnvValues = {
    {"DSGB10_COPY", "1"},          {"DSGB10_RING_RSBAL", "1"},  {"DSGB10_PF_TUNE", "hc,small,down"},
    {"DSGB10_BUDGET", "1"},        {"DSGB10_DRAFT1", "6"},      {"DSGB10_WIDE", "64"},
    {"DSGB10_WIDE_COST", "24,2.2,57,1.1"},                      {"DSGB10_HEAD4_C1", "1"},
    {"DSGB10_CF_ALPHA", "1.0"},    {"DSGB10_BATCH_CF", "1"},    {"DSGB10_BATCH_TRIM", "1"},
    {"DSGB10_BATCH_TRIM_COST", "19,2.9"},                       {"DSGB10_CERT", "1"},
    {"DSGB10_RING_DIRECT", "2"},   {"DSGB10_WIDE_SH", "1"},     {"DSGB10_WIDE_AR", "2"},
    {"DSGB10_HEADG", "2"},         {"DSGB10_COPY_ROWS", "64"},
};

using Args = std::vector<std::string>;

// The flags a start takes, as a map (a switch maps to "").
static std::map<std::string, std::string> taken(const Args& start) {
  std::map<std::string, std::string> m;
  const Args v = served::flags_for(start);
  for (size_t i = 0; i < v.size(); ++i) {
    const bool sw = i + 1 >= v.size() || v[i + 1].rfind("--", 0) == 0;
    m[v[i]] = sw ? "" : v[i + 1];
    if (!sw) ++i;
  }
  return m;
}

static std::string slurp(const std::filesystem::path& p) {
  std::ifstream f(p);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

int main() {
  // ---- a serve start with no settings
  {
    const auto m = taken({"--mode", "serve", "--socket", "/x"});
    for (const auto& [k, v] : kFlagValues) check(m.count(k) && m.at(k) == v, "serve default " + k + " " + v);
    for (const auto& k : kSwitches) check(m.count(k) && m.at(k).empty(), "serve default " + k);
    check(m.size() == kFlagValues.size() + kSwitches.size(), "no flag outside the list is defaulted");
    std::map<std::string, std::string> env;
    served::env_for(true, [&](const char* n, const char* v) { env.insert({n, v}); });
    for (const auto& [k, v] : kEnvValues) check(env.count(k) && env.at(k) == v, "serve default " + k + "=" + v);
    check(env.size() == kEnvValues.size(), "no variable outside the list is defaulted");
    check(env.count("DSGB10_CF_MLP") == 0, "the confidence network's file is the machine's, not a default");
  }
  // ---- other modes take none
  {
    check(taken({"--mode", "prefill"}).empty(), "prefill mode: no flag defaulted");
    check(taken({}).empty(), "no mode: no flag defaulted");
    bool any = false;
    served::env_for(false, [&](const char*, const char*) { any = true; });
    check(!any, "other modes: no variable defaulted");
  }
  // ---- a start that names a setting keeps it
  {
    const auto m = taken({"--mode", "serve", "--slots", "4", "--max-ctx", "262144", "--kv-split", "off", "--pscan", "off"});
    check(!m.count("--slots") && !m.count("--max-ctx") && !m.count("--kv-split") && !m.count("--pscan"), "named flags are not defaulted again");
    check(m.count("--disk-gib") && m.count("--front-mib"), "the others still are");
    const auto o = taken({"--mode", "serve", "--no-draft-head4", "--no-draft-markov8"});
    check(!o.count("--draft-head4") && !o.count("--draft-markov8"), "--no-draft-head4 / --no-draft-markov8 keep the drafters off");
    check(!taken({"--mode", "serve", "--pool-tokens", "1000"}).count("--pool-free-gib"), "--pool-tokens: no --pool-free-gib default");
    check(!taken({"--mode", "serve", "--prefill-rows", "4096"}).count("--prefill-rows-max"), "--prefill-rows: no second set by default");
    check(!taken({"--mode", "serve", "--draft-bf16"}).count("--draft-head4"), "--draft-bf16: no --draft-head4 default");
  }
  // ---- the environment: a variable already set keeps its value (the caller's set is setenv(name, value, 0))
  {
    std::map<std::string, std::string> env = {{"DSGB10_WIDE", "0"}, {"DSGB10_COPY", "0"}};
    served::env_for(true, [&](const char* n, const char* v) { env.insert({n, v}); });
    check(env.at("DSGB10_WIDE") == "0" && env.at("DSGB10_COPY") == "0", "a variable set to its off value stays off");
    check(env.at("DSGB10_CERT") == "1", "an unset one takes the served value");
  }
#ifdef SERVED_ENGINE_DIR
  // ---- every name has a reader, and main.cpp applies the tables and parses every flag
  {
    std::string all;
    for (const auto& e : std::filesystem::recursive_directory_iterator(SERVED_ENGINE_DIR)) {
      const auto ext = e.path().extension().string();
      if (!e.is_regular_file() || e.path().string().find("/tests/") != std::string::npos) continue;
      if (ext == ".cpp" || ext == ".h" || ext == ".cu" || ext == ".cuh") all += slurp(e.path());
    }
    for (const auto& [k, v] : kEnvValues) check(all.find("\"" + k + "\"") != std::string::npos, "the engine reads " + k);
    const std::string main_cpp = slurp(SERVED_MAIN_CPP);
    for (const auto& [k, v] : kFlagValues) check(main_cpp.find("s == \"" + k + "\"") != std::string::npos, "main.cpp parses " + k);
    for (const auto& k : kSwitches) {
      check(main_cpp.find("s == \"" + k + "\"") != std::string::npos, "main.cpp parses " + k);
      check(main_cpp.find("s == \"--no-" + k.substr(2) + "\"") != std::string::npos, "main.cpp parses --no-" + k.substr(2));
    }
    check(main_cpp.find("served::flags_for(") != std::string::npos && main_cpp.find("served::env_for(") != std::string::npos,
          "main.cpp applies the served tables");
  }
#endif
  if (g_bad) {
    std::printf("%d of %d cases failed\n", g_bad, g_n);
    return 1;
  }
  std::printf("%d cases ok\n", g_n);
  return 0;
}
