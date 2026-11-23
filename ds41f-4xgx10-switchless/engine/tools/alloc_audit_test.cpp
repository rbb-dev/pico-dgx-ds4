// The host test of tools/alloc_audit.py: the tree and alloc_sites.txt agree, a call site added in a scratch copy is reported, a
// call removed from it is reported, and a call in a comment or a string is not a call site. Runs the script's own --selftest on
// the engine's sources (the directory compiled in) and passes its exit code on. Needs python3.
#include <cstdio>
#include <cstdlib>
#include <string>

#ifndef ENGINE_SRC_DIR
#error "ENGINE_SRC_DIR names the engine's source directory"
#endif

int main() {
  const std::string src = ENGINE_SRC_DIR;
  const std::string cmd = "python3 " + src + "/tools/alloc_audit.py --selftest --src " + src + " 2>&1";
  FILE* p = popen(cmd.c_str(), "r");
  if (!p) {
    std::printf("FAIL alloc_audit_test: python3 could not be started\n");
    return 1;
  }
  char b[4096];
  std::string last;
  while (std::fgets(b, sizeof b, p)) {
    std::fputs(b, stdout);
    last = b;
  }
  const int rc = pclose(p);
  std::printf("%s alloc_audit_test: exit %d\n", rc == 0 ? "PASS" : "FAIL", rc);
  return rc == 0 ? 0 : 1;
}
