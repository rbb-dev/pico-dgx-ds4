// The host test of tools/store_trace_check.py's `same` on a real failed-give-back trace (tools/store_trace_check_test.py): runs it and
// passes its exit code on. Needs python3.
#include <cstdio>
#include <cstdlib>
#include <string>

#ifndef ENGINE_SRC_DIR
#error "ENGINE_SRC_DIR names the engine's source directory"
#endif

int main() {
  const std::string cmd = std::string("python3 ") + ENGINE_SRC_DIR + "/tools/store_trace_check_test.py 2>&1";
  FILE* p = popen(cmd.c_str(), "r");
  if (!p) {
    std::printf("FAIL store_trace_check_test: python3 could not be started\n");
    return 1;
  }
  char b[4096];
  while (std::fgets(b, sizeof b, p)) std::fputs(b, stdout);
  const int rc = pclose(p);
  std::printf("%s store_trace_check_test: exit %d\n", rc == 0 ? "PASS" : "FAIL", rc);
  return rc == 0 ? 0 : 1;
}
