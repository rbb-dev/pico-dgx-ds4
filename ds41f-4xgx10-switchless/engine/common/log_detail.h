#pragma once
// The detail lines: set once at start by main.cpp (every mode but serve, and serve with DSGB10_LOG_DETAIL=1, for the tests and
// benchmarks that read them). Serving without it prints a request's closing line, errors and rare events; the per-request detail,
// the per-step host timeline (cmd.h g_t_*), the spill, checkpoint and split-settle lines only with it.
inline bool g_log_detail = false;
