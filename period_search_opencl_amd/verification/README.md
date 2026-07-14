# FP32 (df64) verification assets

- `ocl_fp64_baseline.out` — 182-line reference output for the standard test WU
  (`/home/ian/builds/AST/period_search_out_win/period_search_in`, period range
  10.697377–10.9), produced by the FP64 CPU build.
- `ps_validate.cpp` — standalone harness wrapping the *exact* upstream
  Asteroids@Home validator logic (`period_search_validator4.cpp`): symmetric
  relative tolerance per line on period (0.1), rms (0.1) and chisq (0.5);
  pole columns are ignored by the project validator.

Build and use:

```
g++ -O2 -o ps_validate ps_validate.cpp
./ps_validate <result_file> <reference_file>
```

Status (2026-07-14, RX 6800 XT / gfx1030):

| build                        | verdict | worst margins (allowed 0.1 / 0.1 / 0.5)     |
|------------------------------|---------|---------------------------------------------|
| FP32 (df64, default `make`)  | VALID   | per 2.25e-05, rms 2.54e-03, chisq 5.09e-03  |
| `-DPS_HYBRID` diagnostic     | VALID   | per 2.21e-05, rms 2.55e-03, chisq 5.09e-03  |
| `-DPS_REAL64` diagnostic     | VALID   | per 3.05e-05, rms 4.16e-03, chisq 8.33e-03  |

Pole columns (dark/lambda/beta) are NOT validator-checked and do NOT all match
the FP64 result — see `POLE_AUDIT.md` for the full analysis. Exact-print
agreement is dark 102/181, lambda 77/182, beta 65/182; ~43 lines have lambda
and ~40 have beta off by >5° (a different pole won that line). The audit shows
this is objective-landscape bistability, not a df64 arithmetic defect (the
pure-double HYBRID build flips basins vs the double oracle at the same rate),
but it is a real open item: the per-pole `dev` gap (~0.4% median) has to close
before columns 4-6 line up with FP64. The global best line does match: line 51,
per 10.75313 vs 10.75309, pole 265/-36 vs 268/-35.
