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

Pole columns (dark/lambda/beta, ignored by the validator) were checked
separately: 135/182 lines select the identical pole (la/be within 5 deg),
8 lines land on the mirror pole (la+180, -be — the classic lightcurve-inversion
ambiguity), and the remaining 39 are near-tie basin flips where the two devs
differ only in the 3rd-4th decimal (FP32's pick fits *better* in 22 of them).
The pure-double `-DPS_REAL64` build flips *more* lines (99) than FP32 (47),
so the flips are optimizer degeneracy, not FP32 precision loss. The global
best line matches: line 51, per 10.75313 vs 10.75309, pole 265/-36 vs 268/-35.
