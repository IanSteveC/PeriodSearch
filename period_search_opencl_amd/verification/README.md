# FP32 (df64) verification assets

- `ocl_fp64_baseline.out` — 182-line reference output for the standard test WU
  (`/home/ian/builds/AST/period_search_out_win/period_search_in`, period range
  10.697377–10.9). Identical to `~/builds/AST/ocl_goldens/golden_rank8_182.txt`;
  independent FP64 implementations (stock CUDA, rank8 CUDA, Windows CUDA on an
  RTX 3080 Ti, the pre-port OpenCL FP64 kernels on the RX 6800 XT) reproduce it
  to exact print on the pole columns and to the last digit or so elsewhere.
- `ps_validate.cpp` — standalone harness wrapping the *exact* upstream
  Asteroids@Home validator logic (`period_search_validator4.cpp`): symmetric
  relative tolerance per line on period (0.1), rms (0.1) and chisq (0.5);
  pole columns are ignored by the project validator.
- `cmp_poles.py` — column-by-column comparison (exact-print agreement plus
  lambda/beta winner flips >5°); `-v` lists the flipped lines.

Build and use:

```
g++ -O2 -o ps_validate ps_validate.cpp
./ps_validate <result_file> <reference_file>
python3 cmp_poles.py <result_file> <reference_file> [-v]
```

Status (2026-07-14, RX 6800 XT / gfx1030, after the df_f-truncation fix —
see `POLE_AUDIT.md`; `make` now defaults to FP64, `make FP32=1` opts into
df64):

| build                        | verdict | worst margins (allowed 0.1 / 0.1 / 0.5)    | λ/β flips >5° |
|------------------------------|---------|--------------------------------------------|---------------|
| FP64 (default `make`)        | VALID   | per 2.33e-10, rms 0,       chisq 4.69e-08  | 0 / 0         |
| FP32 (`make FP32=1`, df64)   | VALID   | per 1.34e-05, rms 1.57e-03, chisq 3.14e-03 | 11 / 9        |
| `-DPS_HYBRID` diagnostic     | VALID   | per 2.52e-05, rms 8.47e-04, chisq 1.69e-03 | 10 / 11       |

The FP64 build matches the reference pole columns **exactly** (182/182 on
dark, lambda and beta; byte-identical to a pristine-kernel run). FP32 matches
lambda on 171/182 lines within 5°; the 11 residual flips are near-ties
re-broken at the float2 storage precision (HYBRID, with exact double
arithmetic, flips at the same rate) — analysis in `POLE_AUDIT.md`. The global
best line matches exactly: line 51, `10.75308538 (268,-35)` in both FP32 and
the baseline.
