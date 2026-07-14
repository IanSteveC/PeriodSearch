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

Status (2026-07-14, RX 6800 XT / gfx1030 — see `POLE_AUDIT.md` for the full
analysis; `make` defaults to FP64, `make FP32=1` builds the triple-float
FP32 app, `make FP32=1 DF64=1` the faster double-float variant):

| build                           | verdict | worst margins (allowed 0.1 / 0.1 / 0.5)    | λ/β flips >5° | warm wall |
|---------------------------------|---------|--------------------------------------------|---------------|-----------|
| FP64 (default `make`)           | VALID   | per 2.33e-10, rms 0,       chisq 4.69e-08  | 0 / 0         | 13.5 s    |
| FP32 triple (`make FP32=1`)     | VALID   | per 2.52e-05, rms 9.59e-04, chisq 1.92e-03 | 10 / 11       | 45.9 s    |
| FP32 df64 (`… DF64=1`)          | VALID   | per 2.54e-05, rms 8.42e-04, chisq 1.68e-03 | 11 / 10       | 25.8 s    |
| `-DPS_HYBRID` diagnostic        | VALID   | per 2.52e-05, rms 8.47e-04, chisq 1.69e-03 | 10 / 11       |           |
| `-DPS_TRIPLE_HYBRID` diagnostic | VALID   | per 2.33e-10, rms 0,       chisq 4.69e-08  | 0 / 0         | 120 s     |

The FP64 build matches the reference pole columns **exactly** (182/182,
byte-identical to a pristine-kernel run), as does the TRIPLE_HYBRID
diagnostic (3-float storage, double compute — byte-identical to the FP64
build, which proves the storage-width theory). The float-only FP32 builds
match lambda on ~172/182 lines within 5°; the residual flips are chaotic
basin near-ties that re-break under ANY dense per-op rounding difference
from binary64: the triple build's per-op accuracy (~2^-63) beats native
double 1000x and tightens the same-basin dev gap 100x (median 1.2e-6 vs
df64's 1.3e-4), yet the basin-flip rate stays ~2% because accuracy is not
bit-reproducibility. Exact FP32 pole parity would require bit-exact software
binary64 emulation. The global best line matches exactly in every mode:
line 51, `10.75308538 (268,-35)`.
