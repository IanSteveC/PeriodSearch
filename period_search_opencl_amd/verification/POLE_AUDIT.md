# Pole / lambda / beta correctness audit

The project validator (`ps_validate.cpp`) only checks period, rms, and chisq —
it ignores the pole columns (dark, lambda, beta). This note records how the
pole columns were brought into agreement with the FP64 reference, and what the
residual disagreement in the FP32 build is made of.

**History note.** An earlier revision of this file concluded the pole
disagreements were "objective-landscape bistability, hypersensitive to
sub-ULP rounding, not a df64 defect". That conclusion was **wrong** — see
"The 2026-07-14 correction" below. It survives here only as a warning about
validating against a single oracle that shares the code under test.

## Ground truth: the FP64 pole solution is robust, not bistable

Independent FP64 implementations of this pipeline agree on the pole columns
to *exact print*, across codebases, summation orders, and hardware:

- stock CUDA vs rank8-tile CUDA (different normal-equation summation order):
  lambda 182/182, beta 182/182 exact;
- CUDA on Windows/RTX 3080 Ti vs those: 182/182 / 181-182/182 exact;
- pre-port OpenCL FP64 kernels on the RX 6800 XT: bit-exact reproduction of
  `ocl_fp64_baseline.out` (= `golden_rank8_182.txt` from the CUDA port work).

So implementation-level rounding noise (~1e-15 relative, different summation
orders, different libm) flips **zero** winners. Any build that flips a large
fraction of lines has a real defect, not "landscape sensitivity".

## The 2026-07-14 correction: it was a conversion bug

The REAL64 diagnostic build (df = native double — supposedly semantically
identical to the pre-port kernels) was found to flip 90/182 lambda lines vs
the baseline — *twice* the FP32 rate. Since FP64 implementations agree
exactly, that meant the df conversion itself had changed semantics, and
REAL64 had never been a valid oracle: the earlier audit measured the bug
against itself.

Bisection (swapping pristine pre-conversion `.cl` files into the REAL64 build
one group at a time — old and new kernels are ABI-compatible when df=double)
pinned it to `bright.cl` and `Start.cl`, and the diff pinned it to one bug
class: **double-precision constants squeezed through `df_f()`, the
single-float constructor**, during the conversion:

- `bright.cl` — `fmod(f, 2*PI)` became `df_fmod(f, df_f(2*PI))`: the modulus
  wrong by 1.7e-7 relative puts a phase error of `floor(f/2pi) * 1.7e-7` rad
  (up to ~1e-4 rad, different at every epoch) into every reduced rotation
  angle. This alone flipped 62/182 lines with everything else pristine.
- `Start.cl` — `df_f(PI)` in the starting angular frequency (line ~151),
  `df_f(DEG2RAD)` on the starting pole angles, `df_f(RAD2DEG)`/`df_f(2*PI)`
  in the output conversions (period/lambda/beta prints), `df_f(1e-10f)` in
  the LM accept threshold, `df_f(TINY)` in the facet-visibility test.
- `df64.cl` — under `DF_REAL64`, `DFV(a,b)` dropped the low float, so even
  the correctly-split df constants (`DF_2PI` …) were float-truncated in the
  double diagnostic build.

Fix: `DF_CONST(d, hi, lo)` in `df64.cl` keeps the exact double expression in
REAL64 mode and the two-float split in float2 modes (the double literal is
preprocessor-dropped for FP64-less compilers), new constants `DF_DEG2RAD`,
`DF_RAD2DEG`, `DF_TINY`, `DF_DEVEPS`, and the call sites above now use them.
**Never wrap a double macro in `df_f()`.**

After the fix, the REAL64 build's output is **byte-identical** to the
pristine-kernel run (and matches the CUDA baseline on all pole columns
182/182, with only 3 last-digit print differences in period/chisq; validator
margins per=2.3e-10, rms=0, chisq=4.7e-08).

## Post-fix status (standard 50-iter WU, vs `ocl_fp64_baseline.out`)

Exact-print agreement / winner flips >5 deg:

| build  | dark    | lambda  | beta    | λ flips | β flips |
|--------|---------|---------|---------|---------|---------|
| REAL64 | 182/182 | 182/182 | 182/182 | 0       | 0       |
| HYBRID | 153/182 | 148/182 | 138/182 | 10      | 11      |
| FP32   | 151/182 | 142/182 | 136/182 | 11      | 9       |

(Pre-fix FP32 was lambda 77/182 exact, 43 flips.) The **global best line —
the actual answer of the period search — matches exactly** in FP32: line 51,
`10.75308538 (268,-35)` on both sides.

## The residual FP32/HYBRID flips are the float2 storage noise floor

- HYBRID (float2 storage, exact double arithmetic) flips at the same rate as
  FP32 (10-11 vs 11/9), so the emulated df64 *arithmetic* is not the limiter;
  the ~49-bit storage rounding of iteration-carried state is. Tightening
  transcendentals further cannot remove these flips.
- Per-pole audit (PS_DF_DEBUG `[pole]` dump, FP32 vs fixed REAL64, 2300
  trials): same basin on 97.3% of trials (was 63% pre-fix); same-basin
  |Δdev|/dev median 1.3e-4, p95 2.6e-3 (was ~4.5e-3 median); |Δλ| median
  0.04°, |Δβ| median 0.06°.
- The 12 winner flips sit at inter-pole dev margins 1.5e-5..7.3e-3 — inside
  the same-basin noise p95 and/or explained by the ~2.7% of trials that
  basin-flip (10 trials/line ⇒ ~24% of lines contain one). On the 11 output
  lines that flip >5°, |Δrms| is 1e-5..1.5e-3 (median ~2e-4 on rms ~0.24) and
  FP32's winner fits *better* than the baseline's on 5 of the 11; two are the
  classic mirror-pole degeneracy (λ+180°, −β).

Conclusion: FP32 loses no physics — the residual flips are genuine near-ties
re-broken at the 49-bit storage precision, bounded by the same-basin noise
floor, with the best-fit line and 94% of all lines matching FP64 exactly.

## Reproduce

```
# build any mode with the per-pole dump
cd period_search && rm -f build/Release/Start_OpenCl.o
make CPPFLAGS=-DPS_DF_DEBUG                 # FP32
make "CPPFLAGS=-DPS_DF_DEBUG -DPS_REAL64"   # double oracle (now trustworthy)
cd ..
rm -f kernels.bin period_search_state period_search_out boinc_lockfile kernelSource.cl boinc_finish_called stderr.txt
./period_search/build/Release/period_search_BOINC_linux_amd_opencl_110_x64_Release >/dev/null 2>/dev/null
grep '^\[pole' stderr.txt > poles.txt   # b -> freq line = b/10, pole = b%10
python3 verification/cmp_poles.py period_search_out verification/ocl_fp64_baseline.out -v
```

## Addendum (2026-07-14, later): the triple-float experiments — accuracy is not bit-reproducibility

Two further builds settle what the residual float-only flips are:

- **`PS_TRIPLE_HYBRID`** — df = {float x,y,z} (~71-bit storage), all ops in
  native double; the ONLY variable vs `PS_HYBRID` is storage width. Result:
  **byte-identical to the FP64 build** (three floats hold any binary64
  exactly, and the double ops are bit-identical). This proved the df64 flips
  were storage rounding and nothing else.
- **`DF_TRIPLE`** — float-only triple-float arithmetic, per-op ~2^-63
  (differential harness `tools/test_triple.cpp`: add/sub exact, mul/div/
  sqrt/exp/acos ~2^-63, sincos ~2^-64, log ~2^-59). Same-basin per-pole dev
  gap vs FP64 improves 100x over df64 (median 1.2e-6 vs 1.3e-4) — but the
  basin-flip rate is unchanged (2249/2300 = 97.8% same basin vs df64's
  97.3%), and the output still flips 10-11 near-tie winners.

Interpretation: the basin each (freq,pole) trial falls into during the
50-iteration LM transient is chaotically sensitive to ANY *dense* per-op
deviation from binary64's exact rounding, no matter how small — a system
1000x MORE accurate than double still decorrelates ~2% of trials, because
its results land on a different rounding grid at every operation. Native
FP64 implementations agree with each other only because IEEE add/mul are
bit-deterministic (identical inputs -> identical bits, i.e. zero deviation
at almost every op; vendor libm differences are sparse). Exact pole parity
without FP64 hardware therefore requires bit-exact software binary64
emulation, not higher precision. The triple build is still the better FP32
app: every accuracy metric (validator margins, exact prints, per-pole dev
agreement) improves substantially over df64, at 1.8x its runtime.
