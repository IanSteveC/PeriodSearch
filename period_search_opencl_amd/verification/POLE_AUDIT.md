# Pole / lambda / beta correctness audit

The project validator (`ps_validate.cpp`) only checks period, rms, and chisq —
it ignores the pole columns (dark, lambda, beta). Passing it therefore does
*not* by itself prove the pole solutions are physically correct. This note
records a separate, stronger audit of the pole physics.

## Why the output-file pole columns disagree with CPU at all

Each output line is a `best-of-N_POLES` (N_POLES=10) reduction: for one trial
frequency, ten work-groups each run the Levenberg–Marquardt fit from a
different starting pole, and the host writes the one with the smallest `dev`.
The LM objective for this problem is **non-convex with many near-degenerate
optima**, so which starting pole "wins" is hypersensitive to sub-ULP rounding.
Comparing only the winners (as the output file does) conflates two very
different things: whether the solver is correct, and which near-tie it broke.

## The decisive control: HYBRID vs REAL64

Three GPU builds run the *identical* kernels; only the number representation
differs:

- **REAL64** — `df = double`, native FP64 arithmetic (the oracle; VALID vs the
  CPU baseline).
- **HYBRID** — `df = float2` *storage*, but every op computed in native
  `double`. Differs from REAL64 **only** in storage rounding.
- **FP32** — `df = float2`, all arithmetic emulated in float (the real target).

Per-pole winner basin vs REAL64, at identical starting points (1820 trials,
standard 50-iter WU):

| build vs REAL64 | same basin | different basin |
|-----------------|-----------:|----------------:|
| HYBRID          | 63%        | 676 / 1820      |
| FP32            | 63%        | 670 / 1820      |
| FP32 vs HYBRID  | 97%        | 57 / 1820       |

HYBRID does *arithmetically-exact double* work and still lands in a different
basin than REAL64 on 37% of trials — the same rate as FP32. So the basin
choice is driven by storage rounding at the ULP level, i.e. it is a property
of the objective landscape, **not** of the df64 emulated arithmetic. FP32 and
HYBRID agree 97% because they share float2 storage. This is the crux: the
pole "disagreements" are landscape bistability that the pure-double oracle
exhibits just as much.

## What FP32 gets right, measured directly (not via argmax)

Using the `PS_DF_DEBUG` per-pole dump (`[pole ...]` lines) to compare every
pole trial against REAL64 at identical starts:

- **Same-basin endpoints agree tightly.** Fully converged (2000-iter cap,
  1e-6 stop): |Δλ| median 1.1°, p95 4.0°; |Δβ| median 1.4°, p95 4.2°;
  |Δdark| median 0.34; |Δdev|/dev median 4.5e-3. When FP32 and the oracle
  reach the same optimum, they land on the same pole to ~1°.
- **No systematic bias.** Median signed Δdev ≈ +0.4% (FP32 marginally less
  precise, as expected); on the per-line winners after merge, FP32's winner
  fits *better* than CPU's on roughly half the flipped lines.
- **No optimum is lost.** On all 39 argmax-flipped output lines (standard WU),
  the CPU baseline's winning pole is present in FP32's own 10-pole table within
  30° (including the mirror pole), at dev within ~1% of CPU's. FP32 explores
  the same set of optima; it only breaks near-ties differently.
- **Mirror-pole ambiguity** (λ+180°, −β) accounts for 8 of the standard-run
  winner flips — the classic convex-lightcurve inversion degeneracy, not an
  error.

## Reproduce

```
# build any mode with the per-pole dump
cd period_search && rm -f build/Release/Start_OpenCl.o
make CPPFLAGS=-DPS_DF_DEBUG                 # FP32
make "CPPFLAGS=-DPS_DF_DEBUG -DPS_REAL64"   # double oracle
cd ..
rm -f kernels.bin period_search_state period_search_out boinc_lockfile kernelSource.cl boinc_finish_called stderr.txt
./period_search/build/Release/period_search_BOINC_linux_amd_opencl_110_x64_Release >/dev/null 2>/dev/null
grep '^\[pole' stderr.txt > poles.txt   # per-(freq,pole) per/dev/dark/la/be
```

To probe convergence rather than the 50-iteration cap, set the WU's line-11
stop condition below 1 (e.g. `1e-6`) and raise `MAX_N_ITER` in `constants.h`.

## Conclusion

The pole/lambda/beta solutions are correct. Verified directly at the per-pole
level (not through the argmax the validator uses): FP32 converges to the same
poles as the double-precision oracle to ~1°, exhibits no directional bias, and
loses no optimum. The winner disagreements visible in the output file are
near-tie basin flips inherent to the non-convex fit — the pure-double HYBRID
build flips at the same rate — plus the standard mirror-pole inversion
ambiguity, neither of which is a df64 defect.
