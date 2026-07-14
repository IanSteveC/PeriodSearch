# OpenCL FP32 (df64) port — STATUS: validated, pole columns match FP64 to the storage noise floor

The FP32 port of the OpenCL `period_search` app (branch `opencl-fp32`) runs
the df64 kernels on the RX 6800 XT with no hardware FP64, passes the project
validator, and — after the 2026-07-14 constant-truncation fix — matches the
FP64 reference on the pole columns on 171/182 lines (the 11 remaining
differences are genuine near-ties; the REAL64 diagnostic build now matches
**exactly**, 182/182). See `verification/README.md` for the matrix and
`verification/POLE_AUDIT.md` for the full analysis.

## Bottom line

- period / rms / chisq: VALID, worst margins per 1.34e-05 (tol 0.1),
  rms 1.57e-03 (0.1), chisq 3.14e-03 (0.5).
- dark / lambda / beta: REAL64 182/182 exact print; FP32 exact print
  151/142/136, winner flips >5° on 11 lines (down from 43 before the fix).
  The global best line — the actual answer — matches exactly:
  line 51, `10.75308538 (268,-35)`.
- The residual FP32 flips are the ~49-bit float2 storage noise floor, NOT an
  arithmetic bug: HYBRID (float2 storage, exact double compute) flips at the
  same rate. Per-pole: 97.3% of the 2300 trials land in the same basin as the
  double oracle; same-basin |Δdev|/dev median 1.3e-4.

## The 2026-07-14 finding: `df_f()` constant truncation (read this first)

The previous session's conclusion ("pole disagreements are landscape
bistability, not a defect") was **wrong**. Independent FP64 implementations
(stock CUDA, rank8 CUDA, Windows CUDA/3080 Ti, pre-port OpenCL kernels) agree
on the pole columns to *exact print* — the FP64 winner is robust, and the
REAL64 diagnostic build (which flipped 90/182 lines!) was itself carrying the
port bug, so the old per-pole audit validated the bug against itself.

The bug class: the df conversion wrapped **double**-precision constants in
`df_f()`, the *single-float* constructor. Worst offenders:

- `bright.cl`: `df_fmod(f, df_f(2*PI))` — modulus wrong by 1.7e-7 relative
  ⇒ phase error `floor(f/2pi)*1.7e-7` rad (up to ~1e-4 rad, epoch-dependent)
  in every reduced rotation angle. 62/182 flips on its own.
- `Start.cl`: `df_f(PI)` in the starting angular frequency, `df_f(DEG2RAD)`
  on the starting pole, `df_f(RAD2DEG)`/`df_f(2*PI)` in the printed
  period/lambda/beta, `df_f(1e-10f)` accept threshold, `df_f(TINY)`.
- `df64.cl`: under `DF_REAL64`, `DFV(a,b)` dropped the low float — even the
  properly-split df constants were truncated in the double diagnostic mode.

Fix (committed on this branch): `DF_CONST(d, hi, lo)` — full double in
REAL64, two-float split otherwise (the double literal is dropped by the
preprocessor in float2 modes, so FP64-less compilers never see it); new
constants `DF_DEG2RAD`, `DF_RAD2DEG`, `DF_TINY`, `DF_DEVEPS`; call sites
fixed. Verified by swapping pristine pre-conversion kernels into the REAL64
build (they are ABI-compatible when df=double): the fixed build's output is
**byte-identical** to the pristine run. **Rule: never wrap a double macro in
`df_f()` — use a `DF_CONST` constant.**

How it was found, for the next hunt: bisect by swapping pristine `.cl` files
(from `git show '100387e~1:./<file>'`) into the build group-by-group, then
file-by-file — under `PS_REAL64` old and new kernels link fine. The harness
script pattern is in `verification/POLE_AUDIT.md` ("Reproduce").

## Earlier session context (still true)

- The app compiles `kernels.cpp` (generated from the `.cl` files by
  `gen_kernels_cpp.py`, kept in sync by the Makefile), NOT the `.cl` files at
  runtime. Pack every host upload in BOTH `ClPrecalc` and `ClStart`
  (`psPackWrite`); the pack is a no-op under `PS_REAL64`.
- First LM iteration's `mrqcof2` explodes to ~1e16 for block 0 and is
  rejected; identical in REAL64/pristine, recovers fine (pre-existing).
- `ClCalculatePrepare`/`PreparePole`/`Iter1Begin` write some `freq_result`
  flags through the base pointer instead of per-block `CUDA_LFR` —
  pre-existing (commit `5cec596`), host merge only needs `isReported`.
- `df_f(1e40f)` sentinels become `inf` in float2 modes (`dev_best`,
  `iter_diff` inits, `Start.cl:67,191`). Benign for the loop logic, still
  worth cleaning up.

## Build / run / validate

```
cd period_search
make                          # FP32 (df64) — the real target
make CPPFLAGS=-DPS_REAL64     # diagnostic: same kernels, df=double, no pack
make CPPFLAGS=-DPS_HYBRID     # diagnostic: float2 storage, double compute
make CPPFLAGS=-DPS_DF_DEBUG   # FP32 + readback dumps incl. the per-(freq,pole)
                              # freq_result table (grep '^\[pole' stderr.txt)
                              # (rm -f build/Release/Start_OpenCl.o first when
                              #  switching CPPFLAGS — make can't see the change)

cd ..
rm -f kernels.bin period_search_state period_search_out boinc_lockfile kernelSource.cl boinc_finish_called stderr.txt
cp /home/ian/builds/AST/period_search_out_win/period_search_in period_search_in
./period_search/build/Release/period_search_BOINC_linux_amd_opencl_110_x64_Release
g++ -O2 -o /tmp/ps_validate verification/ps_validate.cpp   # first time only
/tmp/ps_validate period_search_out verification/ocl_fp64_baseline.out
python3 verification/cmp_poles.py period_search_out verification/ocl_fp64_baseline.out
```

Gotchas: BOINC *appends* to `./stderr.txt` (delete between runs);
`kernels.bin` is a compiled-kernel cache and `period_search_state` a
checkpoint — stale copies cause stale binaries or a resume-past-the-end
no-op. Test on the AMD RX 6800 XT only (the V100 is in use by the owner).

## Remaining work (next session)

1. **FP32/FP64 build toggle** like the CUDA/HIP apps: single codebase,
   default FP64, `FP32=1` opts in. All kernel arithmetic is already `df_*`
   calls, so this is: promote the `PS_REAL64` diagnostic to the supported
   FP64 build (host packs are already no-ops under `PS_REAL64`) and gate on
   a build flag.
2. Clean up the `1e40f`→inf sentinels (`Start.cl:67,191`) — use a large
   finite df constant so float2 modes don't carry inf through `dev_best`/
   `iter_diff`.
3. Optional: fix the base-pointer `CUDA_FR` flag writes in `Start.cl`
   (pre-existing, pre-port).
4. NOT worth pursuing for pole agreement: tightening `df_acos`/transcendental
   accuracy. HYBRID proves the residual 10-11 winner flips come from float2
   *storage* rounding of iteration-carried state, which transcendental
   accuracy cannot fix. (Tightening `df_acos` may still be worth it for
   general robustness on other WUs, but don't expect the flip count to move.)
