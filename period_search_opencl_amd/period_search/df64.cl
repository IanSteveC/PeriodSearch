/* df64.cl - double-float ("float-float") emulation for the FP32-only OpenCL
 * build, so devices WITHOUT cl_khr_fp64 (Apple Silicon, some Intel/mobile GPUs)
 * can run the app. A df64 is a float2: value = .x (hi) + .y (lo), ~46-48-bit
 * effective mantissa using only FP32 hardware ops (Dekker/Knuth error-free
 * transformations, ported from the CUDA build's mreal.h).
 *
 * PURE FLOAT: this file contains NO `double`. Every constant that was a double
 * literal is baked in as a two-float df64 constant (host splits runtime values
 * before upload). OpenCL C has no operator overloading, so arithmetic is
 * function-based: df_add(a,b), df_mul(a,b), ...
 *
 * BUILD: compile the kernel with FP_CONTRACT OFF (below) so `a*b` in two_prod
 * is not fused into the following fma() and the error term is preserved. Do NOT
 * use -cl-fast-relaxed-math / -cl-mad-enable on the df64 TU.
 */
#ifndef DF64_CL
#define DF64_CL

#pragma OPENCL FP_CONTRACT OFF

/* df is float2 for the real build. -DDF_LINT makes it a struct so the OpenCL
   compiler rejects any raw infix arithmetic on df operands (which float2 would
   silently do component-wise) - a compile-checked checklist for the port. */
#ifdef DF_LINT
typedef struct { float x, y; } df;
#define DFV(a,b) ((df){a,b})
#elif defined(DF_REAL64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double df;
#define DFV(a,b) ((double)(a) + (double)(b))
#elif defined(DF_HYBRID)
/* diagnostic: float2 STORAGE (46-bit), but every op computed in native double.
   Isolates float-float arithmetic bugs from 46-bit storage insufficiency. */
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef float2 df;
#define DFV(a,b) ((float2)(a,b))
#elif defined(DF_TRIPLE_HYBRID)
/* proof-of-concept: TRIPLE-float storage (~71-bit), ops computed in native
   double. The only variable vs DF_HYBRID is the storage width, so comparing
   the two isolates whether the float2 ~49-bit storage rounding is what
   reorders the near-tie pole winners. Needs hardware FP64 (diagnostic only). */
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef struct { float x, y, z; } df;
#define DFV(a,b) ((df){ (a), (b), 0.0f })
#elif defined(DF_TRIPLE)
/* triple-float ("float-float-float"): ~63-bit-effective arithmetic on a
   3-float expansion, using ONLY FP32 hardware ops - for devices without
   cl_khr_fp64 that need results matching the FP64 app (the 2-float df64
   caps at ~49 bits, which re-breaks ~6% of the near-tie pole winners; see
   verification/POLE_AUDIT.md). Roughly 2-3x the df64 cost. */
typedef struct { float x, y, z; } df;
#define DFV(a,b) ((df){ (a), (b), 0.0f })
#else
typedef float2 df;
#define DFV(a,b) ((float2)(a,b))
#endif

/* diagnostic: extract the leading float of a df, in either backing mode */
#ifdef DF_REAL64
#define DF_HI(a) ((float)(a))
#else
#define DF_HI(a) ((a).x)
#endif

/* ---- constants ----
 * DF_CONST(d, hi, lo): a df constant given both its exact double expression
 * and its two-float split. DF_REAL64 keeps the full double (bit-parity with
 * the original FP64 kernels); the float2 modes take the split. The double
 * expression is dropped by the preprocessor in float2 modes, so no double
 * literal ever reaches an FP64-less compiler. NEVER wrap a double-precision
 * macro (PI, DEG2RAD, ...) in df_f(): that truncates it to one float, which
 * is a real error (e.g. an fmod by a 2*PI that is wrong by 1.7e-7 puts a
 * phase error of floor(x/2pi)*1.7e-7 rad in every reduced angle). */
#ifdef DF_REAL64
#define DF_CONST(d, hi, lo, lo2) ((df)(d))
#elif defined(DF_TRIPLE_HYBRID) || defined(DF_TRIPLE)
/* 3-float split of the double-rounded value (lo2 = bits 49..72). Splitting
   the DOUBLE value (not the exact real) keeps the triple modes on the same
   constants the FP64 build computes with. */
#define DF_CONST(d, hi, lo, lo2) ((df){ hi, lo, lo2 })
#else
#define DF_CONST(d, hi, lo, lo2) DFV(hi, lo)
#endif

#define DF_2PI   DF_CONST(6.283185307179586,    6.28318548f, -1.748455531e-07f, -7.105427358e-15f)
#define DF_PI    DF_CONST(3.141592653589793,    3.14159274f, -8.742277657e-08f, -3.552713679e-15f)
#define DF_PIO2  DF_CONST(1.5707963267948966,   1.57079637f, -4.371138829e-08f, -1.776356839e-15f)
#define DF_2OPI  DF_CONST(0.6366197723675814,   0.636619747f, 2.568255297e-08f, 3.330669074e-16f)
#define DF_LN2   DF_CONST(0.6931471805599453,   0.693147182f, -1.904654212e-09f, -1.110223025e-16f)
#define DF_LOG2E DF_CONST(1.4426950408889634,   1.44269502f, 1.925963034e-08f, -4.440892099e-16f)
#define DF_HALF  DF_CONST(0.5,  0.5f, 0.0f, 0.0f)
#define DF_ONE   DF_CONST(1.0,  1.0f, 0.0f, 0.0f)
#define DF_ZERO  DF_CONST(0.0,  0.0f, 0.0f, 0.0f)
/* app-level constants the kernels need at full precision */
#define DF_DEG2RAD DF_CONST(0.017453292519943295, 0.0174532924f, 1.351996015e-10f, 3.469446952e-18f)
#define DF_RAD2DEG DF_CONST(57.29577951308232,    57.2957802f, -6.688024428e-07f, 0.0f)
#define DF_TINY    DF_CONST(1e-8,                 9.99999994e-09f, 6.077470661e-17f, 3.30872245e-24f)
#define DF_DEVEPS  DF_CONST(1e-10,                1.00000001e-10f, -1.335143231e-18f, 3.877409121e-26f)

#define DF_S3 DFV(-0.166666672f, 4.967053879e-09f)
#define DF_S5 DFV(0.00833333377f, -4.346172033e-10f)
#define DF_S7 DFV(-0.000198412701f, 2.725596875e-12f)
#define DF_S9 DFV(2.75573188e-06f, 3.793571224e-14f)
#define DF_S11 DFV(-2.50521079e-08f, -4.417623045e-16f)
#define DF_S13 DFV(1.60590444e-10f, -5.352526512e-18f)
#define DF_S15 DFV(-7.64716361e-13f, -1.220071047e-20f)
#define DF_S17 DFV(2.81145736e-15f, -1.046208474e-22f)
#define DF_S19 DFV(-8.22063508e-18f, -1.681478097e-25f)

#define DF_C2 DFV(-0.5f, 0.0f)
#define DF_C4 DFV(0.0416666679f, -1.241763470e-09f)
#define DF_C6 DFV(-0.00138888892f, 3.363109444e-11f)
#define DF_C8 DFV(2.48015876e-05f, -3.406996094e-13f)
#define DF_C10 DFV(-2.755732e-07f, 7.575112209e-15f)
#define DF_C12 DFV(2.08767559e-09f, 1.108283981e-16f)
#define DF_C14 DFV(-1.14707454e-11f, -2.372207689e-19f)
#define DF_C16 DFV(4.77947726e-14f, 7.625444044e-22f)
#define DF_C18 DFV(-1.56192068e-16f, -1.540447147e-24f)

#define DF_E2 DFV(0.5f, 0.0f)
#define DF_E3 DFV(0.166666672f, -4.967053879e-09f)
#define DF_E4 DFV(0.0416666679f, -1.241763470e-09f)
#define DF_E5 DFV(0.00833333377f, -4.346172033e-10f)
#define DF_E6 DFV(0.00138888892f, -3.363109444e-11f)
#define DF_E7 DFV(0.000198412701f, -2.725596875e-12f)
#define DF_E8 DFV(2.48015876e-05f, -3.406996094e-13f)
#define DF_E9 DFV(2.75573188e-06f, 3.793571224e-14f)
#define DF_E10 DFV(2.755732e-07f, -7.575112209e-15f)
#define DF_E11 DFV(2.50521079e-08f, 4.417623045e-16f)
#define DF_E12 DFV(2.08767559e-09f, 1.108283981e-16f)
#define DF_E13 DFV(1.60590444e-10f, -5.352526512e-18f)

#define DF_A3 DFV(0.166666672f, -4.967053879e-09f)
#define DF_A5 DFV(0.075000003f, -2.980232283e-09f)
#define DF_A7 DFV(0.0446428582f, -1.064368704e-09f)
#define DF_A9 DFV(0.030381944f, 4.139211474e-10f)

/* ---- error-free float transformations (shared by the float-only modes) ---- */
static inline float two_sum(float a, float b, float *err) {
    float s = a + b; float bb = s - a;
    *err = (a - (s - bb)) + (b - bb); return s;
}
static inline float quick_two_sum(float a, float b, float *err) {
    float s = a + b; *err = b - (s - a); return s;
}
static inline float two_prod(float a, float b, float *err) {
    float p = a * b;            /* FP_CONTRACT OFF keeps this a plain mul */
    *err = fma(a, b, -p);       /* explicit fma is always fused/correct */
    return p;
}

/* ---- constructors ---- */
#ifdef DF_HYBRID
/* float2 storage, double compute */
static inline double _dfd(df a){ return (double)a.x + (double)a.y; }
static inline df _dfs(double v){ float hi=(float)v; return (float2)(hi,(float)(v-(double)hi)); }
static inline df df_f(float a){ return (float2)(a,0.0f); }
static inline df df_i(int a){ return _dfs((double)a); }
static inline df df_add(df a, df b){ return _dfs(_dfd(a)+_dfd(b)); }
static inline df df_sub(df a, df b){ return _dfs(_dfd(a)-_dfd(b)); }
static inline df df_neg(df a){ return (float2)(-a.x,-a.y); }
static inline df df_mul(df a, df b){ return _dfs(_dfd(a)*_dfd(b)); }
static inline df df_sqr(df a){ double d=_dfd(a); return _dfs(d*d); }
static inline df df_div(df a, df b){ return _dfs(_dfd(a)/_dfd(b)); }
static inline df df_rcp(df a){ return _dfs(1.0/_dfd(a)); }
static inline int df_lt(df a, df b){ return _dfd(a) <  _dfd(b); }
static inline int df_gt(df a, df b){ return _dfd(a) >  _dfd(b); }
static inline int df_le(df a, df b){ return _dfd(a) <= _dfd(b); }
static inline int df_ge(df a, df b){ return _dfd(a) >= _dfd(b); }
static inline int df_eq(df a, df b){ return _dfd(a) == _dfd(b); }
static inline df df_fabs(df a){ return _dfs(fabs(_dfd(a))); }
static inline df df_min(df a, df b){ return _dfs(fmin(_dfd(a),_dfd(b))); }
static inline df df_max(df a, df b){ return _dfs(fmax(_dfd(a),_dfd(b))); }
static inline int df_isnan(df a){ return isnan(a.x)||isnan(a.y); }
static inline df df_sqrt(df a){ return _dfs(sqrt(_dfd(a))); }
static inline df df_rsqrt(df a){ return _dfs(rsqrt(_dfd(a))); }
static inline df df_exp(df a){ return _dfs(exp(_dfd(a))); }
static inline df df_log(df a){ return _dfs(log(_dfd(a))); }
static inline df df_acos(df a){ return _dfs(acos(_dfd(a))); }
static inline df df_asin_small(df a){ return _dfs(asin(_dfd(a))); }
static inline df df_fmod(df a, df b){ return _dfs(fmod(_dfd(a),_dfd(b))); }
static inline df df_round(df a){ return _dfs(round(_dfd(a))); }
static inline df df_trunc(df a){ return _dfs(trunc(_dfd(a))); }
static inline df df_ldexp(df a, int e){ return _dfs(ldexp(_dfd(a),e)); }
static inline void df_sincos(df x, df* s, df* c){ double d=_dfd(x); *s=_dfs(sin(d)); *c=_dfs(cos(d)); }
#elif defined(DF_TRIPLE_HYBRID)
/* triple-float storage, double compute (see typedef block for rationale) */
static inline double _dfd(df a){ return ((double)a.x + (double)a.y) + (double)a.z; }
static inline df _dfs(double v){
    float h = (float)v;
    float m = (float)(v - (double)h);
    float l = (float)(v - (double)h - (double)m);
    return (df){ h, m, l };
}
static inline df df_f(float a){ return (df){ a, 0.0f, 0.0f }; }
static inline df df_i(int a){ return _dfs((double)a); }
static inline df df_add(df a, df b){ return _dfs(_dfd(a)+_dfd(b)); }
static inline df df_sub(df a, df b){ return _dfs(_dfd(a)-_dfd(b)); }
static inline df df_neg(df a){ return (df){ -a.x, -a.y, -a.z }; }
static inline df df_mul(df a, df b){ return _dfs(_dfd(a)*_dfd(b)); }
static inline df df_sqr(df a){ double d=_dfd(a); return _dfs(d*d); }
static inline df df_div(df a, df b){ return _dfs(_dfd(a)/_dfd(b)); }
static inline df df_rcp(df a){ return _dfs(1.0/_dfd(a)); }
static inline int df_lt(df a, df b){ return _dfd(a) <  _dfd(b); }
static inline int df_gt(df a, df b){ return _dfd(a) >  _dfd(b); }
static inline int df_le(df a, df b){ return _dfd(a) <= _dfd(b); }
static inline int df_ge(df a, df b){ return _dfd(a) >= _dfd(b); }
static inline int df_eq(df a, df b){ return _dfd(a) == _dfd(b); }
static inline df df_fabs(df a){ return _dfs(fabs(_dfd(a))); }
static inline df df_min(df a, df b){ return _dfs(fmin(_dfd(a),_dfd(b))); }
static inline df df_max(df a, df b){ return _dfs(fmax(_dfd(a),_dfd(b))); }
static inline int df_isnan(df a){ return isnan(a.x)||isnan(a.y)||isnan(a.z); }
static inline df df_sqrt(df a){ return _dfs(sqrt(_dfd(a))); }
static inline df df_rsqrt(df a){ return _dfs(rsqrt(_dfd(a))); }
static inline df df_exp(df a){ return _dfs(exp(_dfd(a))); }
static inline df df_log(df a){ return _dfs(log(_dfd(a))); }
static inline df df_acos(df a){ return _dfs(acos(_dfd(a))); }
static inline df df_asin_small(df a){ return _dfs(asin(_dfd(a))); }
static inline df df_fmod(df a, df b){ return _dfs(fmod(_dfd(a),_dfd(b))); }
static inline df df_round(df a){ return _dfs(round(_dfd(a))); }
static inline df df_trunc(df a){ return _dfs(trunc(_dfd(a))); }
static inline df df_ldexp(df a, int e){ return _dfs(ldexp(_dfd(a),e)); }
static inline void df_sincos(df x, df* s, df* c){ double d=_dfd(x); *s=_dfs(sin(d)); *c=_dfs(cos(d)); }
#elif defined(DF_REAL64)
/* isolation build: df == native double, ops are trivial. Runs the exact same
   converted kernels in true FP64 to separate a port bug from an emulation bug. */
static inline df df_f(float a) { return (df)a; }
static inline df df_i(int a)   { return (df)a; }
static inline df df_add(df a, df b){ return a + b; }
static inline df df_sub(df a, df b){ return a - b; }
static inline df df_neg(df a){ return -a; }
static inline df df_mul(df a, df b){ return a * b; }
static inline df df_sqr(df a){ return a * a; }
static inline df df_div(df a, df b){ return a / b; }
static inline df df_rcp(df a){ return 1.0 / a; }
static inline int df_lt(df a, df b){ return a <  b; }
static inline int df_gt(df a, df b){ return a >  b; }
static inline int df_le(df a, df b){ return a <= b; }
static inline int df_ge(df a, df b){ return a >= b; }
static inline int df_eq(df a, df b){ return a == b; }
static inline df df_fabs(df a){ return fabs(a); }
static inline df df_min(df a, df b){ return fmin(a, b); }
static inline df df_max(df a, df b){ return fmax(a, b); }
static inline int df_isnan(df a){ return isnan(a); }
static inline df df_sqrt(df a){ return sqrt(a); }
static inline df df_rsqrt(df a){ return rsqrt(a); }
static inline df df_exp(df a){ return exp(a); }
static inline df df_log(df a){ return log(a); }
static inline df df_acos(df a){ return acos(a); }
static inline df df_asin_small(df a){ return asin(a); }
static inline df df_fmod(df a, df b){ return fmod(a, b); }
static inline df df_round(df a){ return round(a); }
static inline df df_trunc(df a){ return trunc(a); }
static inline df df_ldexp(df a, int e){ return ldexp(a, e); }
static inline void df_sincos(df x, df* s, df* c){ *s = sin(x); *c = cos(x); }
#elif defined(DF_TRIPLE)
/* ============== triple-float, FP32 ops only (~2^-63 per op) ==============
 * A df is a non-overlapping 3-float expansion (Priest/Shewchuk style):
 * value = x + y + z with |y| <~ ulp(x)/2, |z| <~ ulp(y)/2. Products and sums
 * are built from the error-free transforms above; every op ends in a
 * renormalization pass, so results stay canonical and comparisons stay
 * lexicographic. Per-op relative error ~2^-63..-65 - an order below native
 * double's 2^-53, which is the bar for tracking the FP64 app's trajectory. */

/* fold a (roughly ordered) 3-term sum back into canonical form */
static inline df tf_renorm(float c0, float c1, float c2) {
    c1 = two_sum(c1, c2, &c2);
    c0 = two_sum(c0, c1, &c1);
    c1 = two_sum(c1, c2, &c2);
    c0 = quick_two_sum(c0, c1, &c1);
    c1 = quick_two_sum(c1, c2, &c2);
    return (df){ c0, c1, c2 };
}

static inline df df_f(float a) { return (df){ a, 0.0f, 0.0f }; }
static inline df df_i(int a)   { float h = (float)a; return (df){ h, (float)(a - (int)h), 0.0f }; }

static inline df df_add(df a, df b) {
    float s0, s1, e0, e1, t1, e2;
    s0 = two_sum(a.x, b.x, &e0);
    if (!isfinite(s0)) return (df){ s0, 0.0f, 0.0f };
    s1 = two_sum(a.y, b.y, &e1);
    t1 = two_sum(s1, e0, &e2);
    return tf_renorm(s0, t1, (a.z + b.z) + e1 + e2);
}
static inline df df_neg(df a) { return (df){ -a.x, -a.y, -a.z }; }
static inline df df_sub(df a, df b) { return df_add(a, df_neg(b)); }

static inline df df_mul(df a, df b) {
    float p0, q0, p1, q1, p2, q2, s1, e1, e2, low;
    p0 = two_prod(a.x, b.x, &q0);
    if (!isfinite(p0)) return (df){ p0, 0.0f, 0.0f };
    p1 = two_prod(a.x, b.y, &q1);
    p2 = two_prod(a.y, b.x, &q2);
    s1 = two_sum(p1, p2, &e1);
    s1 = two_sum(s1, q0, &e2);
    /* third-order terms: plain float is exact enough at 2^-48 scale */
    low = ((q1 + q2) + (e1 + e2)) + fma(a.y, b.y, fma(a.x, b.z, a.z * b.x));
    return tf_renorm(p0, s1, low);
}
static inline df df_sqr(df a) { return df_mul(a, a); }

/* tf * plain float, used by division/reduction (exact through 2nd order) */
static inline df tf_mulf(df a, float f) {
    float p0, q0, p1, q1, s1, e1;
    p0 = two_prod(a.x, f, &q0);
    if (!isfinite(p0)) return (df){ p0, 0.0f, 0.0f };
    p1 = two_prod(a.y, f, &q1);
    s1 = two_sum(p1, q0, &e1);
    return tf_renorm(p0, s1, fma(a.z, f, q1) + e1);
}

static inline df df_div(df a, df b) {
    /* long division: each float quotient digit adds ~24 bits */
    float q0 = a.x / b.x;
    if (!isfinite(q0) || q0 == 0.0f) return (df){ q0, 0.0f, 0.0f };
    df r = df_sub(a, tf_mulf(b, q0));
    float q1 = r.x / b.x;
    r = df_sub(r, tf_mulf(b, q1));
    float q2 = r.x / b.x;
    r = df_sub(r, tf_mulf(b, q2));
    float q3 = r.x / b.x;                 /* ~2^-72 tail correction */
    df q = tf_renorm(q0, q1, q2);
    return df_add(q, (df){ q3, 0.0f, 0.0f });
}
static inline df df_rcp(df b) { return df_div(DF_ONE, b); }

/* ---- comparisons: canonical form makes lexicographic order exact ---- */
static inline int df_lt(df a, df b) {
    return (a.x < b.x) || (a.x == b.x && (a.y < b.y || (a.y == b.y && a.z < b.z)));
}
static inline int df_gt(df a, df b) {
    return (a.x > b.x) || (a.x == b.x && (a.y > b.y || (a.y == b.y && a.z > b.z)));
}
static inline int df_le(df a, df b) { return !df_gt(a, b); }
static inline int df_ge(df a, df b) { return !df_lt(a, b); }
static inline int df_eq(df a, df b) { return a.x == b.x && a.y == b.y && a.z == b.z; }

static inline df df_fabs(df a) {
    return (a.x < 0.0f || (a.x == 0.0f && (a.y < 0.0f || (a.y == 0.0f && a.z < 0.0f))))
        ? df_neg(a) : a;
}
static inline df df_min(df a, df b) { return df_lt(b, a) ? b : a; }
static inline df df_max(df a, df b) { return df_lt(a, b) ? b : a; }
static inline int df_isnan(df a) { return isnan(a.x) || isnan(a.y) || isnan(a.z); }

static inline df df_round(df a) {
    float r = round(a.x);
    float d = (a.x - r) + a.y;   /* |a.x - r| <= 0.5 -> exact (Sterbenz) */
    if (d > 0.5f) r += 1.0f;
    else if (d < -0.5f) r -= 1.0f;
    return (df){ r, 0.0f, 0.0f };
}
static inline df df_trunc(df a) {
    float t = trunc(a.x);
    df r = df_sub(a, (df){ t, 0.0f, 0.0f });   /* fractional remainder */
    float t2 = trunc(r.x);                     /* in case a.y crossed an integer */
    return df_f(t + t2);
}
static inline df df_ldexp(df a, int e) {
    return (df){ ldexp(a.x, e), ldexp(a.y, e), ldexp(a.z, e) };
}

static inline df df_sqrt(df a) {
    float s0 = sqrt(a.x);
    if (!isfinite(s0) || s0 == 0.0f) return (df){ s0, 0.0f, 0.0f };
    /* two Newton steps on 1/sqrt (no division), then one Heron correction:
       2^-24 seed -> 2^-47 -> arithmetic floor ~2^-64 */
    df y = df_f(1.0f / s0);
    df e = df_sub(DF_ONE, df_mul(a, df_mul(y, y)));
    y = df_add(y, df_mul(y, df_mul(e, DF_HALF)));
    e = df_sub(DF_ONE, df_mul(a, df_mul(y, y)));
    y = df_add(y, df_mul(y, df_mul(e, DF_HALF)));
    df s = df_mul(a, y);
    df d = df_sub(a, df_mul(s, s));
    return df_add(s, df_mul(df_mul(d, y), DF_HALF));
}
static inline df df_rsqrt(df a) {
    float s0 = rsqrt(a.x);
    if (!isfinite(s0) || s0 == 0.0f) return (df){ s0, 0.0f, 0.0f };
    df y = df_f(s0);
    df e = df_sub(DF_ONE, df_mul(a, df_mul(y, y)));
    y = df_add(y, df_mul(y, df_mul(e, DF_HALF)));
    e = df_sub(DF_ONE, df_mul(a, df_mul(y, y)));
    y = df_add(y, df_mul(y, df_mul(e, DF_HALF)));
    return y;
}

/* ---- 3-float Taylor tables (splits of the exact rationals) ---- */
#define DF3_S3 ((df){ -0.166666672f, 4.967053879e-09f, -1.48029741e-16f })
#define DF3_S5 ((df){ 0.00833333377f, -4.346172033e-10f, 1.850371804e-18f })
#define DF3_S7 ((df){ -0.000198412701f, 2.725596875e-12f, -5.438220743e-20f })
#define DF3_S9 ((df){ 2.75573188e-06f, 3.793571224e-14f, 1.508226522e-21f })
#define DF3_S11 ((df){ -2.50521079e-08f, -4.417623045e-16f, -1.178607575e-23f })
#define DF3_S13 ((df){ 1.60590444e-10f, -5.352526512e-18f, -1.683604693e-25f })
#define DF3_S15 ((df){ -7.64716361e-13f, -1.220071047e-20f, 9.393546961e-29f })
#define DF3_S17 ((df){ 2.81145736e-15f, -1.046208474e-22f, -2.20149422e-30f })
#define DF3_S19 ((df){ -8.22063508e-18f, -1.681478097e-25f, 1.319324993e-33f })
#define DF3_S21 ((df){ 1.95729415e-20f, -4.612945514e-28f, 4.654180615e-36f })
#define DF3_C2 ((df){ -0.5f, 0.0f, 0.0f })
#define DF3_C4 ((df){ 0.0416666679f, -1.24176347e-09f, 3.700743526e-17f })
#define DF3_C6 ((df){ -0.00138888892f, 3.363109444e-11f, -1.464877618e-18f })
#define DF3_C8 ((df){ 2.48015876e-05f, -3.406996094e-13f, 6.797775929e-21f })
#define DF3_C10 ((df){ -2.755732e-07f, 7.575112209e-15f, 1.879905267e-22f })
#define DF3_C12 ((df){ 2.08767559e-09f, 1.108283915e-16f, 3.18798788e-24f })
#define DF3_C14 ((df){ -1.14707454e-11f, -2.372207689e-19f, 4.640206186e-27f })
#define DF3_C16 ((df){ 4.77947726e-14f, 7.625444044e-22f, -5.87096685e-30f })
#define DF3_C18 ((df){ -1.56192068e-16f, -1.540447147e-24f, 1.274122294e-32f })
#define DF3_C20 ((df){ 4.11031759e-19f, 3.237511733e-27f, -1.430034451e-34f })
#define DF3_E2 ((df){ 0.5f, 0.0f, 0.0f })
#define DF3_E3 ((df){ 0.166666672f, -4.967053879e-09f, 1.48029741e-16f })
#define DF3_E4 ((df){ 0.0416666679f, -1.24176347e-09f, 3.700743526e-17f })
#define DF3_E5 ((df){ 0.00833333377f, -4.346172033e-10f, 1.850371804e-18f })
#define DF3_E6 ((df){ 0.00138888892f, -3.363109444e-11f, 1.464877618e-18f })
#define DF3_E7 ((df){ 0.000198412701f, -2.725596875e-12f, 5.438220743e-20f })
#define DF3_E8 ((df){ 2.48015876e-05f, -3.406996094e-13f, 6.797775929e-21f })
#define DF3_E9 ((df){ 2.75573188e-06f, 3.793571224e-14f, 1.508226522e-21f })
#define DF3_E10 ((df){ 2.755732e-07f, -7.575112209e-15f, -1.879905267e-22f })
#define DF3_E11 ((df){ 2.50521079e-08f, 4.417623045e-16f, 1.178607575e-23f })
#define DF3_E12 ((df){ 2.08767559e-09f, 1.108283915e-16f, 3.18798788e-24f })
#define DF3_E13 ((df){ 1.60590444e-10f, -5.352526512e-18f, -1.683604693e-25f })
#define DF3_E14 ((df){ 1.14707454e-11f, 2.372207689e-19f, -4.640206186e-27f })
#define DF3_E15 ((df){ 7.64716361e-13f, 1.220071047e-20f, -9.393546961e-29f })
#define DF3_E16 ((df){ 4.77947726e-14f, 7.625444044e-22f, -5.87096685e-30f })

/* ---- sincos: reduce by pi/2, deg-21/20 Taylor on |u|<=pi/4 (~2^-59) ----
 * The REDUCTION constants are 3-splits of the TRUE pi/2 / ln2 (not of the
 * double-rounded values the app-level DF_ constants carry): libm on the FP64
 * build reduces by the true constants too, and a double-rounded pi/2 would
 * put a k*2^-54 error into every reduced angle. */
#define DF3_PIO2X ((df){ 1.57079637f, -4.371138829e-08f, -1.71512451e-15f })
#define DF3_LN2X  ((df){ 0.693147182f, -1.904654212e-09f, -8.783183739e-17f })
static inline void df_sincos(df x, df *s, df *c) {
    df kk = df_round(df_mul(x, DF_2OPI));
    int k = (int)kk.x;
    df u = df_sub(x, df_mul(kk, DF3_PIO2X));
    df v = df_mul(u, u);
    df p = DF3_S21;
    p = df_add(df_mul(p, v), DF3_S19);
    p = df_add(df_mul(p, v), DF3_S17);
    p = df_add(df_mul(p, v), DF3_S15);
    p = df_add(df_mul(p, v), DF3_S13);
    p = df_add(df_mul(p, v), DF3_S11);
    p = df_add(df_mul(p, v), DF3_S9);
    p = df_add(df_mul(p, v), DF3_S7);
    p = df_add(df_mul(p, v), DF3_S5);
    p = df_add(df_mul(p, v), DF3_S3);
    df st = df_add(u, df_mul(u, df_mul(v, p)));   /* sin = u*(1 + v*P(v)) */
    p = DF3_C20;
    p = df_add(df_mul(p, v), DF3_C18);
    p = df_add(df_mul(p, v), DF3_C16);
    p = df_add(df_mul(p, v), DF3_C14);
    p = df_add(df_mul(p, v), DF3_C12);
    p = df_add(df_mul(p, v), DF3_C10);
    p = df_add(df_mul(p, v), DF3_C8);
    p = df_add(df_mul(p, v), DF3_C6);
    p = df_add(df_mul(p, v), DF3_C4);
    p = df_add(df_mul(p, v), DF3_C2);
    df ct = df_add(DF_ONE, df_mul(v, p));          /* cos = 1 + v*P(v) */
    int q = k & 3; if (q < 0) q += 4;
    if (q == 0)      { *s = st;          *c = ct; }
    else if (q == 1) { *s = ct;          *c = df_neg(st); }
    else if (q == 2) { *s = df_neg(st);  *c = df_neg(ct); }
    else             { *s = df_neg(ct);  *c = st; }
}

/* ---- exp: 2^k * e^r, r = x - k*ln2, deg-16 Taylor (~2^-61); clamp to huge
   FINITE values outside float range (keeps LM chisq comparisons inf-free) ---- */
static inline df df_exp(df x) {
    if (isnan(x.x)) return x;
    if (x.x > 88.0f)   return (df){ 3.0e38f, 0.0f, 0.0f };
    if (x.x < -103.0f) return DF_ZERO;
    float kf = round(x.x * 1.44269502f);
    int k = (int)kf;
    df r = df_sub(x, tf_mulf(DF3_LN2X, kf));
    df p = DF3_E16;
    p = df_add(df_mul(p, r), DF3_E15);
    p = df_add(df_mul(p, r), DF3_E14);
    p = df_add(df_mul(p, r), DF3_E13);
    p = df_add(df_mul(p, r), DF3_E12);
    p = df_add(df_mul(p, r), DF3_E11);
    p = df_add(df_mul(p, r), DF3_E10);
    p = df_add(df_mul(p, r), DF3_E9);
    p = df_add(df_mul(p, r), DF3_E8);
    p = df_add(df_mul(p, r), DF3_E7);
    p = df_add(df_mul(p, r), DF3_E6);
    p = df_add(df_mul(p, r), DF3_E5);
    p = df_add(df_mul(p, r), DF3_E4);
    p = df_add(df_mul(p, r), DF3_E3);
    p = df_add(df_mul(p, r), DF3_E2);
    /* e^r = 1 + r + r^2*P(r) */
    df res = df_add(DF_ONE, df_add(r, df_mul(df_mul(r, r), p)));
    return df_ldexp(res, k);
}

/* ---- log: float seed + two Newton steps y += x*exp(-y) - 1 (exp-limited,
   ~2^-60 abs) ---- */
static inline df df_log(df x) {
    float y0 = log(x.x);
    df y = df_f(y0);
    df t = df_mul(x, df_exp(df_neg(y)));
    y = df_add(y, df_sub(t, DF_ONE));
    t = df_mul(x, df_exp(df_neg(y)));
    y = df_add(y, df_sub(t, DF_ONE));
    return y;
}

/* ---- asin core for |t| <= ~0.71: float seed + TWO Newton steps on
   sin(z)-t=0 (cos z >= 0.7, no noise amplification): 2^-24 -> 2^-47 ->
   sincos-noise floor ~2^-59 ---- */
static inline df df_asin_core(df t) {
    float z0 = asin(t.x);
    df z = df_f(z0);
    df s, c;
    df_sincos(z, &s, &c);
    z = df_add(z, df_div(df_sub(t, s), c));
    df_sincos(z, &s, &c);
    z = df_add(z, df_div(df_sub(t, s), c));
    return z;
}
static inline df df_asin_small(df t) { return df_asin_core(t); }
/* ---- acos via half-angle identities only (see the df64 variant below for
   the rationale; 1-x / 1+x are exact by cancellation). The reflections use
   the TRUE-pi splits (like the sincos reduction): libm's acos approximates
   the true function, so reflecting around double-rounded pi would floor the
   error at 2^-53. ---- */
#define DF3_PIX ((df){ 3.14159274f, -8.742277657e-08f, -3.43024902e-15f })
static inline df df_acos(df x) {
    if (df_ge(x, DF_ONE))  return DF_ZERO;
    if (df_le(x, df_neg(DF_ONE))) return DF_PI;
    if (x.x >= 0.5f) {
        df t = df_sqrt(df_mul(df_sub(DF_ONE, x), DF_HALF));
        return df_mul(df_f(2.0f), df_asin_core(t));
    }
    if (x.x <= -0.5f) {
        df t = df_sqrt(df_mul(df_add(DF_ONE, x), DF_HALF));
        return df_sub(DF3_PIX, df_mul(df_f(2.0f), df_asin_core(t)));
    }
    return df_sub(DF3_PIO2X, df_asin_core(x));
}

/* ---- fmod(a,b) = a - b*trunc(a/b) ---- */
static inline df df_fmod(df a, df b) {
    df q = df_div(a, b);
    float n = trunc(q.x);
    return df_sub(a, tf_mulf(b, n));
}
#else
static inline df df_f(float a) { return DFV(a, 0.0f); }
static inline df df_i(int a)   { float h = (float)a; return DFV(h, (float)(a - (int)h)); }

/* ---- basic arithmetic (QD accurate variants) ---- */
static inline df df_add(df a, df b) {
    float s1, s2, t1, t2;
    s1 = two_sum(a.x, b.x, &s2);
    if (!isfinite(s1)) return DFV(s1, 0.0f);
    t1 = two_sum(a.y, b.y, &t2);
    s2 += t1; s1 = quick_two_sum(s1, s2, &s2);
    s2 += t2; s1 = quick_two_sum(s1, s2, &s2);
    return DFV(s1, s2);
}
static inline df df_neg(df a) { return DFV(-a.x, -a.y); }
static inline df df_sub(df a, df b) { return df_add(a, df_neg(b)); }
static inline df df_mul(df a, df b) {
    float p1, p2;
    p1 = two_prod(a.x, b.x, &p2);
    if (!isfinite(p1)) return DFV(p1, 0.0f);
    p2 = fma(a.x, b.y, fma(a.y, b.x, fma(a.y, b.y, p2)));
    p1 = quick_two_sum(p1, p2, &p2);
    return DFV(p1, p2);
}
static inline df df_sqr(df a) { return df_mul(a, a); }
static inline df df_div(df a, df b) {
    float q1 = a.x / b.x;
    if (!isfinite(q1) || q1 == 0.0f) return DFV(q1, 0.0f);
    df r = df_sub(a, df_mul(b, df_f(q1)));
    float q2 = (r.x + r.y) / b.x;
    r = df_sub(r, df_mul(b, df_f(q2)));
    float q3 = (r.x + r.y) / b.x;
    float s1, s2; s1 = quick_two_sum(q1, q2, &s2);
    return df_add(DFV(s1, s2), df_f(q3));
}

/* ---- comparisons ---- */
static inline int df_lt(df a, df b) { return (a.x < b.x) || (a.x == b.x && a.y < b.y); }
static inline int df_gt(df a, df b) { return (a.x > b.x) || (a.x == b.x && a.y > b.y); }
static inline int df_le(df a, df b) { return !df_gt(a, b); }
static inline int df_ge(df a, df b) { return !df_lt(a, b); }
static inline int df_eq(df a, df b) { return a.x == b.x && a.y == b.y; }

/* ---- misc ---- */
static inline df df_fabs(df a) { return (a.x < 0.0f || (a.x == 0.0f && a.y < 0.0f)) ? df_neg(a) : a; }
static inline df df_min(df a, df b) { return df_lt(b, a) ? b : a; }
static inline df df_max(df a, df b) { return df_lt(a, b) ? b : a; }
static inline int df_isnan(df a) { return isnan(a.x) || isnan(a.y); }

static inline df df_round(df a) {
    float r = round(a.x);
    float d = (a.x - r) + a.y;   /* |a.x - r| <= 0.5 -> exact (Sterbenz) */
    if (d > 0.5f) r += 1.0f;
    else if (d < -0.5f) r -= 1.0f;
    return DFV(r, 0.0f);
}
static inline df df_trunc(df a) {
    float t = trunc(a.x);
    df r = df_sub(a, df_f(t));   /* fractional remainder */
    float t2 = trunc(r.x);       /* in case a.y crossed an integer */
    return df_f(t + t2);
}

static inline df df_rcp(df b) {
    float r0 = 1.0f / b.x;
    if (!isfinite(r0) || r0 == 0.0f) return DFV(r0, 0.0f);
    df e = df_sub(DF_ONE, df_mul(b, df_f(r0)));
    return df_add(df_f(r0), df_mul(df_f(r0), e));
}
static inline df df_sqrt(df a) {
    float s0 = sqrt(a.x);
    if (!isfinite(s0) || s0 == 0.0f) return DFV(s0, 0.0f);
    df d = df_sub(a, df_mul(df_f(s0), df_f(s0)));
    float corr = (d.x + d.y) * (0.5f / s0);
    float r1, r2; r1 = quick_two_sum(s0, corr, &r2);
    return DFV(r1, r2);
}

/* ---- sincos: reduce by pi/2 (df arithmetic, args pre-reduced to [0,2pi) by
   the app's fmod), deg-19/18 Taylor on |u|<=pi/4 ---- */
static inline df df_sin_taylor(df u) {
    df x2 = df_mul(u, u);
    df p = DF_S19;
    p = df_add(df_mul(p, x2), DF_S17);
    p = df_add(df_mul(p, x2), DF_S15);
    p = df_add(df_mul(p, x2), DF_S13);
    p = df_add(df_mul(p, x2), DF_S11);
    p = df_add(df_mul(p, x2), DF_S9);
    p = df_add(df_mul(p, x2), DF_S7);
    p = df_add(df_mul(p, x2), DF_S5);
    p = df_add(df_mul(p, x2), DF_S3);
    /* sin = u + u*x2*p (p already carries the sign pattern) */
    return df_add(u, df_mul(u, df_mul(x2, p)));
}
static inline df df_cos_taylor(df u) {
    df x2 = df_mul(u, u);
    df p = DF_C18;
    p = df_add(df_mul(p, x2), DF_C16);
    p = df_add(df_mul(p, x2), DF_C14);
    p = df_add(df_mul(p, x2), DF_C12);
    p = df_add(df_mul(p, x2), DF_C10);
    p = df_add(df_mul(p, x2), DF_C8);
    p = df_add(df_mul(p, x2), DF_C6);
    p = df_add(df_mul(p, x2), DF_C4);
    p = df_add(df_mul(p, x2), DF_C2);
    return df_add(DF_ONE, df_mul(x2, p));
}
static inline void df_sincos(df x, df *s, df *c) {
    df kk = df_round(df_mul(x, DF_2OPI));
    int k = (int)kk.x;
    df u = df_sub(x, df_mul(kk, DF_PIO2));
    df st = df_sin_taylor(u);
    df ct = df_cos_taylor(u);
    int q = k & 3; if (q < 0) q += 4;
    if (q == 0)      { *s = st;          *c = ct; }
    else if (q == 1) { *s = ct;          *c = df_neg(st); }
    else if (q == 2) { *s = df_neg(st);  *c = df_neg(ct); }
    else             { *s = df_neg(ct);  *c = st; }
}

/* ---- exp: 2^k * exp(r), r = x - k*ln2, deg-13 Taylor; clamp overflow to a
   huge FINITE value (keeps LM chisq comparisons inf-free) ---- */
static inline df df_exp_taylor(df r) {
    df p = DF_E13;
    p = df_add(df_mul(p, r), DF_E12);
    p = df_add(df_mul(p, r), DF_E11);
    p = df_add(df_mul(p, r), DF_E10);
    p = df_add(df_mul(p, r), DF_E9);
    p = df_add(df_mul(p, r), DF_E8);
    p = df_add(df_mul(p, r), DF_E7);
    p = df_add(df_mul(p, r), DF_E6);
    p = df_add(df_mul(p, r), DF_E5);
    p = df_add(df_mul(p, r), DF_E4);
    p = df_add(df_mul(p, r), DF_E3);
    p = df_add(df_mul(p, r), DF_E2);
    p = df_add(df_mul(p, r), DF_ONE);
    p = df_add(df_mul(p, r), DF_ONE);
    return p;
}
static inline df df_ldexp(df a, int k) { return DFV(ldexp(a.x, k), ldexp(a.y, k)); }
static inline df df_exp(df x) {
    if (isnan(x.x)) return x;
    if (x.x > 25.0f)  return DFV(7.2e10f, 0.0f);   /* huge finite, see mreal.h */
    if (x.x < -87.0f) return DF_ZERO;
    float kf = round(x.x * 1.44269502f);
    int k = (int)kf;
    df r = df_sub(x, df_mul(df_f(kf), DF_LN2));
    return df_ldexp(df_exp_taylor(r), k);
}

/* ---- log: float seed + one Newton step  y += x*exp(-y) - 1 ---- */
static inline df df_log(df x) {
    float y0 = log(x.x);
    df y = df_f(y0);
    /* two Newton steps y += x*exp(-y) - 1 (24 -> ~44 bits; second step covers
       exp's ~2^-44 residual). Relative error still degrades as x -> 1 where
       log -> 0, but the app only takes log() of O(0.1..few) scattering
       coefficients, far from that singularity. */
    df t = df_mul(x, df_exp(df_neg(y)));
    y = df_add(y, df_sub(t, DF_ONE));
    t = df_mul(x, df_exp(df_neg(y)));
    y = df_add(y, df_sub(t, DF_ONE));
    return y;
}

/* ---- asin core for |t| <= ~0.71: float seed + one Newton step on
   sin(z)-t=0 (z += (t - sin z)/cos z). cos z >= 0.7 in this range, so the
   sincos evaluation noise is never amplified; quadratic convergence takes
   the 2^-24 seed straight to the ~2^-46 df noise floor. ---- */
static inline df df_asin_core(df t) {
    float z0 = asin(t.x);
    df s, c; df_sincos(df_f(z0), &s, &c);
    return df_add(df_f(z0), df_div(df_sub(t, s), c));
}
static inline df df_asin_small(df t) { return df_asin_core(t); }
/* ---- acos via half-angle identities only. Newton directly on cos(y)-x=0
   amplifies the sincos noise by 1/sin(y), which wrecks the |x|~1 boundary
   (opposition zone, x=ee.ee0 reaches 1-9e-8); the half-angle forms keep the
   asin argument <= ~0.71 where there is no amplification. 1-x / 1+x are
   exact in df by cancellation, so the tiny opposition-zone values survive.
   x >= 0.5:  acos(x) =        2*asin(sqrt((1-x)/2))   (t <= 0.5)
   |x| < 0.5: acos(x) = pi/2 - asin(x)
   x <= -0.5: acos(x) = pi -   2*asin(sqrt((1+x)/2))   (t <= 0.5)  ---- */
static inline df df_acos(df x) {
    if (df_ge(x, DF_ONE))  return DF_ZERO;
    if (df_le(x, df_neg(DF_ONE))) return DF_PI;
    if (x.x >= 0.5f) {
        df t = df_sqrt(df_mul(df_sub(DF_ONE, x), DF_HALF));
        return df_mul(df_f(2.0f), df_asin_core(t));
    }
    if (x.x <= -0.5f) {
        df t = df_sqrt(df_mul(df_add(DF_ONE, x), DF_HALF));
        return df_sub(DF_PI, df_mul(df_f(2.0f), df_asin_core(t)));
    }
    return df_sub(DF_PIO2, df_asin_core(x));
}

/* ---- fmod(a,b) = a - b*trunc(a/b) ---- */
static inline df df_fmod(df a, df b) {
    df q = df_div(a, b);
    float n = trunc(q.x);
    return df_sub(a, df_mul(df_f(n), b));
}
#endif /* DF_REAL64 else (float-float impl) */

#endif /* DF64_CL */
