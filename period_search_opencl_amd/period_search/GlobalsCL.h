#pragma OPENCL FP_CONTRACT OFF  /* df64: two_prod must not be contracted */

/* FP32 build: no cl_khr_fp64 - all reals are df64 (float2), see df64.cl */
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

//struct __attribute__((packed)) freq_context
//struct mfreq_context
//struct __attribute__((aligned(8))) mfreq_context
//#ifdef NVIDIA
//struct mfreq_context
//#else
//typedef struct mfreq_context
//#endif
typedef struct mfreq_context
{
	//df* Area;
	//df* Dg;
	//df* alpha;
	//df* covar;
	//df* dytemp;
	//df* ytemp;

	df Area[MAX_N_FAC + 1];
	/* The point- and fit-dimensioned work arrays (alpha, covar, dytemp,
	   ytemp, jp_*, e_*, de, de0) live in a separate runtime-sized scratch
	   buffer - one slice of freq_context.scrStride doubles per work-group,
	   at the offsets recorded in freq_context - instead of compile-time
	   worst-case arrays here. That cuts per-context memory ~6x (2.27 MB ->
	   ~0.4 MB for typical workunits). */
	df beta[MAX_N_PAR + 1];
	df atry[MAX_N_PAR + 1];
	df da[MAX_N_PAR + 1];
	df cg[MAX_N_PAR + 1];
	df Blmat[4][4];
	df Dblm[3][4][4];
	df dave[MAX_N_PAR + 1];
	df dyda[MAX_N_PAR + 1];

	df sh_big[BLOCK_DIM];
	df chck[4];
	df pivinv;
	df ave;
	df freq;
	df Alamda;
	df Chisq;
	df Ochisq;
	df rchisq;
	df trial_chisq;
	df iter_diff, dev_old, dev_new;

	int Niter;
	int np, np1, np2;
	int isInvalid, isAlamda, isNiter;
	int icol;
	//df conw_r;

	int ipiv[MAX_N_PAR + 1];
	int indxc[MAX_N_PAR + 1];
	int indxr[MAX_N_PAR + 1];
	int sh_icol[BLOCK_DIM];
	int sh_irow[BLOCK_DIM];
} CUDA_LCC;

//struct freq_context
//typedef struct __attribute__((aligned(8))) freq_context
//#ifdef NVIDIA
//struct freq_context
//#else
//typedef struct freq_context
//#endif
struct freq_context
{
	df Phi_0;
	df logCl;
	df cl;
	//df logC;
	df lambda_pole[N_POLES + 1];
	df beta_pole[N_POLES + 1];


	df par[4];
	df Alamda_start;
	df Alamda_incr;

	//df cgFirst[MAX_N_PAR + 1];
	df tim[MAX_N_OBS + 1];
	df ee[MAX_N_OBS + 1][3];	// df* ee;
	df ee0[MAX_N_OBS + 1][3];	// df* ee0;
	df Sig[MAX_N_OBS + 1];
	df Weight[MAX_N_OBS + 1];
	df Brightness[MAX_N_OBS + 1];
	df Fc[MAX_N_FAC + 1][MAX_LM + 1];
	df Fs[MAX_N_FAC + 1][MAX_LM + 1];
	df Darea[MAX_N_FAC + 1];
	df Nor[MAX_N_FAC + 1][3];
	df Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1];
	df Pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1];
	df conw_r;

	int ia[MAX_N_PAR + 1];

	int Dg_block;
	int lastone;
	int lastma;
	int ma;
	int Mfit, Mfit1;
	int Mmax, Lmax;
	int n;
	int Ncoef, Ncoef0;
	int Numfac;
	int Numfac1;
	int Nphpar;
	int ndata;
	int Is_Precalc;

	/* runtime dimensions + per-context offsets (in doubles) into the
	   scratch buffer that replaced the fixed-size work arrays */
	int lcPoints1;
	int scrStride;
	int offAlpha;
	int offCovar;
	int offDytemp;
	int offYtemp;
	int offJpScale;
	int offJpDphp1;
	int offJpDphp2;
	int offJpDphp3;
	int offE1;
	int offE2;
	int offE3;
	int offE01;
	int offE02;
	int offE03;
	int offDe;
	int offDe0;
};

//struct freq_result
//struct __attribute__((aligned(8))) freq_result
//#ifdef NVIDIA
//struct freq_result
//#else
//typedef struct freq_result
//#endif
struct freq_result
{
	df dark_best, per_best, dev_best, dev_best_x2, la_best, be_best, freq;
	int isReported, isInvalid, isNiter;
};
