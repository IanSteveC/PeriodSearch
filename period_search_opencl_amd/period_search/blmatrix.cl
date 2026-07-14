 //beta, lambda rotation matrix and its derivatives

 //  8.11.2006


//#include <math.h>
//#include "globals_CUDA.h"

void blmatrix(__global struct mfreq_context* CUDA_LCC, df bet, df lam)
{
	df cb, sb, cl, sl;
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	df_sincos(bet, &sb, &cb);
  	df_sincos(lam, &sl, &cl);
	(*CUDA_LCC).Blmat[1][1] = df_mul(cb, cl);
	(*CUDA_LCC).Blmat[1][2] = df_mul(cb, sl);
	(*CUDA_LCC).Blmat[1][3] = df_neg(sb);
	(*CUDA_LCC).Blmat[2][1] = df_neg(sl);
	(*CUDA_LCC).Blmat[2][2] = cl;
	(*CUDA_LCC).Blmat[2][3] = DF_ZERO;
	(*CUDA_LCC).Blmat[3][1] = df_mul(sb, cl);
	(*CUDA_LCC).Blmat[3][2] = df_mul(sb, sl);
	(*CUDA_LCC).Blmat[3][3] = cb;

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//{
	//	printf("bet: %10.7f, lam: %10.7f\n", bet, lam);
	//	printf("Blmat[1][1]: %10.7f, Blmat[2][1]: %10.7f, Blmat[3][1]: %10.7f\n", (*CUDA_LCC).Blmat[1][1], (*CUDA_LCC).Blmat[2][1], (*CUDA_LCC).Blmat[3][1]);
	//	printf("Blmat[1][2]: %10.7f, Blmat[2][2]: %10.7f, Blmat[3][2]: %10.7f\n", (*CUDA_LCC).Blmat[1][2], (*CUDA_LCC).Blmat[2][2], (*CUDA_LCC).Blmat[3][2]);
	//	printf("Blmat[1][3]: %10.7f, Blmat[2][3]: %10.7f, Blmat[3][3]: %10.7f\n", (*CUDA_LCC).Blmat[1][3], (*CUDA_LCC).Blmat[2][3], (*CUDA_LCC).Blmat[3][3]);
	//}

	/* Ders. of Blmat w.r.t. bet */
	(*CUDA_LCC).Dblm[1][1][1] = df_mul(df_neg(sb), cl);
	(*CUDA_LCC).Dblm[1][1][2] = df_mul(df_neg(sb), sl);
	(*CUDA_LCC).Dblm[1][1][3] = df_neg(cb);
	(*CUDA_LCC).Dblm[1][2][1] = DF_ZERO;
	(*CUDA_LCC).Dblm[1][2][2] = DF_ZERO;
	(*CUDA_LCC).Dblm[1][2][3] = DF_ZERO;
	(*CUDA_LCC).Dblm[1][3][1] = df_mul(cb, cl);
	(*CUDA_LCC).Dblm[1][3][2] = df_mul(cb, sl);
	(*CUDA_LCC).Dblm[1][3][3] = df_neg(sb);
	/* Ders. w.r.t. lam */
	(*CUDA_LCC).Dblm[2][1][1] = df_mul(df_neg(cb), sl);
	(*CUDA_LCC).Dblm[2][1][2] = df_mul(cb, cl);
	(*CUDA_LCC).Dblm[2][1][3] = DF_ZERO;
	(*CUDA_LCC).Dblm[2][2][1] = df_neg(cl);
	(*CUDA_LCC).Dblm[2][2][2] = df_neg(sl);
	(*CUDA_LCC).Dblm[2][2][3] = DF_ZERO;
	(*CUDA_LCC).Dblm[2][3][1] = df_mul(df_neg(sb), sl);
	(*CUDA_LCC).Dblm[2][3][2] = df_mul(sb, cl);
	(*CUDA_LCC).Dblm[2][3][3] = DF_ZERO;
}
