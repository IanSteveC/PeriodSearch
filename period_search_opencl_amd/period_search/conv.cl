//Convexity regularization function

//  8.11.2006


df conv(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__local df* res,
	int nc,
	int tmpl,
	int tmph,
	int brtmpl,
	int brtmph)
{
	int i, j, k;
	df tmp = df_f(0.0f);
	df dtmp;
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	//j = blockIdx.x * (CUDA_Numfac1)+brtmpl;
	j = brtmpl;
	for (i = brtmpl; i <= brtmph; i++, j++)
	{
		//tmp += CUDA_Area[j] * CUDA_Nor[i][nc];
		tmp = df_add(tmp, df_mul((*CUDA_LCC).Area[j], (*CUDA_CC).Nor[i][nc]));
	}

	res[threadIdx.x] = tmp;

	//if (threadIdx.x == 0)
	//    printf("conv>>> [%d] jp-1[%3d] res[%3d]: %10.7f\n", blockIdx.x, nc, threadIdx.x, res[threadIdx.x]);

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

	//parallel reduction
	k = BLOCK_DIM >> 1;
	while (k > 1)
	{
		if (threadIdx.x < k)
			res[threadIdx.x] = df_add(res[threadIdx.x], res[threadIdx.x + k]);
		k = k >> 1;
		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		tmp = df_add(res[0], res[1]);
	}
	//parallel reduction end
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

	for (j = tmpl; j <= tmph; j++)
	{
		dtmp = DF_ZERO;
		if (j <= (*CUDA_CC).Ncoef)
		{
			for (i = 1; i <= (*CUDA_CC).Numfac; i++)
			{
				/* Darea[i] * Dg[i][j] == Area[i] * Dsph[i][j] (Area = Darea*g) */
				dtmp = df_add(dtmp, df_mul(df_mul((*CUDA_LCC).Area[i], (*CUDA_CC).Dsph[i][j]), (*CUDA_CC).Nor[i][nc]));

				//if (blockIdx.x == 0 && j == 8)
				//	printf("[%d][%3d]  Darea[%4d]: %.7f, Dg[%4d]: %.7f, Nor[%3d][%3d]: %10.7f\n",
				//		blockIdx.x, threadIdx.x, i, (*CUDA_CC).Darea[i], mm, (*CUDA_LCC).Dg[mm], i, nc, (*CUDA_CC).Nor[i][nc]);
			}
		}

		(*CUDA_LCC).dyda[j] = dtmp;

		//if (blockIdx.x == 0) // && threadIdx.x == 1)
		//    printf("[mrqcof_curve1_last -> conv] [%d][%3d] jp - 1: %3d, j[%3d] dyda[%3d]: %10.7f\n",
		//        blockIdx.x, threadIdx.x, nc, j, j, (*CUDA_LCC).dyda[j]);
	}
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

	return (tmp);
}
