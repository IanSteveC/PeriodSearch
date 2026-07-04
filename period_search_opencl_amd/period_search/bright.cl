//computes integrated brightness of all visible and iluminated areas
//  and its derivatives

//  8.11.2006


void matrix_neo(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* cg,
	int lnp1,
	int Lpoints,
	int num)
{
	__private double f, cf, sf, pom, pom0, alpha;
	__private double ee_1, ee_2, ee_3, ee0_1, ee0_2, ee0_3, t, tmat;
	__private int lnp;

	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	int brtmph, brtmpl;
	brtmph = Lpoints / BLOCK_DIM;
	if (Lpoints % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > Lpoints) brtmph = Lpoints;
	brtmpl++;

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//{
	//	printf("Blmat[1][1]: %10.7f, Blmat[2][1]: %10.7f, Blmat[3][1]: %10.7f\n", (*CUDA_LCC).Blmat[1][1], (*CUDA_LCC).Blmat[2][1], (*CUDA_LCC).Blmat[3][1]);
	//	printf("Blmat[1][2]: %10.7f, Blmat[2][2]: %10.7f, Blmat[3][2]: %10.7f\n", (*CUDA_LCC).Blmat[1][2], (*CUDA_LCC).Blmat[2][2], (*CUDA_LCC).Blmat[3][2]);
	//	printf("Blmat[1][3]: %10.7f, Blmat[2][3]: %10.7f, Blmat[3][3]: %10.7f\n", (*CUDA_LCC).Blmat[1][3], (*CUDA_LCC).Blmat[2][3], (*CUDA_LCC).Blmat[3][3]);
	//}

	lnp = lnp1 + brtmpl - 1;
	//printf("lnp: %3d = lnp1: %3d + brtmpl: %3d - 1 | lnp++: %3d\n", lnp, lnp1, brtmpl, lnp + 1);

	int q = (*CUDA_CC).Ncoef0 + 2;
	//if (blockIdx.x == 0)
	//	printf("[neo] [%3d] cg[%3d]: %10.7f\n", blockIdx.x,  q, (*CUDA_LCC).cg[q]);

	for (int jp = brtmpl; jp <= brtmph; jp++)
	{
		lnp++;

		ee_1 = (*CUDA_CC).ee[lnp][0];		// position vectors
		ee0_1 = (*CUDA_CC).ee0[lnp][0];
		ee_2 = (*CUDA_CC).ee[lnp][1];
		ee0_2 = (*CUDA_CC).ee0[lnp][1];
		ee_3 = (*CUDA_CC).ee[lnp][2];
		ee0_3 = (*CUDA_CC).ee0[lnp][2];
		t = (*CUDA_CC).tim[lnp];

		//if (blockIdx.x == 0)
		//	printf("jp[%3d] lnp[%3d], %10.7f, %10.7f, %10.7f, %10.7f, %10.7f, %10.7f\n",
		//		jp, lnp, ee_1, ee_2, ee_3, ee0_1, ee0_2, ee0_3);

		//printf("tim[%3d]: %10.7f\n", lnp, t);
		//printf("lnp: %3d, ee[%d]: %.7f, ee0[%d]: %.7f\n", lnp, lnp * 3 + 0, (*CUDA_CC).ee[lnp][0], lnp, (*CUDA_CC).ee0[lnp][0]);

		alpha = acos(ee_1 * ee0_1 + ee_2 * ee0_2 + ee_3 * ee0_3);


		//if (blockIdx.x == 0 && threadIdx.x == 0)
		//	printf("[neo] alpha[%3d]: %.7f, cg[%3d]: %10.7f\n", jp, alpha, q, (*CUDA_LCC).cg[q]);

		/* Exp-lin model (const.term=1.) */
		double f = exp(-alpha / cg[(*CUDA_CC).Ncoef0 + 2]);	//f is temp here

		//if (blockIdx.x == 0 && threadIdx.x == 0)
		//	printf("[neo] [%2d][%3d] jp[%3d] f: %10.7f, cg[%3d] %10.7f, alpha %10.7f\n",
		//		blockIdx.x, threadIdx.x, jp, f, (*CUDA_CC).Ncoef0 + 2, cg[(*CUDA_CC).Ncoef0 + 2], alpha);

		(*CUDA_LCC).jp_Scale[jp] = 1 + cg[(*CUDA_CC).Ncoef0 + 1] * f + (cg[(*CUDA_CC).Ncoef0 + 3] * alpha);
		(*CUDA_LCC).jp_dphp_1[jp] = f;
		(*CUDA_LCC).jp_dphp_2[jp] = cg[(*CUDA_CC).Ncoef0 + 1] * f * alpha / (cg[(*CUDA_CC).Ncoef0 + 2] * cg[(*CUDA_CC).Ncoef0 + 2]);
		(*CUDA_LCC).jp_dphp_3[jp] = alpha;

		//if (blockIdx.x == 0)
		//	printf("[neo] [%d][%3d] jp_Scale[%3d]: %10.7f, jp_dphp_1[]: %10.7F, jp_dphp_2[]: %10.7f, jp_dphp_3[]: %10.7f\n",
		//		blockIdx.x, threadIdx.x, jp, (*CUDA_LCC).jp_Scale[jp], (*CUDA_LCC).jp_dphp_1[jp], (*CUDA_LCC).jp_dphp_2[jp], (*CUDA_LCC).jp_dphp_3[jp]);

		//  matrix start
		f = cg[(*CUDA_CC).Ncoef0] * t + (*CUDA_CC).Phi_0;
		f = fmod(f, 2 * PI); /* may give little different results than Mikko's */
		sf = sincos(f, &cf);

		//if (threadIdx.x == 0)
		//	printf("jp[%3d] [%3d] cf: %10.7f, sf: %10.7f\n", jp, blockIdx.x, cf, sf);

		//if (num == 1 && blockIdx.x == 0 && jp == brtmpl)
		//{
		//	printf("[%2d][%3d][%3d] f: % .6f, cosF: % .6f, sinF: % .6f\n", blockIdx.x, threadIdx.x, jp, f, cf, sf);
		//}

		//	/* rotation matrix, Z axis, angle f */

		tmat = cf * (*CUDA_LCC).Blmat[1][1] + sf * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = cf * (*CUDA_LCC).Blmat[1][2] + sf * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = cf * (*CUDA_LCC).Blmat[1][3] + sf * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).e_1[jp] = pom + tmat * ee_3;
		(*CUDA_LCC).e0_1[jp] = pom0 + tmat * ee0_3;

		//if (blockIdx.x == 0)
		//	printf("[%3d] jp[%3d] %10.7f, %10.7f\n", threadIdx.x, jp, (*CUDA_LCC).e_1[jp], (*CUDA_LCC).e0_1[jp]);

		tmat = (-sf) * (*CUDA_LCC).Blmat[1][1] + cf * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (-sf) * (*CUDA_LCC).Blmat[1][2] + cf * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (-sf) * (*CUDA_LCC).Blmat[1][3] + cf * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).e_2[jp] = pom + tmat * ee_3;
		(*CUDA_LCC).e0_2[jp] = pom0 + tmat * ee0_3;

		tmat = 0 * (*CUDA_LCC).Blmat[1][1] + 0 * (*CUDA_LCC).Blmat[2][1] + 1 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = 0 * (*CUDA_LCC).Blmat[1][2] + 0 * (*CUDA_LCC).Blmat[2][2] + 1 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = 0 * (*CUDA_LCC).Blmat[1][3] + 0 * (*CUDA_LCC).Blmat[2][3] + 1 * (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).e_3[jp] = pom + tmat * ee_3;
		(*CUDA_LCC).e0_3[jp] = pom0 + tmat * ee0_3;

		tmat = cf * (*CUDA_LCC).Dblm[1][1][1] + sf * (*CUDA_LCC).Dblm[1][2][1] + 0 * (*CUDA_LCC).Dblm[1][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = cf * (*CUDA_LCC).Dblm[1][1][2] + sf * (*CUDA_LCC).Dblm[1][2][2] + 0 * (*CUDA_LCC).Dblm[1][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = cf * (*CUDA_LCC).Dblm[1][1][3] + sf * (*CUDA_LCC).Dblm[1][2][3] + 0 * (*CUDA_LCC).Dblm[1][3][3];
		(*CUDA_LCC).de[jp][1][1] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][1][1] = pom0 + tmat * ee0_3;

		tmat = cf * (*CUDA_LCC).Dblm[2][1][1] + sf * (*CUDA_LCC).Dblm[2][2][1] + 0 * (*CUDA_LCC).Dblm[2][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = cf * (*CUDA_LCC).Dblm[2][1][2] + sf * (*CUDA_LCC).Dblm[2][2][2] + 0 * (*CUDA_LCC).Dblm[2][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = cf * (*CUDA_LCC).Dblm[2][1][3] + sf * (*CUDA_LCC).Dblm[2][2][3] + 0 * (*CUDA_LCC).Dblm[2][3][3];
		(*CUDA_LCC).de[jp][1][2] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][1][2] = pom0 + tmat * ee0_3;

		tmat = (-t * sf) * (*CUDA_LCC).Blmat[1][1] + (t * cf) * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (-t * sf) * (*CUDA_LCC).Blmat[1][2] + (t * cf) * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (-t * sf) * (*CUDA_LCC).Blmat[1][3] + (t * cf) * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).de[jp][1][3] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][1][3] = pom0 + tmat * ee0_3;

		tmat = -sf * (*CUDA_LCC).Dblm[1][1][1] + cf * (*CUDA_LCC).Dblm[1][2][1] + 0 * (*CUDA_LCC).Dblm[1][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = -sf * (*CUDA_LCC).Dblm[1][1][2] + cf * (*CUDA_LCC).Dblm[1][2][2] + 0 * (*CUDA_LCC).Dblm[1][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = -sf * (*CUDA_LCC).Dblm[1][1][3] + cf * (*CUDA_LCC).Dblm[1][2][3] + 0 * (*CUDA_LCC).Dblm[1][3][3];
		(*CUDA_LCC).de[jp][2][1] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][2][1] = pom0 + tmat * ee0_3;

		tmat = -sf * (*CUDA_LCC).Dblm[2][1][1] + cf * (*CUDA_LCC).Dblm[2][2][1] + 0 * (*CUDA_LCC).Dblm[2][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = -sf * (*CUDA_LCC).Dblm[2][1][2] + cf * (*CUDA_LCC).Dblm[2][2][2] + 0 * (*CUDA_LCC).Dblm[2][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = -sf * (*CUDA_LCC).Dblm[2][1][3] + cf * (*CUDA_LCC).Dblm[2][2][3] + 0 * (*CUDA_LCC).Dblm[2][3][3];
		(*CUDA_LCC).de[jp][2][2] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][2][2] = pom0 + tmat * ee0_3;

		tmat = (-t * cf) * (*CUDA_LCC).Blmat[1][1] + (-t * sf) * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (-t * cf) * (*CUDA_LCC).Blmat[1][2] + (-t * sf) * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (-t * cf) * (*CUDA_LCC).Blmat[1][3] + (-t * sf) * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).de[jp][2][3] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][2][3] = pom0 + tmat * ee0_3;

		tmat = 0 * (*CUDA_LCC).Dblm[1][1][1] + 0 * (*CUDA_LCC).Dblm[1][2][1] + 1 * (*CUDA_LCC).Dblm[1][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = 0 * (*CUDA_LCC).Dblm[1][1][2] + 0 * (*CUDA_LCC).Dblm[1][2][2] + 1 * (*CUDA_LCC).Dblm[1][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = 0 * (*CUDA_LCC).Dblm[1][1][3] + 0 * (*CUDA_LCC).Dblm[1][2][3] + 1 * (*CUDA_LCC).Dblm[1][3][3];
		(*CUDA_LCC).de[jp][3][1] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][3][1] = pom0 + tmat * ee0_3;

		tmat = 0 * (*CUDA_LCC).Dblm[2][1][1] + 0 * (*CUDA_LCC).Dblm[2][2][1] + 1 * (*CUDA_LCC).Dblm[2][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = 0 * (*CUDA_LCC).Dblm[2][1][2] + 0 * (*CUDA_LCC).Dblm[2][2][2] + 1 * (*CUDA_LCC).Dblm[2][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = 0 * (*CUDA_LCC).Dblm[2][1][3] + 0 * (*CUDA_LCC).Dblm[2][2][3] + 1 * (*CUDA_LCC).Dblm[2][3][3];
		(*CUDA_LCC).de[jp][3][2] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][3][2] = pom0 + tmat * ee0_3;

		tmat = 0 * (*CUDA_LCC).Blmat[1][1] + 0 * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = 0 * (*CUDA_LCC).Blmat[1][2] + 0 * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = 0 * (*CUDA_LCC).Blmat[1][3] + 0 * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).de[jp][3][3] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][3][3] = pom0 + tmat * ee0_3;
	}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);  //__syncthreads();
}

void bright(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* cg,
	int jp,
	int Lpoints1,
	int Inrel)
{
	double cl, cls, dnom, s, Scale;
	double e_1, e_2, e_3, e0_1, e0_2, e0_3, de[4][4], de0[4][4];
	int ncoef0, ncoef, i, j, incl_count = 0;

	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	ncoef0 = (*CUDA_CC).Ncoef0;//ncoef - 2 - CUDA_Nphpar;
	ncoef = (*CUDA_CC).ma;
	cl = exp(cg[ncoef - 1]); /* Lambert */
	cls = cg[ncoef];       /* Lommel-Seeliger */

	/* matrix from neo */
	/* derivatives */
	e_1 = (*CUDA_LCC).e_1[jp];
	e_2 = (*CUDA_LCC).e_2[jp];
	e_3 = (*CUDA_LCC).e_3[jp];
	e0_1 = (*CUDA_LCC).e0_1[jp];
	e0_2 = (*CUDA_LCC).e0_2[jp];
	e0_3 = (*CUDA_LCC).e0_3[jp];
	de[1][1] = (*CUDA_LCC).de[jp][1][1];
	de[1][2] = (*CUDA_LCC).de[jp][1][2];
	de[1][3] = (*CUDA_LCC).de[jp][1][3];
	de[2][1] = (*CUDA_LCC).de[jp][2][1];
	de[2][2] = (*CUDA_LCC).de[jp][2][2];
	de[2][3] = (*CUDA_LCC).de[jp][2][3];
	de[3][1] = (*CUDA_LCC).de[jp][3][1];
	de[3][2] = (*CUDA_LCC).de[jp][3][2];
	de[3][3] = (*CUDA_LCC).de[jp][3][3];
	de0[1][1] = (*CUDA_LCC).de0[jp][1][1];
	de0[1][2] = (*CUDA_LCC).de0[jp][1][2];
	de0[1][3] = (*CUDA_LCC).de0[jp][1][3];
	de0[2][1] = (*CUDA_LCC).de0[jp][2][1];
	de0[2][2] = (*CUDA_LCC).de0[jp][2][2];
	de0[2][3] = (*CUDA_LCC).de0[jp][2][3];
	de0[3][1] = (*CUDA_LCC).de0[jp][3][1];
	de0[3][2] = (*CUDA_LCC).de0[jp][3][2];
	de0[3][3] = (*CUDA_LCC).de0[jp][3][3];

	/*Integrated brightness (phase coeff. used later) */
	double lmu, lmu0, dsmu, dsmu0, sum1, sum10, sum2, sum20, sum3, sum30;
	double br, ar, tmp1, tmp2, tmp3, tmp4, tmp5;
	short int incl[MAX_N_FAC];
	double dbr[MAX_N_FAC];

	br = 0;
	tmp1 = 0;
	tmp2 = 0;
	tmp3 = 0;
	tmp4 = 0;
	tmp5 = 0;

	j = 1;
	for (i = 1; i <= (*CUDA_CC).Numfac; i++, j++)
	{
		lmu = e_1 * (*CUDA_CC).Nor[i][0] + e_2 * (*CUDA_CC).Nor[i][1] + e_3 * (*CUDA_CC).Nor[i][2];
		lmu0 = e0_1 * (*CUDA_CC).Nor[i][0] + e0_2 * (*CUDA_CC).Nor[i][1] + e0_3 * (*CUDA_CC).Nor[i][2];

		if ((lmu > TINY) && (lmu0 > TINY))
		{
			dnom = lmu + lmu0;
			s = lmu * lmu0 * (cl + cls / dnom);
			ar = (*CUDA_LCC).Area[j];
			br += ar * s;

			incl[incl_count] = i;
			/* Darea[i] * s * Dg[i][k] == Darea[i] * s * g * Dsph[i][k]
			   == (Area[i] * s) * Dsph[i][k]: fold g into the weight and
			   gather from the one read-only, facet-major Dsph shared by
			   all work-groups instead of the per-context Dg matrix */
			dbr[incl_count] = ar * s;
			incl_count++;

			double lmu0_dnom = lmu0 / dnom;
			dsmu = cls * (lmu0_dnom * lmu0_dnom) + cl * lmu0;
			double lmu_dnom = lmu / dnom;
			dsmu0 = cls * (lmu_dnom * lmu_dnom) + cl * lmu;


			sum1 = (*CUDA_CC).Nor[i][0] * de[1][1] + (*CUDA_CC).Nor[i][1] * de[2][1] + (*CUDA_CC).Nor[i][2] * de[3][1];
			sum10 = (*CUDA_CC).Nor[i][0] * de0[1][1] + (*CUDA_CC).Nor[i][1] * de0[2][1] + (*CUDA_CC).Nor[i][2] * de0[3][1];
			tmp1 += ar * (dsmu * sum1 + dsmu0 * sum10);
			sum2 = (*CUDA_CC).Nor[i][0] * de[1][2] + (*CUDA_CC).Nor[i][1] * de[2][2] + (*CUDA_CC).Nor[i][2] * de[3][2];
			sum20 = (*CUDA_CC).Nor[i][0] * de0[1][2] + (*CUDA_CC).Nor[i][1] * de0[2][2] + (*CUDA_CC).Nor[i][2] * de0[3][2];
			tmp2 += ar * (dsmu * sum2 + dsmu0 * sum20);
			sum3 = (*CUDA_CC).Nor[i][0] * de[1][3] + (*CUDA_CC).Nor[i][1] * de[2][3] + (*CUDA_CC).Nor[i][2] * de[3][3];
			sum30 = (*CUDA_CC).Nor[i][0] * de0[1][3] + (*CUDA_CC).Nor[i][1] * de0[2][3] + (*CUDA_CC).Nor[i][2] * de0[3][3];
			tmp3 += ar * (dsmu * sum3 + dsmu0 * sum30);

			tmp4 += lmu * lmu0 * ar;
			tmp5 += ar * lmu * lmu0 / (lmu + lmu0);
		}
	}

	Scale = (*CUDA_LCC).jp_Scale[jp];
	i = (jp - 1) * DYT_STRIDE + (ncoef0 - 3 + 1);
	/* Ders. of brightness w.r.t. rotation parameters */
	(*CUDA_LCC).dytemp[i] = Scale * tmp1;

	i++;
	(*CUDA_LCC).dytemp[i] = Scale * tmp2;
	i++;
	(*CUDA_LCC).dytemp[i] = Scale * tmp3;

	i++;
	/* Ders. of br. w.r.t. phase function params. */
	(*CUDA_LCC).dytemp[i] = br * (*CUDA_LCC).jp_dphp_1[jp];
	i++;
	(*CUDA_LCC).dytemp[i] = br * (*CUDA_LCC).jp_dphp_2[jp];
	i++;
	(*CUDA_LCC).dytemp[i] = br * (*CUDA_LCC).jp_dphp_3[jp];

	/* Ders. of br. w.r.t. cl, cls */
	(*CUDA_LCC).dytemp[(jp - 1) * DYT_STRIDE + (ncoef - 1)] = Scale * tmp4 * cl;
	(*CUDA_LCC).dytemp[(jp - 1) * DYT_STRIDE + (ncoef)] = Scale * tmp5;

	/* Scaled brightness */
	(*CUDA_LCC).ytemp[jp] = br * Scale;

	ncoef0 -= 3;
	int iStart;
	int d, d1, dr;

	iStart = Inrel + 1;
	d = (jp - 1) * DYT_STRIDE + iStart;

	d1 = d + 1;
	dr = 2;

	/* Derivatives of brightness w.r.t. g-coeffs */
	if (incl_count)
	{
		for (i = iStart; i <= ncoef0; i += 2, d += dr, d1 += dr)
		{
			double tmp = 0, tmp1 = 0;
			double l_dbr = dbr[0];
			int l_incl = incl[0];
			tmp = l_dbr * (*CUDA_CC).Dsph[l_incl][i];
			if ((i + 1) <= ncoef0)
			{
				tmp1 = l_dbr * (*CUDA_CC).Dsph[l_incl][i + 1];
			}

			for (j = 1; j < incl_count; j++)
			{
				double l_dbr = dbr[j];
				int l_incl = incl[j];
				tmp += l_dbr * (*CUDA_CC).Dsph[l_incl][i];
				if ((i + 1) <= ncoef0)
				{
					tmp1 += l_dbr * (*CUDA_CC).Dsph[l_incl][i + 1];
				}
			}

			(*CUDA_LCC).dytemp[d] = Scale * tmp;
			if ((i + 1) <= ncoef0)
			{
				(*CUDA_LCC).dytemp[d1] = Scale * tmp1;
			}
		}
	}
	else
	{
		for (i = 1; i <= ncoef0; i++, d++)
			(*CUDA_LCC).dytemp[d] = 0;
	}

	//return(0);
}


/* ====================================================================== */
/* 2026 work-group-cooperative curve1.                                    */
/*                                                                        */
/* The old path gave each work-item one data point and had it walk all    */
/* Numfac facets alone, keeping per-point visible-facet lists in private  */
/* arrays (short incl[MAX_N_FAC], double dbr[MAX_N_FAC]) - ~10 KB of      */
/* scratch per work-item that can never live in registers, measured with  */
/* CL_KERNEL_PRIVATE_MEM_SIZE. The whole work-group now cooperates on one */
/* point at a time:                                                       */
/*                                                                        */
/*   - the per-point geometry (the former matrix_neo pass, and its        */
/*     de, de0, e and jp per-point global buffers) is computed for       */
/*     GEO_BATCH points at a time, one work-item per point, into local    */
/*     memory;                                                            */
/*   - the facet pass strides the facets across the work-group and        */
/*     stores each facet weight (zero when invisible) to a local array -  */
/*     deterministic, no compaction races;                                */
/*   - the six per-point sums reduce through a fixed-order local-memory   */
/*     tree;                                                              */
/*   - the derivative pass runs one work-item per parameter column:       */
/*     Dsph reads coalesce across the group, the transposed dytemp row    */
/*     write coalesces, and the dave column sums fall out for free.       */
/*                                                                        */
/* Only barrier(CLK_LOCAL_MEM_FENCE) is used - no sub-group or wave-size  */
/* assumptions - and every reduction order is fixed, so results are       */
/* deterministic and identical on wave32 (RDNA) and wave64 (GCN).         */
/* ====================================================================== */

/* per-point geometry layout in local memory (GEO_SIZE doubles):
   0..8   de[1..3][1..3]  ((r-1)*3 + c-1)
   9..17  de0[1..3][1..3]
   18..20 e_1..e_3        21..23 e0_1..e0_3
   24 Scale   25 dphp1   26 dphp2   27 dphp3 */

void bright_point_geometry(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* cg,
	int lnp,
	__local double* g)
{
	__private double f, cf, sf, pom, pom0, alpha;
	__private double ee_1, ee_2, ee_3, ee0_1, ee0_2, ee0_3, t, tmat;

	ee_1 = (*CUDA_CC).ee[lnp][0];
	ee0_1 = (*CUDA_CC).ee0[lnp][0];
	ee_2 = (*CUDA_CC).ee[lnp][1];
	ee0_2 = (*CUDA_CC).ee0[lnp][1];
	ee_3 = (*CUDA_CC).ee[lnp][2];
	ee0_3 = (*CUDA_CC).ee0[lnp][2];
	t = (*CUDA_CC).tim[lnp];

	/* ee and ee0 are unit vectors, so the dot product is mathematically in
	   [-1, 1]; near opposition it lands within ~1e-7 of 1.0 and an
	   out-of-range rounding would turn this point - and every trial
	   frequency using it - into NaN (same guard as the CUDA app) */
	alpha = acos(fmin(1.0, fmax(-1.0, ee_1 * ee0_1 + ee_2 * ee0_2 + ee_3 * ee0_3)));

	/* Exp-lin model (const.term=1.) */
	f = exp(-alpha / cg[(*CUDA_CC).Ncoef0 + 2]);
	g[24] = 1 + cg[(*CUDA_CC).Ncoef0 + 1] * f + (cg[(*CUDA_CC).Ncoef0 + 3] * alpha);
	g[25] = f;
	g[26] = cg[(*CUDA_CC).Ncoef0 + 1] * f * alpha / (cg[(*CUDA_CC).Ncoef0 + 2] * cg[(*CUDA_CC).Ncoef0 + 2]);
	g[27] = alpha;

	//  matrix start
	f = cg[(*CUDA_CC).Ncoef0] * t + (*CUDA_CC).Phi_0;
	f = fmod(f, 2 * PI);
	sf = sincos(f, &cf);

	/* rotation matrix, Z axis, angle f; same expressions as the old
	   matrix_neo, only the outputs go to local memory */
	tmat = cf * (*CUDA_LCC).Blmat[1][1] + sf * (*CUDA_LCC).Blmat[2][1];
	pom = tmat * ee_1;
	pom0 = tmat * ee0_1;
	tmat = cf * (*CUDA_LCC).Blmat[1][2] + sf * (*CUDA_LCC).Blmat[2][2];
	pom += tmat * ee_2;
	pom0 += tmat * ee0_2;
	tmat = cf * (*CUDA_LCC).Blmat[1][3] + sf * (*CUDA_LCC).Blmat[2][3];
	g[18] = pom + tmat * ee_3;
	g[21] = pom0 + tmat * ee0_3;

	tmat = (-sf) * (*CUDA_LCC).Blmat[1][1] + cf * (*CUDA_LCC).Blmat[2][1];
	pom = tmat * ee_1;
	pom0 = tmat * ee0_1;
	tmat = (-sf) * (*CUDA_LCC).Blmat[1][2] + cf * (*CUDA_LCC).Blmat[2][2];
	pom += tmat * ee_2;
	pom0 += tmat * ee0_2;
	tmat = (-sf) * (*CUDA_LCC).Blmat[1][3] + cf * (*CUDA_LCC).Blmat[2][3];
	g[19] = pom + tmat * ee_3;
	g[22] = pom0 + tmat * ee0_3;

	tmat = (*CUDA_LCC).Blmat[3][1];
	pom = tmat * ee_1;
	pom0 = tmat * ee0_1;
	tmat = (*CUDA_LCC).Blmat[3][2];
	pom += tmat * ee_2;
	pom0 += tmat * ee0_2;
	tmat = (*CUDA_LCC).Blmat[3][3];
	g[20] = pom + tmat * ee_3;
	g[23] = pom0 + tmat * ee0_3;

	/* de[.][1], de0[.][1]: w.r.t. beta */
	tmat = cf * (*CUDA_LCC).Dblm[1][1][1] + sf * (*CUDA_LCC).Dblm[1][2][1];
	pom = tmat * ee_1;
	pom0 = tmat * ee0_1;
	tmat = cf * (*CUDA_LCC).Dblm[1][1][2] + sf * (*CUDA_LCC).Dblm[1][2][2];
	pom += tmat * ee_2;
	pom0 += tmat * ee0_2;
	tmat = cf * (*CUDA_LCC).Dblm[1][1][3] + sf * (*CUDA_LCC).Dblm[1][2][3];
	g[0] = pom + tmat * ee_3;
	g[9] = pom0 + tmat * ee0_3;

	/* de[.][2], de0[.][2]: w.r.t. lambda */
	tmat = cf * (*CUDA_LCC).Dblm[2][1][1] + sf * (*CUDA_LCC).Dblm[2][2][1];
	pom = tmat * ee_1;
	pom0 = tmat * ee0_1;
	tmat = cf * (*CUDA_LCC).Dblm[2][1][2] + sf * (*CUDA_LCC).Dblm[2][2][2];
	pom += tmat * ee_2;
	pom0 += tmat * ee0_2;
	tmat = cf * (*CUDA_LCC).Dblm[2][1][3] + sf * (*CUDA_LCC).Dblm[2][2][3];
	g[1] = pom + tmat * ee_3;
	g[10] = pom0 + tmat * ee0_3;

	/* de[.][3], de0[.][3]: w.r.t. the rotation rate (angle = omega*t) */
	tmat = (-t * sf) * (*CUDA_LCC).Blmat[1][1] + (t * cf) * (*CUDA_LCC).Blmat[2][1];
	pom = tmat * ee_1;
	pom0 = tmat * ee0_1;
	tmat = (-t * sf) * (*CUDA_LCC).Blmat[1][2] + (t * cf) * (*CUDA_LCC).Blmat[2][2];
	pom += tmat * ee_2;
	pom0 += tmat * ee0_2;
	tmat = (-t * sf) * (*CUDA_LCC).Blmat[1][3] + (t * cf) * (*CUDA_LCC).Blmat[2][3];
	g[2] = pom + tmat * ee_3;
	g[11] = pom0 + tmat * ee0_3;

	tmat = -sf * (*CUDA_LCC).Dblm[1][1][1] + cf * (*CUDA_LCC).Dblm[1][2][1];
	pom = tmat * ee_1;
	pom0 = tmat * ee0_1;
	tmat = -sf * (*CUDA_LCC).Dblm[1][1][2] + cf * (*CUDA_LCC).Dblm[1][2][2];
	pom += tmat * ee_2;
	pom0 += tmat * ee0_2;
	tmat = -sf * (*CUDA_LCC).Dblm[1][1][3] + cf * (*CUDA_LCC).Dblm[1][2][3];
	g[3] = pom + tmat * ee_3;
	g[12] = pom0 + tmat * ee0_3;

	tmat = -sf * (*CUDA_LCC).Dblm[2][1][1] + cf * (*CUDA_LCC).Dblm[2][2][1];
	pom = tmat * ee_1;
	pom0 = tmat * ee0_1;
	tmat = -sf * (*CUDA_LCC).Dblm[2][1][2] + cf * (*CUDA_LCC).Dblm[2][2][2];
	pom += tmat * ee_2;
	pom0 += tmat * ee0_2;
	tmat = -sf * (*CUDA_LCC).Dblm[2][1][3] + cf * (*CUDA_LCC).Dblm[2][2][3];
	g[4] = pom + tmat * ee_3;
	g[13] = pom0 + tmat * ee0_3;

	tmat = (-t * cf) * (*CUDA_LCC).Blmat[1][1] + (-t * sf) * (*CUDA_LCC).Blmat[2][1];
	pom = tmat * ee_1;
	pom0 = tmat * ee0_1;
	tmat = (-t * cf) * (*CUDA_LCC).Blmat[1][2] + (-t * sf) * (*CUDA_LCC).Blmat[2][2];
	pom += tmat * ee_2;
	pom0 += tmat * ee0_2;
	tmat = (-t * cf) * (*CUDA_LCC).Blmat[1][3] + (-t * sf) * (*CUDA_LCC).Blmat[2][3];
	g[5] = pom + tmat * ee_3;
	g[14] = pom0 + tmat * ee0_3;

	tmat = (*CUDA_LCC).Dblm[1][3][1];
	pom = tmat * ee_1;
	pom0 = tmat * ee0_1;
	tmat = (*CUDA_LCC).Dblm[1][3][2];
	pom += tmat * ee_2;
	pom0 += tmat * ee0_2;
	tmat = (*CUDA_LCC).Dblm[1][3][3];
	g[6] = pom + tmat * ee_3;
	g[15] = pom0 + tmat * ee0_3;

	tmat = (*CUDA_LCC).Dblm[2][3][1];
	pom = tmat * ee_1;
	pom0 = tmat * ee0_1;
	tmat = (*CUDA_LCC).Dblm[2][3][2];
	pom += tmat * ee_2;
	pom0 += tmat * ee0_2;
	tmat = (*CUDA_LCC).Dblm[2][3][3];
	g[7] = pom + tmat * ee_3;
	g[16] = pom0 + tmat * ee0_3;

	g[8] = 0.0;
	g[17] = 0.0;
}

void bright_curve1_wg(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* cg,
	int Inrel,
	int Lpoints,
	__local double* wAll,   /* [2 * (MAX_N_FAC + 1)] facet weights for a point pair */
	__local double* geoB,   /* [GEO_BATCH * GEO_SIZE] staged geometry */
	__local double* red)    /* [6 * BLOCK_DIM] reduction scratch */
{
	int3 threadIdx;
	threadIdx.x = get_local_id(0);
	const int tid = threadIdx.x;

	const int nf = (*CUDA_CC).Numfac;
	const int nc0 = (*CUDA_CC).Ncoef0;
	const int nshape = nc0 - 3;
	const int ma = (*CUDA_CC).ma;
	const int iStart = Inrel + 1;
	const int lnp0 = (*CUDA_LCC).np;

	const double cl = exp(cg[ma - 1]); /* Lambert */
	const double cls = cg[ma];         /* Lommel-Seeliger */

	/* derivative pass: threads 0..63 own point A's columns, threads 64..127
	   own point B's - both halves of the group stay busy */
	const int myhalf = tid >> 6;
	const int c = iStart + (tid & 63);
	double davec[2];
	davec[0] = 0; davec[1] = 0;
	double lave = 0;

	__local double* wA = wAll;
	__local double* wB = wAll + (MAX_N_FAC + 1);

	int jp0, p, f, k;

	for (jp0 = 1; jp0 <= Lpoints; jp0 += GEO_BATCH)
	{
		int nb = Lpoints - jp0 + 1;
		if (nb > GEO_BATCH) nb = GEO_BATCH;

		if (tid < nb)
			bright_point_geometry(CUDA_LCC, CUDA_CC, cg, lnp0 + jp0 + tid, &geoB[tid * GEO_SIZE]);
		barrier(CLK_LOCAL_MEM_FENCE);

		/* two points per sweep: every facet's normal and area are loaded
		   once and feed both points' sums */
		for (p = 0; p < nb; p += 2)
		{
			const int jpA = jp0 + p;
			const int haveB = (p + 1 < nb);
			__local const double* gA = &geoB[p * GEO_SIZE];
			__local const double* gB = &geoB[(p + (haveB ? 1 : 0)) * GEO_SIZE];

			double brA = 0, t1A = 0, t2A = 0, t3A = 0, t4A = 0, t5A = 0;
			double brB = 0, t1B = 0, t2B = 0, t3B = 0, t4B = 0, t5B = 0;
			for (f = 1 + tid; f <= nf; f += BLOCK_DIM)
			{
				const double n0 = (*CUDA_CC).Nor[f][0];
				const double n1 = (*CUDA_CC).Nor[f][1];
				const double n2 = (*CUDA_CC).Nor[f][2];
				const double ar = (*CUDA_LCC).Area[f];

				{
					double lmu = gA[18] * n0 + gA[19] * n1 + gA[20] * n2;
					double lmu0 = gA[21] * n0 + gA[22] * n1 + gA[23] * n2;
					double w = 0.0;
					if ((lmu > TINY) && (lmu0 > TINY))
					{
						double dnom = lmu + lmu0;
						double s = lmu * lmu0 * (cl + cls / dnom);
						brA += ar * s;
						w = ar * s;
						double lmu0_dnom = lmu0 / dnom;
						double dsmu = cls * (lmu0_dnom * lmu0_dnom) + cl * lmu0;
						double lmu_dnom = lmu / dnom;
						double dsmu0 = cls * (lmu_dnom * lmu_dnom) + cl * lmu;
						double sum1 = n0 * gA[0] + n1 * gA[3] + n2 * gA[6];
						double sum10 = n0 * gA[9] + n1 * gA[12] + n2 * gA[15];
						double sum2 = n0 * gA[1] + n1 * gA[4] + n2 * gA[7];
						double sum20 = n0 * gA[10] + n1 * gA[13] + n2 * gA[16];
						double sum3 = n0 * gA[2] + n1 * gA[5] + n2 * gA[8];
						double sum30 = n0 * gA[11] + n1 * gA[14] + n2 * gA[17];
						t1A += ar * (dsmu * sum1 + dsmu0 * sum10);
						t2A += ar * (dsmu * sum2 + dsmu0 * sum20);
						t3A += ar * (dsmu * sum3 + dsmu0 * sum30);
						t4A += lmu * lmu0 * ar;
						t5A += ar * lmu * lmu0 / (lmu + lmu0);
					}
					wA[f] = w;
				}
				if (haveB)
				{
					double lmu = gB[18] * n0 + gB[19] * n1 + gB[20] * n2;
					double lmu0 = gB[21] * n0 + gB[22] * n1 + gB[23] * n2;
					double w = 0.0;
					if ((lmu > TINY) && (lmu0 > TINY))
					{
						double dnom = lmu + lmu0;
						double s = lmu * lmu0 * (cl + cls / dnom);
						brB += ar * s;
						w = ar * s;
						double lmu0_dnom = lmu0 / dnom;
						double dsmu = cls * (lmu0_dnom * lmu0_dnom) + cl * lmu0;
						double lmu_dnom = lmu / dnom;
						double dsmu0 = cls * (lmu_dnom * lmu_dnom) + cl * lmu;
						double sum1 = n0 * gB[0] + n1 * gB[3] + n2 * gB[6];
						double sum10 = n0 * gB[9] + n1 * gB[12] + n2 * gB[15];
						double sum2 = n0 * gB[1] + n1 * gB[4] + n2 * gB[7];
						double sum20 = n0 * gB[10] + n1 * gB[13] + n2 * gB[16];
						double sum3 = n0 * gB[2] + n1 * gB[5] + n2 * gB[8];
						double sum30 = n0 * gB[11] + n1 * gB[14] + n2 * gB[17];
						t1B += ar * (dsmu * sum1 + dsmu0 * sum10);
						t2B += ar * (dsmu * sum2 + dsmu0 * sum20);
						t3B += ar * (dsmu * sum3 + dsmu0 * sum30);
						t4B += lmu * lmu0 * ar;
						t5B += ar * lmu * lmu0 / (lmu + lmu0);
					}
					wB[f] = w;
				}
			}

			/* reduce point A's six sums, then point B's (fixed-order tree) */
			for (int pt = 0; pt < (haveB ? 2 : 1); pt++)
			{
				red[0 * BLOCK_DIM + tid] = pt ? brB : brA;
				red[1 * BLOCK_DIM + tid] = pt ? t1B : t1A;
				red[2 * BLOCK_DIM + tid] = pt ? t2B : t2A;
				red[3 * BLOCK_DIM + tid] = pt ? t3B : t3A;
				red[4 * BLOCK_DIM + tid] = pt ? t4B : t4A;
				red[5 * BLOCK_DIM + tid] = pt ? t5B : t5A;
				barrier(CLK_LOCAL_MEM_FENCE);
				for (k = BLOCK_DIM >> 1; k > 0; k >>= 1)
				{
					if (tid < k)
					{
						red[0 * BLOCK_DIM + tid] += red[0 * BLOCK_DIM + tid + k];
						red[1 * BLOCK_DIM + tid] += red[1 * BLOCK_DIM + tid + k];
						red[2 * BLOCK_DIM + tid] += red[2 * BLOCK_DIM + tid + k];
						red[3 * BLOCK_DIM + tid] += red[3 * BLOCK_DIM + tid + k];
						red[4 * BLOCK_DIM + tid] += red[4 * BLOCK_DIM + tid + k];
						red[5 * BLOCK_DIM + tid] += red[5 * BLOCK_DIM + tid + k];
					}
					barrier(CLK_LOCAL_MEM_FENCE);
				}
				/* park this point's reduced sums in the unused tail of its
				   weight array (facets are 1-based and nf + 6 < MAX_N_FAC) so
				   the next point's tree cannot clobber them */
				if (tid < 6)
					(pt ? wB : wA)[nf + 1 + tid] = red[tid * BLOCK_DIM];
				if (tid == 0)
				{
					__local const double* gg = pt ? gB : gA;
					const double ymod = red[0 * BLOCK_DIM] * gg[24];
					(*CUDA_LCC).ytemp[jpA + pt] = ymod;
					lave += ymod;
				}
				barrier(CLK_LOCAL_MEM_FENCE);
			}

			/* derivative rows for both points at once; four independent
			   accumulators break the serial dependency chain */
			{
				const int mypt = (haveB && myhalf) ? 1 : 0;
				const int jp = jpA + mypt;
				__local const double* g = mypt ? gB : gA;
				__local const double* w = mypt ? wB : wA;
				const int active = (myhalf == 0) || haveB;

				if (active && c <= ma)
				{
					double v;
					if (c <= nshape)
					{
						double a0 = 0, a1 = 0, a2 = 0, a3 = 0;
						int fe = nf - 3;
						for (f = 1; f <= fe; f += 4)
						{
							a0 += w[f] * (*CUDA_CC).Dsph[f][c];
							a1 += w[f + 1] * (*CUDA_CC).Dsph[f + 1][c];
							a2 += w[f + 2] * (*CUDA_CC).Dsph[f + 2][c];
							a3 += w[f + 3] * (*CUDA_CC).Dsph[f + 3][c];
						}
						for (; f <= nf; f++)
							a0 += w[f] * (*CUDA_CC).Dsph[f][c];
						v = g[24] * ((a0 + a1) + (a2 + a3));
					}
					else if (c == nshape + 1) v = g[24] * w[nf + 2];
					else if (c == nshape + 2) v = g[24] * w[nf + 3];
					else if (c == nshape + 3) v = g[24] * w[nf + 4];
					else if (c == nc0 + 1) v = w[nf + 1] * g[25];
					else if (c == nc0 + 2) v = w[nf + 1] * g[26];
					else if (c == nc0 + 3) v = w[nf + 1] * g[27];
					else if (c == ma - 1) v = g[24] * w[nf + 5] * cl;
					else v = g[24] * w[nf + 6];

					(*CUDA_LCC).dytemp[(jp - 1) * DYT_STRIDE + c] = v;
					davec[mypt] += v;
				}
			}
			/* wAll/red/geo slots are reused by the next pair */
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}

	/* dave: column sums per point-parity, combined A-half plus B-half.
	   Threads 0..63 hold the sums of even-indexed points of each pair,
	   threads 64..127 the odd ones; combine through local memory. */
	red[tid] = davec[myhalf];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (Inrel == 1 && myhalf == 0 && c <= ma)
	{
		(*CUDA_LCC).dave[c] = red[tid] + red[tid + 64];
	}
	if (tid == 0)
	{
		(*CUDA_LCC).np = lnp0 + Lpoints;
		if (Inrel == 1)
			(*CUDA_LCC).ave = lave;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}

