
#if !defined(_WIN32)
#define dgemm dgemm_
#endif

#define q(i, k) Q[Ir[(i)] + (k) * N]
#define e(i) err[(i) - start_row_index]
#define qs(k1, k2) Qs[(k1) + (k2) * K]


#include "mex.h"
#include "string.h"
//int main(int argc, char** argv)
//main(r, w, Q, Qs, p, x, a, bR, dr, reg, bI)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

	mwIndex *Ir, *Jc;
	double *r, *w, *Q, *Qs, *p, *x, *bR, *dr, *bI;
	mwSize N, K;
	double reg, au;

	Ir = mxGetIr(prhs[0]);
	Jc = mxGetJc(prhs[0]);
	r = mxGetPr(prhs[0]);
	N = mxGetM(prhs[0]);
	w = mxGetPr(prhs[1]);
	Q = mxGetPr(prhs[2]);
	K = mxGetN(prhs[2]);
	Qs = mxGetPr(prhs[3]);
	p = mxGetPr(prhs[4]);
	x = mxGetPr(prhs[5]);
	au = mxGetScalar(prhs[6]);
	bR = mxGetPr(prhs[7]);
	dr = mxGetPr(prhs[8]);
	reg = mxGetScalar(prhs[9]);
	bI = mxGetPr(prhs[10]);
	
	//mexPrintf("%d, %d, %d, %d, %d, %d", M, N, K, mxGetM(prhs[2]), mxGetM(prhs[3]), mxGetM(prhs[8]));

	plhs[0] = mxCreateDoubleMatrix(K, 1, mxREAL);
	double *p_new = mxGetPr(plhs[0]);

	size_t start_row_index = Jc[0];
	size_t end_row_index = Jc[1];

	double* err = new double[end_row_index - start_row_index];

	for (size_t i = start_row_index; i < end_row_index; ++i)
	{
		e(i) = r[i] - bI[Ir[i]];
		for (size_t k = 0; k < K; ++k){
			e(i) -= p[k] * q(i, k);
		}
	}

	for (size_t k = 0; k < K; ++k)
	{
		double numer = 0;
		for (size_t i = start_row_index; i < end_row_index; ++i)
		{
			numer += w[i] * (e(i) * q(i, k) + p[k]* q(i, k) * q(i, k));

		}
		numer += x[k];
		if (au > 1e-10)
		{
			numer += au * dr[k];
			for (size_t kk = 0; kk < K; ++kk){
				if (kk != k){
					numer -= au * p[kk] * qs(kk, k);
				}
			}
		    numer -= au * bR[k];
		}
		double dom = au * qs(k, k) + reg;

		for (size_t i = start_row_index; i < end_row_index; ++i)
		{
			dom += w[i] * q(i, k) * q(i, k);
		}
		double new_value = numer / dom;

		for (size_t i = start_row_index; i < end_row_index; ++i)
		{
			e(i) = e(i) + (p[k] - new_value) * q(i, k);
		}
		p[k] = new_value;

	}

	delete[] err;

    
	memcpy(p_new, p, K * sizeof(double));


}
