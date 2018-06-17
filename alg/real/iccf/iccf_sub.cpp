
#if !defined(_WIN32)
#define dgemm dgemm_
#endif

#define p(u, k) P[(u) + (k) * M]
#define q(i, k) Q[Ir[(i)] + (k) * N]
#define x(u, k) XU[(u) + (k) * M]
#define e(i) err[(i) - start_row_index]
#define qs(k1, k2) Qs[(k1) + (k2) * K]
#define pn(u,k) P_new[(u) + (k) * M]


#include "mex.h"
#include "string.h"
//int main(int argc, char** argv)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

	mwIndex *Ir, *Jc;
	double *r, *w, *Q, *P, *XU, *a, *d, *Qs, *bI;
	mwSize M, N, K;
	double reg;

	Ir = mxGetIr(prhs[0]);
	Jc = mxGetJc(prhs[0]);
	r = mxGetPr(prhs[0]);
	N = mxGetM(prhs[0]);
	M = mxGetN(prhs[0]);
	w = mxGetPr(prhs[1]);
	Q = mxGetPr(prhs[2]);
	K = mxGetN(prhs[2]);
	P = mxGetPr(prhs[3]);
	XU = mxGetPr(prhs[4]);
	reg = mxGetScalar(prhs[5]);
	a = mxGetPr(prhs[6]);
	d = mxGetPr(prhs[7]);
	Qs = mxGetPr(prhs[8]);
    bI = mxGetPr(prhs[9]);
    double* bR = new double[K];
    for(size_t k=0;k<K;++k)
    {
        bR[k] = 0;
        for(size_t i = 0;i<N; ++i)
        {
            bR[k] += Q[i + k * N] * d[i] * bI[i];
        }
    }
	//mexPrintf("%d, %d, %d, %d, %d, %d", M, N, K, mxGetM(prhs[2]), mxGetM(prhs[3]), mxGetM(prhs[8]));

	plhs[0] = mxCreateDoubleMatrix(M, K, mxREAL);
	double *P_new = mxGetPr(plhs[0]);

	for (size_t u = 0; u < M; ++u)
	{
		size_t start_row_index = Jc[u];
		size_t end_row_index = Jc[u + 1];
		double au = a[u];

		//if (end_row_index <= start_row_index)
		//	continue;
		double* err = new double[end_row_index - start_row_index];
		for (size_t i = start_row_index; i < end_row_index; ++i)
		{
			e(i) = r[i] - bI[Ir[i]];
			for (size_t k = 0; k < K; ++k){
				e(i) -= p(u, k) * q(i, k);
			}
		}

		for (size_t k = 0; k < K; ++k)
		{
			double numer = 0;
			for (size_t i = start_row_index; i < end_row_index; ++i)
			{
				numer += w[i] * (e(i) * q(i, k) + p(u, k)* q(i, k) * q(i, k)) + au * d[Ir[i]] * r[i] * q(i, k);

			}
			for (size_t kk = 0; kk < K; ++kk){
				if (kk != k){
					numer -= au * p(u, kk) * qs(kk, k);
				}
			}
            numer += x(u, k);
            numer -= au * bR[k];
			double dom = au * qs(k, k) + reg;

			for (size_t i = start_row_index; i < end_row_index; ++i)
			{
				dom += w[i] * q(i, k) * q(i, k);
			}
			double new_value = numer / dom;

			for (size_t i = start_row_index; i < end_row_index; ++i)
			{
				e(i) = e(i) + (p(u, k) - new_value) * q(i, k);
			}
			p(u, k) = new_value;

		}

		delete[] err;
	}
    delete[] bR;
	memcpy(P_new, P, M * K * sizeof(double));


}