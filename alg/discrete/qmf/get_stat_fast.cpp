#define p(i, d) P[(i) * D + (d)]
#define s(i, k) S[(i) * K + (k)]
#define q(j, d) Q[(j) + N * (d)]
#define t(j, k) T[(j) + N * (k)]
#define qs(d1, d2) Qs[(d1) * D + d2]
#define a(d1, d2) A[(d1) * D + d2]

#include "mex.h"
#include "string.h"
//function [A,b] = get_stat(Rt, Wt, S, P, Q, T, QtQ)
// total dimension is (K+D)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	mwIndex *Ir = mxGetIr(prhs[0]);
	mwIndex *Jc = mxGetJc(prhs[0]);
	mwSize m = mxGetN(prhs[0]);
	mwSize N = mxGetM(prhs[0]);
	double *R = mxGetPr(prhs[0]);
	double *W = mxGetPr(prhs[1]);
	double *S = mxGetPr(prhs[2]); // K x m
	mwSize K = mxGetM(prhs[2]);
	double *P = mxGetPr(prhs[3]); // D x m
	mwSize D = mxGetM(prhs[3]);
	double *Q = mxGetPr(prhs[4]); // N x D
	double *T = mxGetPr(prhs[5]); // N x K
	double *Qs = mxGetPr(prhs[6]);// D x D

	plhs[0] = mxCreateDoubleMatrix(D, D, mxREAL);
	double *A = mxGetPr(plhs[0]);
	plhs[1] = mxCreateDoubleMatrix(D, 1, mxREAL);
	double *b = mxGetPr(plhs[1]);

	for (mwSize i = 0; i < m; ++i) {
		mwSize row_start = Jc[i];
		mwSize row_end = Jc[i + 1];

		for (mwSize d1 = 0; d1 < D; ++d1) {
			for (mwSize d2 = 0; d2 < D; ++d2) {
				a(d1, d2) += qs(d1, d2);
			}
		}

		for (mwSize d = 0; d < D; ++d) {
			b[d] += -p(i, d);
		}

		for (mwSize row = row_start; row < row_end; row++) {
			mwSize j = Ir[row];
			double r = R[row];
			double w = W[row];
			double r_ = 0;
			for (mwSize k = 0; k < K; ++k) {
				r_ += s(i, k) * t(j, k);
			}
			for (mwSize d1 = 0; d1 < D; ++d1) {
				for (mwSize d2 = 0; d2 < D; ++d2) {
					a(d1, d2) += w * q(j, d1) * q(j,d2);
				}
			}

			for (mwSize d = 0; d < D; ++d) {
				b[d] += q(j, d) * ((r - r_) * w + r);
			}

		}


	}

}