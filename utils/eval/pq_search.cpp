#define x(i, f) X[(f) * M + (i)] - 1
#define y(j, f) Y[(f) * N + (j)] - 1
//#define look(i, j, f) lookup[(y(j, f)) + (f * C + (x(i, f))) * C]
#include "mex.h"
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <queue>
#include <limits>
using std::pair;
using std::vector;
using std::make_pair;
// result = pq_search(X, Y, lookup, train, k)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	uint32_t * X = (uint32_t *)mxGetData(prhs[0]);
	size_t M = mxGetM(prhs[0]);
	size_t F = mxGetN(prhs[0]);
	uint32_t * Y = (uint32_t *)mxGetData(prhs[1]);
	size_t N = mxGetM(prhs[1]);
	double* lookup = mxGetPr(prhs[2]);
	size_t C = mxGetM(prhs[2]);
	mwIndex *Ir = mxGetIr(prhs[3]);
	mwIndex *Jc = mxGetJc(prhs[3]);
	size_t K = (size_t)mxGetScalar(prhs[4]);

	plhs[0] = plhs[0] = mxCreateNumericMatrix(M, K, mxUINT32_CLASS, mxREAL);
	uint32_t *result = (uint32_t *)mxGetData(plhs[0]);

	vector<pair<double, size_t>> pred(N, make_pair(0,-1));

	for (size_t i = 0; i < M; ++i) {
		size_t start = Jc[i];
		size_t end = Jc[i + 1];
		for (size_t j = 0; j < N; ++j) {
			double val = 0;
			for (size_t f = 0; f < F; ++f) {
                size_t jc = y(j,f);
                size_t ic = x(i,f) + f * C;
				val += lookup[jc + ic * C];
			}
			pred[j] = make_pair(-val, j + 1);
		}
		for (size_t row = start; row < end; ++row) {
			size_t j = Ir[row];
			pred[j].first = std::numeric_limits<double>::max();
		}
		std::partial_sort(pred.begin(), pred.begin() + K, pred.end());
		for (int k = 0; k < K; ++k) {
			result[M * k + i] = pred[k].second;
		}
	}
	pred.clear();

}