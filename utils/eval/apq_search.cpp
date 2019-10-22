#define Cu(b, w) center_score[ (b) + (w) * B ]
#define U(i, d) user[ (i) + (d) * M]
#define I(j, b) code[ (j) + (b) * N] - 1
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
// result = apq_search(user, code, center, train, k)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	double * user = mxGetPr(prhs[0]);
	size_t M = mxGetM(prhs[0]);
	size_t D = mxGetN(prhs[0]);
	uint32_t * code = (uint32_t *)mxGetData(prhs[1]);
	size_t N = mxGetM(prhs[1]);
	size_t B = mxGetN(prhs[1]);
	double* center = mxGetPr(prhs[2]);
	size_t W = mxGetM(prhs[2]);
	size_t F = mxGetN(prhs[2]);
	mwIndex *Ir = mxGetIr(prhs[3]);
	mwIndex *Jc = mxGetJc(prhs[3]);
	size_t K = (size_t)mxGetScalar(prhs[4]);

	plhs[0] = mxCreateNumericMatrix(M, K, mxUINT32_CLASS, mxREAL);
	uint32_t *result = (uint32_t *)mxGetData(plhs[0]);

	vector<pair<double, size_t>> pred(N, make_pair(0,-1));
	double* center_score = new double[B * W];
	size_t Ds = F / B;
	for (size_t i = 0; i < M; ++i) {
		for (size_t b = 0; b < B; ++b) {
			for (size_t w = 0; w < W; ++w) {
				Cu(b, w) = 0;
				for (size_t d = 0; d < Ds; ++d) {
					Cu(b, w) += U(i, (d + b * Ds) % D) * center[(d + b * Ds) * W + w];
				}
			}
		}

		for (size_t j = 0; j < N; ++j) {
			double val = 0;
			for (size_t b = 0; b < B; ++b) {
				val += Cu(b, I(j, b));
			}
			pred[j] = make_pair(-val, j + 1);
		}
		size_t start = Jc[i];
		size_t end = Jc[i + 1];
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
	delete[] center_score;

}