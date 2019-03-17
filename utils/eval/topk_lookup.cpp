#ifdef _WIN32
#include <intrin.h>
#define popcount __popcnt
#elif __linux__
#define popcount __builtin_popcount
#endif
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
// topk_lookup(X, Y, train, k, is_dotp);
// is_dotp=1, inner product
// is_dotp=0, hamming distance

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	bool is_dotp = *mxGetLogicals(prhs[4]);
	mwIndex *Ir = mxGetIr(prhs[2]);
	mwIndex *Jc = mxGetJc(prhs[2]);
	size_t K = mxGetScalar(prhs[3]);
	size_t M = mxGetM(prhs[0]);
	size_t F = mxGetN(prhs[0]);
	size_t N = mxGetM(prhs[1]);
	void * X = mxGetData(prhs[0]);
	void * Y = mxGetData(prhs[1]);

	plhs[0] = mxCreateNumericMatrix(M, K, mxUINT32_CLASS, mxREAL);
	uint32_t *result = (uint32_t *)mxGetData(plhs[0]);
	vector<pair<double, size_t>> pred(N, make_pair(0, -1));

	for (int i = 0; i < M; ++i) {
		size_t start = Jc[i];
		size_t end = Jc[i + 1];
		for (int j = 0; j < N; ++j) {
			double val = 0;
			for (size_t f = 0; f < F; ++f) {
				if (is_dotp) {
					double x = *((double*)X + i + f * M);
					double y = *((double*)Y + j + f * N);
					val += x * y;
				}
				else
				{
					uint32_t x = *((uint32_t*)X + i + f * M);
					uint32_t y = *((uint32_t*)Y + j + f * N);
					val -= popcount(x^y);
				}
			}
			//if (!is_dotp) {
			//	val = F * 32 - 2 * val;
			//}
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
}