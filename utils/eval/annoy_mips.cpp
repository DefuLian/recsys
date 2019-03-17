#include "mex.h"
#include "annoylib.h"
#include "kissrandom.h"
#include <set>
// [result] = annoy_mips(P, Q, train, k, n_trees, search_k)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	double* P = mxGetPr(prhs[0]);
	double* Q = mxGetPr(prhs[1]);
	size_t M = mxGetN(prhs[0]);
	size_t D = mxGetM(prhs[0]);
	size_t N = mxGetN(prhs[1]);
	mwIndex *Ir = mxGetIr(prhs[2]);
	mwIndex *Jc = mxGetJc(prhs[2]);
	size_t K = (size_t)mxGetScalar(prhs[3]);
	size_t n_trees = (size_t)mxGetScalar(prhs[4]);
	size_t search_k = (size_t)mxGetScalar(prhs[5]);// K * n_trees;

	plhs[0] = mxCreateNumericMatrix(M, K, mxUINT32_CLASS, mxREAL);
	uint32_t *result = (uint32_t *)mxGetData(plhs[0]);

	//AnnoyIndex<uint32_t, double, DotProduct, Kiss64Random> index = AnnoyIndex<uint32_t, double, DotProduct, Kiss64Random>(D);
	AnnoyIndex<uint32_t, double, Euclidean, Kiss64Random> index = AnnoyIndex<uint32_t, double, Euclidean, Kiss64Random>(D);
	for (uint32_t j = 0; j < N; ++j) {
		index.add_item(j + 1, Q + D * j);
	}
	
	index.build(n_trees);

	std::vector<uint32_t> closest;
	std::set<uint32_t> train;
	for (size_t i = 0; i < M; ++i) {
		size_t start = Jc[i];
		size_t end = Jc[i + 1];
		train.insert(Ir + start, Ir + end);
		index.get_nns_by_vector(P + D * i, K + (end - start), search_k, &closest, nullptr);

		for (size_t n = 0, k = 0; n < closest.size() && k < K; ++n) {
			if (train.find(closest[n] - 1) == train.end())
				result[(k++)*M + i] = closest[n];
		}
		closest.clear();
		train.clear();
	}
}