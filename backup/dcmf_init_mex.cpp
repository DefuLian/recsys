
#if !defined(_WIN32)
#define dgemm dgemm_
#endif

#define d(nn, kk) D[(nn) + (kk)*n]
#define ds(k1, k2) Ds[(k1) + (k2) * k]
#define lambda(eps) tanh((eps)/2)/(eps)/4

#include "mex.h"
#include "string.h"
#include "math.h"

//int main(int argc, char** argv)

// usage dcmf(r_i, b_i, D_i, x_i, Ds, iter, beta, alpha, err, eps);
// assume user i have n non-zero entries, and length of b_i is k, 
// D_i is of size n x k
// r_i is of size n x 1
// x_i is of size k x 1
// err store errors of non-zero entries
// eps store prediction of non-zero entries
// Ds = D.' *D, of size k x k

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *r, *D, *b, *x, *err, *eps, *Ds;
	double alpha, beta, maxItr;
	mwSize n, k;
	
	r = mxGetPr(prhs[0]);
	n = mxGetM(prhs[0]); // modified
	b = mxGetPr(prhs[1]);
	k = mxGetN(prhs[1]);
	D = mxGetPr(prhs[2]);
	x = mxGetPr(prhs[3]);
	Ds = mxGetPr(prhs[4]);
	maxItr = mxGetScalar(prhs[5]);
	beta = mxGetScalar(prhs[6]);
	alpha = mxGetScalar(prhs[7]);
	err = mxGetPr(prhs[8]);
	if(nrhs > 9){
		eps = mxGetPr(prhs[9]);
	}
	bool converge = false;
	mwSize it;
	
	plhs[0] = mxCreateDoubleMatrix(k, 1, mxREAL);
	double *b_new = mxGetPr(plhs[0]);
	memcpy(b_new, b, k * sizeof(double));
	
	while (!converge){
		double tol_val = 0;
        for (mwSize k_iter = 0; k_iter < k; k_iter ++){
            double ss = 0;
			double dom = alpha;
			for(mwSize n_iter = 0; n_iter < n; ++n_iter){
				if(nrhs > 9){ // classification case
					ss += (err[n_iter] + lambda(eps[n_iter]) * b[k_iter] * d(n_iter, k_iter)) * d(n_iter, k_iter);
					dom += lambda(eps[n_iter]) * d(n_iter, k_iter) * d(n_iter, k_iter);
				}
				else{ // regression case
					ss += (err[n_iter] + b[k_iter] * d(n_iter, k_iter))*d(n_iter, k_iter);
					dom += d(n_iter, k_iter) * d(n_iter, k_iter);
				}
				
			}
			
			ss += x[k_iter];
			
			if(beta > 1e-10){
				for (mwSize kk = 0; kk < k; ++kk){
					if (kk != k_iter){
						ss -=  beta * b[kk] * ds(kk, k_iter);
					}
				}
				dom += beta * ds(k_iter, k_iter);
			}
			
			
			double bb_old = b[k_iter];
			double bb_new = ss / dom;
			
			for(mwSize n_iter = 0; n_iter < n; ++n_iter){
				if(nrhs > 9){ // classification case
					double new_eps = eps[n_iter] + (bb_new - bb_old) * d(n_iter, k_iter);
					err[n_iter] = err[n_iter] + (lambda(eps[n_iter]) * bb_old - new_eps * bb_new) * d(n_iter, k_iter);
					eps[n_iter] = new_eps;
				}else{ // regression case
					err[n_iter] = err[n_iter] + (bb_old - bb_new) * d(n_iter, k_iter);
				}
			}
			b[k_iter] = bb_new;
			tol_val += fabs(bb_new - bb_old);
        }
		
		if ((it >= (int)maxItr-1) || (tol_val <= 1e-6))
            converge = true;
        it ++;
	}
	
	memcpy(b_new, b, k * sizeof(double));
}