
#if !defined(_WIN32)
#define dgemm dgemm_
#endif

#define p(u, k) P[(u) + (k) * M]
#define q(i, k) Q[(i) + (k) * N]
#define x(u, k) X[(u) + (k) * M]
#define qs(k1, k2) Qs[(k1) + (k2) * KK]
#define lambda(eps) tanh((eps)/2)/(eps+1e-16)/4
#define err_r(r, p) (r - p)
#define err_c(r, p) ((r)/4 - lambda(p) * (p))
#define pred(index) V[(index) - start_row_index] 

#include "mex.h"
#include "string.h"
#include "math.h"

//int main(int argc, char** argv)

// usage dcmf_init_all_mex(R, Q, P, X, Ds, iter, alpha, is_classify);
// assume user i have n non-zero entries, and length of b_i is k, 
// D_i is of size n x k
// r_i is of size n x 1
// x_i is of size k x 1
// err store errors of non-zero entries
// eps store prediction of non-zero entries
// Ds = D.' *D, of size k x k

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	mwIndex *Ir, *Jc;
	double *R, *Q, *P, *X, *E, *Qs;
	double alpha, maxItr;
	mwSize M, N, KK;
	
	Ir = mxGetIr(prhs[0]);
	Jc = mxGetJc(prhs[0]);
	R = mxGetPr(prhs[0]);
	N = mxGetM(prhs[0]); // modified
	M = mxGetN(prhs[0]);
	Q = mxGetPr(prhs[1]);
	KK = mxGetN(prhs[1]);
	P = mxGetPr(prhs[2]);
	X = mxGetPr(prhs[3]);
	Qs = mxGetPr(prhs[4]);
	bool isreg = mxGetM(prhs[4])!=0;
	maxItr = mxGetScalar(prhs[5]);
	bool is_classify = *mxGetLogicals(prhs[6]);
	
	bool converge = false;
	mwSize it;
	
	plhs[0] = mxCreateDoubleMatrix(M, KK, mxREAL);
	double *P_new = mxGetPr(plhs[0]);
	
	for(mwSize u = 0; u < M; ++u)
	{
		converge = false;

		mwSize start_row_index = Jc[u];
		mwSize end_row_index = Jc[u + 1];
		
		double* V = new double[end_row_index - start_row_index];
		for(mwSize item_index = start_row_index; item_index < end_row_index; ++item_index)
		{
			mwSize i = Ir[item_index];
			pred(item_index) = 0;
			for (mwSize k = 0; k < KK; k ++){
				pred(item_index) += p(u,k) * q(i,k);
			}
			
		}
		
		while (!converge)
		{
			mwSize no_change_count = 0;

			for (mwSize k = 0; k < KK; k ++){
				double ss = 0;
				
				for(mwSize item_index = start_row_index; item_index < end_row_index; ++item_index)
				{
					mwSize i = Ir[item_index];
					double r = R[item_index];
					double v = pred(item_index);
					
					if(is_classify){ // classification case
						ss += (err_c(r,v) + lambda(v) * p(u,k) * q(i,k)) * q(i,k);
						
					}
					else{ // regression case
						ss += (err_r(r,v) + p(u,k) * q(i,k)) * q(i,k);
						
					}
					
				}
				
				ss += x(u,k);
				
				if(isreg){
					for (mwSize k1 = 0; k1 < KK; ++k1){
						if (k1 != k){
							ss -=  p(u,k1) * qs(k1, k);
						}
					}
				}
				//if (u==840)
                //    mexPrintf("%f,",ss);
				double p_new = p(u,k);
                
                if(fabs(ss)>1e-6)
                    if(ss > 0)
                        p_new = 1;
                    else if(ss < 0)
                        p_new = -1;
					
				
				if(fabs(p_new - p(u,k)) > 1e-6) //have changes
				{
					for(mwSize item_index = start_row_index; item_index < end_row_index; ++item_index)
					{
						mwSize i = Ir[item_index];
						pred(item_index) = pred(item_index) + (p_new - p(u,k)) * q(i,k);
					}
					p(u,k) = p_new;
				}
				else
					no_change_count ++;
				
				
				
			}
            //if (u==840)
            //    mexPrintf("\n");
			if ((it >= (int)maxItr-1) || (no_change_count == KK))
				converge = true;
			it ++;
		}
		delete[] V;
	}
	
	
	memcpy(P_new, P, M * KK * sizeof(double));
}