#define err_r(r, p) (r - p)
#define err_c(r, p) ((r)/4 - lambda(p) * (p))
#define lambda(eps) tanh((eps)/2)/(eps+1e-16)/4
#define d(i,f) Du[(i)+n*(f)]
#include <string.h>
#include <math.h>
#include <mex.h>
// ccd_logit_mex(r, Du, b, DtD, x, r_, is_logit,max_iter) given a user u
// DtD is usually set rho * (D'*D - Du'Du)
// r is the rating vector, of size n x 1, n is the number of ratings by user u
// r_ is predicted rating vector of the same size n x 1
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *r   = mxGetPr(prhs[0]);
    double *Du  = mxGetPr(prhs[1]);
    double *b   = mxGetPr(prhs[2]);
    double *DtD = mxGetPr(prhs[3]);
    double *x   = mxGetPr(prhs[4]);
    double *r_  = mxGetPr(prhs[5]);
    bool isreg  = mxGetM(prhs[3])!=0;
    mwSize  n   = mxGetM(prhs[0]);
    mwSize  k   = mxGetN(prhs[1]);
    bool islogit = *mxGetLogicals(prhs[6]);
    double max_iter = *mxGetPr(prhs[7]);
    
    plhs[0] = mxCreateDoubleMatrix(k, 1, mxREAL);
	double *b_new = mxGetPr(plhs[0]);
    
    mwSize it = 0, no_change_count = 0;
    bool converge = false;
    while(!converge)
    {
        no_change_count = 0;
        for(mwSize f=0; f<k; ++f)
        {
            double ss = 0;
            for(mwSize i=0; i<n; ++i)
                if(islogit)
                    ss += (err_c(r[i], r_[i]) + lambda(r_[i]) * b[f] * d(i,f)) * d(i,f);
                else
                    ss += (err_r(r[i], r_[i]) + b[f] * d(i,f)) * d(i,f);
            //mexPrintf("%f,", ss);
            
            ss += x[k];
            //mexPrintf("%f,", ss);
            if(isreg){
                for(mwSize f1=0;f1<k;++f1)
                    if(f1!=f)
                        ss -= b[f1]* DtD[f1+f*k];
            }
            //mexPrintf("%f,", ss);
            double bb = b[f];
            if(fabs(ss)>1e-10)
                if(ss>0)
                    bb = 1;
                else if(ss<0)
                    bb = -1;
            
            if(fabs(bb-b[f])>1e-10)
            {
                for(mwSize i=0; i<n; ++i){
                    r_[i] = r_[i] + (bb - b[f])*d(i,f);
                }
                b[f] = bb;
            }
            else
                no_change_count ++;
            
        }
        //mexPrintf("\n");
        if((it >= (int)max_iter-1) || (no_change_count == k))
            converge = true;
        it ++;
        
    }
    memcpy(b_new, b, k * sizeof(double));
}