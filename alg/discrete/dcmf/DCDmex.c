#include <string.h>
#include <math.h>
#include <mex.h>
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    double *b;/*nframes(N) x D*/
    double *MM;/*nframes(N) x knn*/
    double *Ms;/*ncodebook(K) x nfeatures(D)*/
    double *x;
    double maxItr;
    mwSize r;
   
    mwSize i,k,it,no_change_count;
    double ss;
    bool converge = false;
    
    b = mxGetPr(prhs[0]);
    MM = mxGetPr(prhs[1]);
    Ms = mxGetPr(prhs[2]);
    x = mxGetPr(prhs[3]);
    maxItr = *(mxGetPr(prhs[4]));
     
    r = mxGetN(prhs[1]);

     
    it = 0;
    
    while (!converge){
        no_change_count = 0;
        for (k = 0; k < r; k ++){
            ss = 0;
            for (i = 0; i < r; i ++)
                if (i != k)
                    ss += MM[k+i*r]*b[i];
            /*mexPrintf("%d,%f",k,ss);*/
            ss -= Ms[k]+x[k];
            /*mexPrintf(",%f,%f,%f,%f\n",ss, Ms[k], x[k], Ms[k]+ x[k]);*/
            if (ss > 0){
                if (b[k] == -1)
                    no_change_count ++;
                else
                    b[k] = -1;
            }
            else if (ss < 0){
                if (b[k] == 1)
                    no_change_count ++;
                else
                    b[k] = 1;
            }
            else
                no_change_count ++;
        }
        if ((it >= (int)maxItr-1) || (no_change_count == r))
            converge = true;
        it ++;
    }
    /*mexPrintf("\n");*/
}
    
