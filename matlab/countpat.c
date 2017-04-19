#include "mex.h"

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    double *x,*f;
    size_t x1,x2,f1,f2;
    double *c = 0;
    int i,j,k,m,n;

    // input:
    x = mxGetPr(prhs[0]);
    f = mxGetPr(prhs[1]);

    // input dimensions:
    x1 = mxGetM(prhs[0]);
    x2 = mxGetN(prhs[0]);
    f1 = mxGetM(prhs[1]);
    f2 = mxGetN(prhs[1]);
       
    //mexPrintf("x1: %d x2: %d f1: %d f2: %d\n",x1,x2,f1,f2);
    
    // output:
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    c = mxGetPr(plhs[0]);
    
    for (j=0; j<x2-f2+1; j++) {
        for (i=0; i<x1-f1+1; i++) {
            // check for pattern match at x(i,j):
            n = 0;
            m = 0;
            while (m < f1){
                k = 0;
                while (k < f2) {
                    //mexPrintf("i: %d j: %d k: %d m: %d x: %d f: %d\n",i,j,k,m,(int)x[(j+k)*x1 + i + m],(int)f[k*f1+m]);
                    if (((int)x[(j+k)*x1 + i + m] == (int)f[k*f1+m]) || ((int)f[k*f1+m] == 2)) { // 4 is wildcard
                        n++;
                    } else {
                        // break:
                        m = f1;
                        break;
                    }
                    k++;
                    //mexPrintf("n: %d\n",n);
                }
                m++;
            }
            
            if (n == f1*f2) {
                c[0]++;
            }
        }
    }
    return;
}