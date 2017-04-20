/*  Search for the pattern f in x, store the number of hits in x[0] */
void countpat(int * out,int * x, int * f, int x1, int x2, int f1, int f2) {
    int i,j,m,k,n;
	int c = 0;
    for (j=0; j<x2-f2+1; j++) {
        for (i=0; i<x1-f1+1; i++) {
            // check for pattern match at x(i,j):
            n = 0;
            m = 0;
            while (m < f1){
                k = 0;
                while (k < f2) {
                    if ((x[(j+k)*x1 + i + m] == f[k*f1+m]) || (f[k*f1+m] == 2)) { // 4 is wildcard
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
                c++;
            }
        }
    }
	out[0] = c;
}