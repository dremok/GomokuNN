""" Example of wrapping a C function that takes C int arrays as input using
    the Numpy declarations from Cython """

# cimport the Cython declarations for numpy
cimport numpy as np

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example, but good practice)
np.import_array()

# cdefine the signature of our c function
cdef extern from "countpat.h":
    void countpat(int * out, int * x, int * f, int x1, int x2, int f1, int f2);

# create the wrapper code, with numpy type annotations
def countpat_func(np.ndarray[int, ndim=1, mode="c"] out not None,
				  np.ndarray[int, ndim=2, mode="c"] x not None,
                  np.ndarray[int, ndim=2, mode="c"] f not None):
    countpat(<int*> np.PyArray_DATA(out),
			 <int*> np.PyArray_DATA(x),
             <int*> np.PyArray_DATA(f),
             x.shape[0],x.shape[1],f.shape[0],f.shape[1])