# We follow this example: 
# http://www.scipy-lectures.org/advanced/interfacing_with_c/interfacing_with_c.html#id13

# Compile like this (needs countpat.h, countpat.c, _countpat.pyx, setup.py):
# python setup.py build_ext -i
    
import numpy as np
import countpat

c = np.zeros(1,dtype = np.int);
7

f = np.array([[0],[1],[1],[0]], dtype=np.int)


f = np.array([[ 2,  2,  2,  2,  2,  0],
              [ 2,  2,  2,  2, -1,  2],
              [ 2,  2,  2, -1,  2,  2],
              [ 2,  2, -1,  2,  2,  2],
              [ 2, -1,  2,  2,  2,  2],
              [ 0,  2,  2,  2,  2,  2]],dtype=np.int)

#f = np.array([[0],[1],[1],[0]], dtype=np.int, order = 'c')

c = np.array([0], dtype=np.int);

#print(x[6,4],x[6,5],x[6,6],x[6,7])#
#print(x)#
#print(f)
#print(f.shape)
#print(f[0,0],f[0,1],f[0,2],f[0,3])
#print(c)
		
#x = np.zeros((13,13),dtype = np.int);
#f = np.zeros((1,10),dtype = np.int);
#x[0] = 10;
countpat.countpat_func(c, T(x), T(f))
print(c)

countpat.countpat_func(c, x, f)
print(c)


countpat.countpat_func(c, x, ff['pattern'][37])
print(c)

gg = g[:]
countpat.countpat_func(c, x, gg)
print(c)

#%%

f,w = getFeatures()

c = countpattern(x,1,f)
