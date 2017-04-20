# we follow http://www.scipy-lectures.org/advanced/interfacing_with_c/interfacing_with_c.html#id13

import numpy as np
import countpat

c = np.zeros(1,dtype = np.int);
x = np.zeros((13,13),dtype = np.int);
f = np.zeros((1,10),dtype = np.int);

countpat.countpat_func(c, x, f)
print(c[0])

