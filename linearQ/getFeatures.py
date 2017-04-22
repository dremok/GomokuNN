# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import convolve2d
import countpat

#%%

def getFeatures():

    f_gen = list();
    f_gen.append(np.array([[0,1,1,0]], dtype=np.int))
    f_gen.append(np.array([[0,1,1,1,0]], dtype=np.int))
    f_gen.append(np.array([[0,1,1,1,1,0]], dtype=np.int))
    f_gen.append(np.array([[0,1,0,1,1,0]], dtype=np.int))
    f_gen.append(np.array([[0,-1,-1,0]], dtype=np.int))
    f_gen.append(np.array([[0,-1,-1,-1,-1,0]], dtype=np.int))
    f_gen.append(np.array([[0,-1,-1,0,-1,0]], dtype=np.int))
    f_gen.append(np.array([[-1,-1,0,-1,-1]], dtype=np.int))
    f_gen.append(np.array([[1,-1,-1,-1,-1,0]], dtype=np.int))
    
    w = np.array([0.1360,0.2662,0.9232,0.1967,1.5843,-0.1594,-0.6747,-1.3054,-1.3054,-0.7254,-0.5805,-1.1453])
    
    # diagonal versions:
    f_diag = list();
    w_diag = list();
    for n in range(0,len(f_gen)):
        if (f_gen[n].shape[0] == 1):
            f_ = 2*np.ones((f_gen[n].shape[1],f_gen[n].shape[1]),dtype = np.int)
            f_ = f_ - np.diag(np.diag(f_)) + np.diag(f_gen[n][0])
            f_diag.append(f_)
            w_diag.append(w[n])        
    f_gen = f_gen + f_diag
    w      = np.concatenate((w,w_diag))
    
    def T(x):
        y = np.zeros((x.shape[1],x.shape[0]),dtype = int)
        for i in range(0,x.shape[0]):
            for j in range(0,x.shape[1]):
                y[j,i] = x[i,j]
        return y
    def LR(x):
        y = np.zeros(x.shape,dtype = int)
        for i in range(0,x.shape[0]):
            for j in range(0,x.shape[1]):
                y[i,j] = x[i,x.shape[1]-j-1]
        return y
    def UD(x):
        y = np.zeros(x.shape,dtype = int)
        for i in range(0,x.shape[0]):
            for j in range(0,x.shape[1]):
                y[i,j] = x[x.shape[0]-i-1,j]
        return y
    
    def getSymmetries(x):
        xsym = list()
        xsym.append(x);
        xsym.append(LR(x));
        xsym.append(UD(x));
        xsym.append(LR(UD(x)));
        xsym.append(T(x));
        xsym.append(LR(T(x)));
        xsym.append(UD(T(x)));
        xsym.append(LR(UD(T(x))));
        return xsym
    
    f_all = list()
    w_all = list()
    f_group = list()
    for n in range(0,len(f_gen)):
        fsymAll = getSymmetries(f_gen[n])
        
        # collect the non-reduntant ones:
        fsym = list();
        for j in range(0,len(fsymAll)):
            isRedundant = 0;
            for k in range(0,len(fsym)):
                if (fsym[k].shape == fsymAll[j].shape):
                    if np.all(fsym[k] == fsymAll[j]):
                        isRedundant = 1
            
            if not isRedundant:
                fsym.append(fsymAll[j])
        
        f_all   = f_all + fsym;    
        w_all   = w_all + len(fsym)*[w[n]]
        f_group = f_group + len(fsym)*[n]

    f = {'pattern': f_all, 'group': np.array(f_group,dtype = int)}
    w = np.array(w_all)
    return (f,w)
    
def countpattern(b,p,f):
    # b: board
    # p: player
    # f: features
    b_ = b*p
    c_ = np.array([0],dtype = int)
    c = np.zeros(len(f['pattern']),dtype = np.int)
    for j in range(0,len(f['pattern'])):
        countpat.countpat_func(c_, b_, f['pattern'][j])
        c[j] = c_[0]
    return c
   
def phi(s,a,f):
    s_= s[0]
    p = s[1]
    b[a[0], a[1]] = p
    c = countpattern(b,p,f);
    b[a[0], a[1]] = 0
    return c
    
def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)
 
#%%

def argmaxQ(s, f, w):
    b = s[0]
    p = s[1]
    
    # get all possible actions:
    c = conv2(abs(b),np.ones((3,3),dtype = np.int))
    a = [(a[0],a[1]) for a in np.argwhere((c > 0) & (b == 0))]
    
    # calculate phi(s,a) for all possible actions:
    nActions = len(a)
    phi_a = np.zeros((nActions,len(w)),dtype = np.int)
    
    for j in range(0,len(a)):
        phi_a[j] = phi(s,a[j],f)
    
    # calculate Q(s,a) for all possible actions: 
    Q = phi_a @ w;
    
    # find the best action (NB: for ties, first one will be chosen)
    k = np.argmax(Q)
    a_opt = a[k]
    Q_opt = Q[k]
    
    return (a_opt, Q_opt, a, phi_a)
    
#%%
    
def  = qAgent(s,w,f,epsilon)
#% assumes there is at least one legal move

#% s -> b
    s_ = s[0]
p = s[1]
    n = sqrt(len(s_);
    #b = reshape(s(1:end - 1),n,n);
    
    isRandom = 0;
    if all(b == 0)
     #   % start in the middle:
        n = b.shape[0];
        a = (n/2, n/2)
        phiAll = [1 a zeros(1,numel(w))];
    else
        [a,~,phia,as]  = argmaxQ(s,f,w);
        if rand < epsilon
            isRandom = 1;
            
      #      % choose random action:
            ii = randi(size(as,1),1,1);
            a = as(ii,:);
        else
            
        end
        
        if size(as,1) > 100, as = as(1:100,:); end
        phiAll = [as'; phia];
        phiAll = [size(as,1) phiAll(:)'];
    
        return [a isRandom phiAll]
    #####

    
#%%

[f,w] = getFeatures()

#%%

x = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0,-1, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int)

s = (x,1)

(a_opt, Q_opt, a, phi_a) = argmaxQ(s,f,w)


#%%

c = conv2(abs(x),np.ones((3,3),dtype = int))

a = np.argwhere(c > 0)

#%%

#countpat.countpat_func(c, x, f['pattern'][0])

c = countpattern(x,1,ff)

c_ = np.array([0],dtype = int)

countpat.countpat_func(c_, x.T, f['pattern'][36])
print(c_)

#%% Test phi
p = 1
b = x
s = (b, p)
a = (7, 7)
phi_ = phi(s, a, ff)


#%%
    

c_ = np.zeros(1,dtype = int);
c = np.zeros(len(f['pattern']),dtype = np.int)
for j in range(0,len(f['pattern'])):
    print(j)
    countpat.countpat_func(c_, x, f['pattern'][j])
    c[j] = c_[0]

    
a = np.array([[1, 2, 3], [4, 5, 6]], order='C')
np.isfortran(a)
False
>>> b = a.T
>>> b
array([[1, 4],
       [2, 5],
       [3, 6]])
>>> np.isfortran(b)
True


#%% syntax test

if not 7 == 7:
    print('e')
else:
    print('er')

b = np.array([[3, 4, 4],[1, 2, 3]])
#b = np.array([3, 4, 5])

for i, b_ in enumerate(b):
    print([i, np.where(b_ > 3)])







