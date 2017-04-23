# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import countpat
import random

#%%

def getFeatures():

    f_gen = [[0,1,1,0],
    [0,1,1,1,0],
    [0,1,1,1,1,0],
    [0,1,0,1,1,0],
    [1,1,1,1,1],
    [0,-1,-1,0],
    [0,-1,-1,-1,0],
    [0,-1,-1,-1,-1,0],
    [0,-1,-1,0,-1,0],
    [-1,-1,0,-1,-1],
    [-1,-1,-1,0,-1],
    [1,-1,-1,-1,-1,0]]    
    f_gen = [np.array([x],dtype = np.int) for x in f_gen]
    
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
    w = np.zeros((max(f['group'])+1,))
    for k in range(0,len(f['pattern'])):
        w[f['group'][k]] = w_all[k]
    return (f,w)
    
def countpattern(b,p,f):
    # b: board
    # p: player
    # f: features
    b_ = b*p
    c_ = np.array([0],dtype = int)
    c = np.zeros(np.max(f['group'])+1,dtype = np.int)
    for j in range(0,len(f['pattern'])):
        countpat.countpat_func(c_, b_, f['pattern'][j])
        c[f['group'][j]] += c_[0]
    return c
   
def phi(s,a,f):
    b = s[0]
    p = s[1]
    b[a[0], a[1]] = p
    c = countpattern(b,p,f);
    b[a[0], a[1]] = 0
    return c

def argmaxQ(s,f,w):
    b = s[0]

    # get all possible actions:
    def conv2(x,y,mode='same'):
        return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)
    c = conv2(abs(b),np.ones((3,3),dtype = np.int))
    a = [(a[0],a[1]) for a in np.argwhere((c > 0) & (b == 0))]
    
    # calculate phi(s,a) for all possible actions:
    nActions = len(a)
    phi_a = np.zeros((nActions,len(w)),dtype = np.int)
    
    for j in range(0,len(a)):
        phi_a[j] = phi(s,a[j],f)
    
    # calculate Q(s,a) for all possible actions: 
    Q = phi_a @ w;
    
    # find the best action (ties resolved randomly)
    m = np.max(Q)
    ii = np.where(Q > m - 1e-6)[0]
    if len(ii) > 1:
        k = ii[random.sample(range(0,len(ii)),1)[0]]
    else:
        k = ii[0]
    a_opt = a[k]
    Q_opt = Q[k]
    return (a_opt, Q_opt, a, phi_a)   
    
def choose_action(s,w,f,epsilon):
    # output:
    # a: the chosen action, 
    # a_: list of candidate actions, 
    # phi_a_: phi-value for each candidate)
    b = s[0]
    if not b.any():
        # empty board, play in the middle:
        n = b.shape[0];
        jMid = (int)((n+1)/2-1)
        a = (jMid, jMid)
        a_ = [a]
        phi_a_ = np.zeros((1,len(w)),dtype = np.int)
    else:
        a,_,a_,phi_a_  = argmaxQ(s,f,w);
        if random.random() < epsilon:
            # choose random action:
            a = a_[random.randint(0,len(a_)-1)]
    return (a,a_,phi_a_)

def check_win(b,a):
    WIN_CONDITION = 5;
    def check_win_1d(x):
        score = np.convolve(x, np.ones(WIN_CONDITION, dtype=np.int))
        win = max(abs(score)) == WIN_CONDITION
        return win
    (i,j) = a
    win = check_win_1d(b[i, :])
    if not win: win = check_win_1d(b[:,j])
    if not win: win = check_win_1d(b.diagonal(j - i))
    if not win: win = check_win_1d(np.fliplr(b).diagonal((b.shape[1] - j - 1) - i))
    return win

def roll_out(w,f,epsilon):
    n = 13
    b = np.zeros((n,n),dtype = int)
    is_over = 0
    p = 1

    game_record = list()
    while not is_over:
        if np.all(b):
            # draw
            break
        
        # state
        s = (np.array(b),p)
        
        # choose action
        a,a_,phi_a_ = choose_action(s,w,f,epsilon)
        
        # update board        
        b[a[0],a[1]] = p
        
        # check win:
        is_win = check_win(b,a)
        if is_win:
            is_over = 1
            r = 1
        else:
            r = 0

        # new entry in game record: (s,a,r,a_,phi_a_)
        rec = (s,a,r,a_,np.array(phi_a_))
        game_record.append(rec)
        
        # next player's turn
        p = -p;
    return game_record

class experience:
    n = 0;        # where to write next item
    m = 0;        # total number of item currently in memory
    nMax = 10000;
    data = list()
    
    def add(self,x):
        if self.n < self.nMax:
            #self.data[self.n] = x
            self.data.append(x)
            self.n = self.n + 1
        else:
            self.data[0] = x
            self.n = 1
        self.m = max(self.n,self.m)
    
    def store_game(self,game_record):
        # all but last state transition for first player:
        for j in range(0,len(game_record)-2,2):
            self.add((game_record[j][0],    # s
                      game_record[j][1],    # a
                      game_record[j][2],    # r
                      game_record[j+2][0],  # s_next
                      game_record[j+2][3],  # a_next (canditate moves at s_next)
                      game_record[j+2][4])) # phi_next (phi-values at s_next)
        # all but last state transition for second player:
        for j in range(1,len(game_record)-2,2):
            self.add((game_record[j][0],    # s
                      game_record[j][1],    # a
                      game_record[j][2],    # r
                      game_record[j+2][0],  # s_next
                      game_record[j+2][3],  # a_next (canditate moves at s_next)
                      game_record[j+2][4])) # phi_next (phi-values at s_next)
        # last state transition for loser:
        self.add(( game_record[-2][0],  # s
                   game_record[-2][1],  # a
                  -game_record[-1][2],  # r (-1 times winner's reward)
                   None,None,None))     # "None" here because final state
        # last state transition for winner:
        self.add((game_record[-1][0],  # s
                  game_record[-1][1],  # a
                  game_record[-1][2],  # r
                  None,None,None))     # "None" here because final state

class adamClass:
    a   = 0.001
    b1  = 0.9
    b2  = 0.999
    eps = 1e-8
    n   = 0
    def __init__(self,w):
        self.m = np.zeros_like(w)
        self.v = np.zeros_like(w)
    def update(self,w):
        adam.n = adam.n + 1
        adam.m = adam.b1*adam.m + (1-adam.b1)*dLdw
        adam.v = adam.b2*adam.v + (1 - adam.b2)*np.power(dLdw,2)
        mHat = adam.m / (1 - adam.b1**adam.n)
        vHat = adam.v / (1 - adam.b2**adam.n)
        return (w + adam.a * mHat / (np.sqrt(vHat) + adam.eps))
                  
#%%

mini_batch_size = 50
epsilon = 0.1;

# features, initial weights:
f,w = getFeatures()
# w = np.array([ 0.15881415,  0.20494743,  0.60102403,  0.09290497,  0.96104338,       -0.01478161, -0.62512229, -1.13322032, -0.54985369, -0.89275337,       -1.13227525, -1.18833344,  0.17363202,  0.22542401,  0.54344071,        0.06310447,  0.94988106, -0.01218425, -0.49280391, -1.19674693,       -0.40013123, -0.8735151 , -1.67728488, -1.13100639])

# adam:
adam = adamClass(w)

# initialise experience:
D = experience()
while D.m < mini_batch_size:
    game_record = roll_out(w,f,epsilon)
    D.store_game(game_record)

wAll  = list()
LAll  = list()
    
#%% 

# start learning:
for kk in range(0,3000):    
    # play a game:
    game_record = roll_out(w,f,epsilon)
    
    # add it to experience:
    D.store_game(game_record)
    
    # stochastic gradient descent
    for _ in range(0,5):
        # mini-batch:
        L    = 0                # loss
        dLdw = np.zeros_like(w) # gradient
        for k in random.sample(range(0,D.m),mini_batch_size):
            (s,a,r,s_,a_,phi_a_) = D.data[k]
            if s_ is None:
                # final state:
                y = r
            else:
                Q = phi_a_ @ w
                y = np.max(Q)
            phi_a = phi(s,a,f)
            x = (y - np.dot(w,phi_a))
            L = L + x**2
            dLdw = dLdw + x*phi_a
        L = -L / mini_batch_size
        dLdw = dLdw / mini_batch_size
            
        # adam step:
        w = adam.update(w)
        
        wAll.append(np.array(w))
        LAll.append(L)
    if kk%10 == 0:
        print(kk)
    
#%%

w_ = np.empty((len(wAll),len(w)))
L_ = np.empty((len(LAll),))

for j in range(0,len(wAll)):
    w_[j][:] = wAll[j]
    L_[j] = LAll[j]

plt.figure(1)
plt.clf()
plt.subplot(121)
plt.plot(w_)
plt.subplot(122)
plt.plot(L_)

