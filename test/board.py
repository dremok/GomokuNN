# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 18:50:39 2017

@author: Siven
"""

import numpy as np

n = 13;

#%% board logic

def checkWin(x):
    score = np.convolve(x,np.ones(5,dtype = np.int8))
    win = max(abs(score)) == 5
    return win

def updateBoard(board,player,i,j):
   assert(board[i,j] == 0)
   # make move
   if player == 1:
       board[i,j] = 1
   else:
       board[i,j] = -1
   # check win
   win = checkWin(board[i,:]) 
   if not win: win = checkWin(board[:,j])
   if not win: win = checkWin(board.diagonal(j-i))
   if not win: win = checkWin(np.fliplr(board).diagonal((n-j-1)-i))
   return [board,win]


#%% test it

board = np.zeros((n,n), dtype=np.int8)   
board[:] = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 1,-1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0,-1,-1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

# no win:
[board,win]= updateBoard(board,2,5,3)
print(win)
# row win:
[board,win]= updateBoard(board,1,5,7)
print(win)


board = np.zeros((n,n), dtype=np.int8)   
board[:] = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1,-1, 1,-1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0,-1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0,-1,-1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# counter diagonal win:
[board,win]= updateBoard(board,2,2,10)
print(win)

board = np.zeros((n,n), dtype=np.int8)   
board[:] = [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# diagonal win:
[board,win]= updateBoard(board,1,4,10)
print(win)
            