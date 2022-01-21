'''
=============================================================
Created and Commented by :
Farid Muhammad Eryanto - 1103181221

These Codes are reproduced from following github repository :
https://github.com/ansh941/MnistSimpleCNN

These repository was created as final exam task for 
Telkom University's Machine Learning G5 Subject
=============================================================
'''


import numpy as np 

cnt = 1
best = 10000
curr = 10000
for i in range(10):
    for j in range(10):
        for k in range(10):
            w1 = np.loadtxt("../logs/modela1/wrong%03d.txt"%(301+i)).astype(np.int)
            w2 = np.loadtxt("../logs/modela3/wrong%03d.txt"%(301+j)).astype(np.int)
            w3 = np.loadtxt("../logs/modela4/wrong%03d.txt"%(301+k)).astype(np.int)

            board = np.zeros((10000))
            board[w1] += 1
            board[w2] += 1
            board[w3] += 1
            board = board // 2
            curr = np.sum(board)
            if curr < best:
                best = curr
            print("%4d %4d %4d"%(cnt, curr, best))
            cnt += 1
