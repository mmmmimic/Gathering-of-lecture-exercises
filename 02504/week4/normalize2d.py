import numpy as np
import sys
import math
import copy

def normalize2d(p):
    # return T, q = Tp, mean of q is 0 and variance of q is 1
    # data structure: p: list
    # q: list
    # T: ndarray
    p = np.array(p)
    p = p.reshape(-1, 3)
    p = np.transpose(p)
    mp = np.array([[np.mean(p[0,:])],[np.mean(p[1,:])],[np.mean(p[2,:])]])
    dp = p - mp
    d = np.dot(dp[0,:].T,dp[0,:])+np.dot(dp[1,:].T,dp[1,:])
    s = 1/math.sqrt(2*np.mean(d))
    T = np.array([[s,0,-mp[0]*s],[0,s,-mp[1]*s],[0,0,1]])
    p1 = np.dot(T, p)
    mp = np.array([[np.mean(p1[0,:])],[np.mean(p1[1,:])],[np.mean(p1[2,:])]])
    #print(mp)
    dp = p1 - mp
    d = np.dot(dp[0,:].T,dp[0,:])+np.dot(dp[1,:].T,dp[1,:])
    #print(np.mean(d))
    return T

if __name__=="__main__":
    p = []
    for i in range(3):
        for j in range(3):
                p.append(np.array([[i],[j],[1]]))
    T = normalize2d(p)
    print(T)