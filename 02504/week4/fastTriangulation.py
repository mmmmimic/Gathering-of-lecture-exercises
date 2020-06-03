import numpy as np
import math
import copy
import sys

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
    s = 1/math.sqrt(np.mean(d)/2)
    T = np.array([[s,0,-mp[0]*s],[0,s,-mp[1]*s],[0,0,1]])
    p1 = np.dot(T, p)
    mp = np.array([[np.mean(p1[0,:])],[np.mean(p1[1,:])],[np.mean(p1[2,:])]])
    #print(mp)
    dp = p1 - mp
    d = np.dot(dp[0,:].T,dp[0,:])+np.dot(dp[1,:].T,dp[1,:])
    #print(np.mean(d))
    return T

def fastTriangulation(q1, q2, K1, K2, R, t):
    M2 = np.hstack((R, t))
    M1 = np.hstack((R, np.array([[0],[0],[0]])))
    P1 = np.dot(K1, M1)
    P2 = np.dot(K2, M2)
    P11 = P1[0, :]
    P12 = P1[1, :]
    P13 = P1[2, :]
    P21 = P2[0, :]
    P22 = P2[1, :]
    P23 = P2[2, :]
    x1 = q1[0]
    y1 = q1[1]
    x2 = q2[0]
    y2 = q2[1]
    B = np.zeros((4,4))
    B[0,:] = (P13*x1-P11)
    B[1,:] = (P13*y1-P12)
    B[2,:] = (P23*x2-P21)
    B[3,:] = (P23*y2-P22)
    u,w,v = np.linalg.svd(B)
    Q = v[-1,:]
    Q = Q/Q[3]
    #print(np.dot(P1, v))
    #print(np.dot(P2, v))
    return Q

def estimateFundamentalPre(q1, q2):
    x1 = q1[0]
    y1 = q1[1]
    x2 = q2[0]
    y2 = q2[1]
    B = np.array([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])
    return B

def estimateFundamental(q1, q2):
    l = len(q1)
    B = np.zeros((l , 9))
    for i in range(l):
        B[i,:] = estimateFundamentalPre(q1[i], q2[i])
    u,w,v = np.linalg.svd(B)
    v = v[-1,:]
    F = np.array([[v[0], v[3], v[6]], [v[1], v[4], v[7]],[v[2], v[5], v[8]]])
    return F

def estimateHomographyPre(q1, q2):
    x1 = q1[0]
    y1 = q1[1]
    x2 = q2[0]
    y2 = q2[1]
    B = np.array([[0, -x2, x2*y1, 0, -y2, y2*y1, 0, -1, y1],
    [x2, 0, -x2*x1, y2, 0, -y2*x1, 1, 0, -x1],
    [-x2*y1, x2*x1, 0, -y2*y1, y2*x1, 0, -y1, x1, 0]])
    return B

def estimateHomography(q1, q2):
    l = len(q1)
    B = np.zeros((l*3 , 9))
    for i in range(l):
        B[i:i+3,:] = estimateHomographyPre(q1[i], q2[i])
    u,w,v = np.linalg.svd(B)
    v = v[-1,:]
    H = np.array([[v[0], v[3], v[6]], [v[1], v[4], v[7]],[v[2], v[5], v[8]]])
    return H



if __name__=="__main__":
    '''
    point = np.array([[10],[4],[10],[1]])
    K1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    K2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    R = np.array([[1,0,0],[0,1,0],[0,0,1]])
    t = np.array([[-5],[0],[2]])
    M2 = np.hstack((R, t))
    M1 = np.hstack((R, np.array([[0],[0],[0]])))
    P1 = np.dot(K1, M1)
    P2 = np.dot(K2, M2)
    q1 = np.dot(P1, point)
    q2 = np.dot(P2, point)
    q1 = q1/q1[2]
    q2 = q2/q2[2]
    Q = fastTriangulation(q1, q2, K1, K2, R, t)
    print(Q)
    '''
    '''
    K1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    K2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    R = np.array([[1,0,0],[0,1,0],[0,0,1]])
    t = np.array([[-5],[0],[2]])
    M2 = np.hstack((R, t))
    M1 = np.hstack((R, np.array([[0],[0],[0]])))
    P1 = np.dot(K1, M1)
    P2 = np.dot(K2, M2)
    q1 = []
    q2 = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                point = np.array([[i],[j],[10+10*k],[1]])
                qq1 = np.dot(P1, point)
                q1.append(qq1/qq1[2])
                qq2 = np.dot(P2, point)
                q2.append(qq2/qq2[2])
    T1 = normalize2d(q1)
    T2 = normalize2d(q2)
    q1 = [np.dot(T1, p) for p in q1]
    q2 = [np.dot(T2, p) for p in q2]
    F = estimateFundamental(q1, q2)
    F = np.dot(F, T1)
    F = np.dot(np.transpose(T2), F)
    print(F)
    '''
    K1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    K2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    R = np.array([[1,0,0],[0,1,0],[0,0,1]])
    t = np.array([[-5],[0],[2]])
    M2 = np.hstack((R, t))
    M1 = np.hstack((R, np.array([[0],[0],[0]])))
    P1 = np.dot(K1, M1)
    P2 = np.dot(K2, M2)
    q1 = []
    q2 = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                point = np.array([[i],[j],[10+10*k],[1]])
                qq1 = np.dot(P1, point)
                q1.append(qq1/qq1[2])
                qq2 = np.dot(P2, point)
                q2.append(qq2/qq2[2])
    T1 = normalize2d(q1)
    T2 = normalize2d(q2)
    q1 = [np.dot(T1, p) for p in q1]
    q2 = [np.dot(T2, p) for p in q2]
    H = estimateHomography(q1, q2)
    H = np.dot(H, T1)
    H = np.dot(np.transpose(T2), H)
    print(H)