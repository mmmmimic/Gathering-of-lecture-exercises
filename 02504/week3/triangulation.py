#!/usr/bin/env python
''' 
Created by Manxi Lin
s192230@student.dtu.dk
'''
import numpy as np
import sys
import copy
import cv2
import pinholeCamera
import math
from mpl_toolkits.mplot3d import axes3d, Axes3D
import itertools
from matplotlib import pyplot as plt

## reconstruct a 3D point from the pictures in two cameras
def triangulate(q1, q2, K1, K2, R, t):
    ## q1 and q2 should be 2D
    q1 = [[q1[0]], [q1[1]], [1]]
    q2 = [[q2[0]], [q2[1]], [1]]
    a = np.linalg.inv(K1)
    v1 = np.dot(np.linalg.inv(K1), q1)
    v2 = np.dot(np.linalg.inv(K2), q2)
    H = np.hstack((R,t))
    H = np.vstack((H,np.array([0,0,0,1])))
    H = np.linalg.inv(H)
    O1 = np.array([[0],[0],[0],[1]])
    O2 = np.dot(H, O1)
    O1 = O1[:3]
    O2 = O2[:3]
    u = np.cross(np.transpose(v1), np.transpose(v2))
    u = np.transpose(u)
    n1 = np.cross(np.transpose(u), np.transpose(v1))
    n2 = np.cross(np.transpose(u), np.transpose(v2))
    T1 = np.hstack((v1, v2, O2-O1))
    T1 = np.dot(n1/2, T1)
    T2 = np.hstack((v1, v2, O1-O2))
    T2 = np.dot(n2/2, T2)
    T1 = np.transpose(T1)
    T2 = np.transpose(T2)
    n = np.vstack((n1, n2))
    ans = np.dot(np.linalg.inv(np.dot(n, np.hstack((v1, v2)))),np.vstack((np.dot(n1, (O1-O2)), np.dot(n2, (O2-O1)))))
    s1 = ans[0]
    s2 = ans[1]
    A = O1+s1*v1
    B = O2+s2*v2
    Q = (A+B)/2
    return Q
    
def skewSys(p):
    return [[0, -p[2], p[1]], [p[0], 0, -p[0]], [-p[1], p[2], 0]]

def det(p):
    d = 0
    for each in p:
        d = d+each**2
    return np.sqrt(d)

if __name__=="__main__":
    n = 4
    points = pinholeCamera.box3d(n)
    p = [points[1]]
    print(p)
    K = np.array([[1000, 0, 300],[0, 1000, 200],[0, 0, 1]])
    R = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
    t = np.array([[0],[0],[0]])
    q1 = pinholeCamera.projectPoints(K, R, t, p)[0]
    #print(q1)
    #R = np.array([[1/np.sqrt(2), -1/np.sqrt(2), 0],[1/np.sqrt(2), 1/np.sqrt(2), 0],[0, 0, 1]])
    R = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
    t = np.array([[0],[-1],[-5]])
    q2 = pinholeCamera.projectPoints(K, R, t, p)[0]
    #print(q2)
    Q = triangulate(q1, q2, K, K, R, t)
    print(Q)
    sys.exit(0)
    #fig = plt.figure()
    #ax = Axes3D(fig)
    #for each in points:
    #    ax.scatter(each[0], each[1], each[2], color='blue')
    #plt.show()
    p = []
    p.append(np.array([1, 1, 1]))
    print(p)
    K = np.array([[1000, 0, 300],[0, 1000, 200],[0, 0, 1]])
    R = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
    t = np.array([[0],[0],[0]])
    q1 = pinholeCamera.projectPoints(K, R, t, p)[0]
    #print(q1)
    R = np.array([[1/np.sqrt(2), -1/np.sqrt(2), 0],[1/np.sqrt(2), 1/np.sqrt(2), 0],[0, 0, 1]])
    t = np.array([[0],[-1],[-5]])
    p = []
    p.append(np.array([1, 1, 1]))
    q2 = pinholeCamera.projectPoints(K, R, t, p)[0]
    #print(q2)
    print(q1[:2])
    print(q2[:2])
    Q = triangulate(q1, q2, K, K, R, t)
    print(Q)
    q2 = pinholeCamera.projectPoints(K, R, t, [Q])[0]
    K = np.array([[1000, 0, 300],[0, 1000, 200],[0, 0, 1]])
    R = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
    t = np.array([[0],[0],[0]])
    q1 = pinholeCamera.projectPoints(K, R, t, [Q])[0]
    q2 = q2/q2[2]
    q1 = q1/q1[2]
    print(q1[:2])
    print(q2[:2])



