#!/usr/bin/env python
''' 
Created by Manxi Lin
s192230@student.dtu.dk
'''

import numpy as np
from matplotlib import pyplot as plt
import cv2
import sys
import itertools
from mpl_toolkits.mplot3d import axes3d, Axes3D
import copy
import math

## 02.E
########################################################################################
def box3d(n):
    # 12+3=15 edges, -0.5~0.5, N=15n-16 points, 16 corner points, assume n is even
    # note that n>=2, if n<=1, then N is negative
    if n<=1:
        raise ValueError
        sys.exit(1)
    if n%2==1:
        print("Warning! n is odd")
    points = []
    # decide the vertex points of box
    xoyo = [-0.5, 0.5]
    xy = [x for x in itertools.product(xoyo, repeat=2)]
    ver = []
    for each in xy:
        ver.append(np.array([each[0],each[1],-0.5]))
    for each in xy:
        ver.append(np.array([each[0],each[1],0.5]))
    # draw line
    for p1 in ver:
        for p2 in ver:
                l = line3d(p1,p2,n)
                for each in l:
                    points.append(each)
    
    # draw a cross(6 vertex points)
    l = line3d(np.array([-0.5,0,0]),np.array([0.5,0,0]),n)
    for each in l:
        points.append(each)
    l = line3d(np.array([0,-0.5,0]),np.array([0,0.5,0]),n)
    for each in l:
        points.append(each)
    l = line3d(np.array([0,0,-0.5]),np.array([0,0,0.5]),n)
    for each in l:
        points.append(each)
    
    # eliminate the repeat points
    counter = 0
    p = copy.copy(points)
    idx = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if np.array_equal(np.round(points[i],5),np.round(points[j],5)):
                idx.append(j)
    points = []
    for i in range(len(p)):
        if idx.count(i)==0:
            points.append(p[i])
            #print(p[i])            
    return points




def line3d(p1, p2, n):
    d = (p2-p1)/(n-1) # divide the line into pieces
    line = []
    l = list(p2-p1)
    if list(p2-p1).count(0)==2:
        for i in range(n):
            line.append(p1+d*i)
    return line
###############################################################################################
## 02.F
def projectPoints(K, R, t, Q):
    # K,R,t,Q are respectively calibration matrix, rotation, translation, a list of 3D coordinates
    q_list = []
    for each in Q:
        M = np.hstack((R,t))
        H = M
        M = np.dot(K,M)
        each = np.transpose(np.array([each[0], each[1], each[2], 1]))
        q = np.dot(M, each)
        H = np.dot(H, each)
        if q[2]==0:
            print('Warning! q is not homogeneous!')
            print(q)
        else:
            q = q/q[2]
        q_list.append(q[:-1])
    return q_list
####################################################################################################
## 02.G
def _projectPoints(K, R, t, dist, Q):
    # K,R,t,Q are respectively calibration matrix, rotation, translation, a list of 3D coordinates
    # dist is the cofficients of radial distortion
    q_list = []
    Pb = np.array([[1, 0, K[0, 2]], [0, 1, K[1, 2]], [0, 0, 1]])
    for each in Q:
        M = np.hstack((R,t))
        K[0, 2] = 0
        K[1, 2] = 0
        M = np.dot(K,M)
        each = np.array([[each[0]], [each[1]], [each[2]], [1]])
        q = np.dot(M, each)
        r = np.sqrt((q[0]/q[2])**2+(q[1]/q[2])**2)
        theta = math.atan2(q[1]/q[2], q[0]/q[2])
        if dist.shape[0]<3:
            print("The function must work for at least coefficients!")
            raise ValueError
        coff = 1
        for i in range(dist.shape[0]):
            coff = coff+pow(r, 2*(i+1))*dist[i]
        r = coff*r
        q[0] = r*math.cos(theta)
        q[1] = r*math.sin(theta)
        q[2] = 1
        q = np.dot(Pb, q)
        q_list.append(q[:-1])
    return q_list

def rot(tx, ty, tz):
    rotz = np.array([[math.cos(tz),-math.sin(tz),0],[math.sin(tz),math.cos(tz),0],[0,0,1]])
    roty = np.array([[math.cos(ty),0,math.sin(ty)],[0,1,0],[-math.sin(ty),0,math.cos(ty)]])
    rotx = np.array([[1,0,0],[0,math.cos(tx),-math.sin(tx)],[0,math.sin(tx),math.cos(tx)]])
    return np.dot(np.dot(rotz, roty), rotx)

if __name__=="__main__":  
    #n = 20
    #points = box3d(n)
    #fig = plt.figure()
    #ax = Axes3D(fig)
    #for each in points:
    #    ax.scatter(each[0], each[1], each[2], color='blue')
    #plt.show()
    #print(len(points))
    #print(points)
    #K = np.array([[800, 0, 100],[0, 800, 100],[0, 0, 1]])
    #R = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
    #t = np.array([[0],[-1],[-5]])
    #Q = [np.array([1, 0, 10]), np.array([0, 1, 10]), np.array([0, 0, 10]), np.array([1, 2, 15])]
    #q = projectPoints(K, R, t, Q)
    #dist = np.array([-1e-6, 0, 0])
    #q = _projectPoints(K, R, t, dist, Q)
    #print(q)
    #q = projectPoints(K, R, t, Q)
    #dist = np.array([-1e-5, 0, 0])
    #q = _projectPoints(K, R, t, dist, Q)
    R = rot(1, 1, -0.5)
    t = np.array([[0],[0],[-7]])
    dist1 = np.array([-1e-6, 1e-12, 0])
    dist2 = np.array([-1e-6, 0, 0])
    K = np.array([[4000, 0, 960],[0, 4000, 540],[0, 0, 1]])
    n = 20
    points = box3d(n)
    q1 = _projectPoints(K, R, t, dist1, points)
    #print(q1)
    R = rot(1, 1, -0.5)
    t = np.array([[0],[0],[-7]])
    dist1 = np.array([-1e-6, 1e-12, 0])
    dist2 = np.array([-1e-6, 0, 0])
    K = np.array([[4000, 0, 960],[0, 4000, 540],[0, 0, 1]])
    points = box3d(n)
    q2 = _projectPoints(K, R, t, dist2, points)  
    #print(q2)
    for each in q1:
        plt.plot(each[0], each[1], '.b')
    for each in q2:
        plt.plot(each[0], each[1], '.r')
    plt.show()  
    error = 0
    for i in range(len(q1)):
        e = q1[i]-q2[i]
        error = error+e[0]**2+e[1]**2
    
    ave_error = error/len(q1)
    print(error)
    print(ave_error)

