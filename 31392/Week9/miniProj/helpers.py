import numpy as np
import cv2 as cv2
from numpy.linalg import inv, pinv
import matplotlib.pyplot as plt
import time as t

def getK():
    return np.array([[7.188560e+02, 0.000000e+00, 6.071928e+02],
                     [0, 7.188560e+02, 1.852157e+02],
                     [0, 0, 1]])

def getTruePose():
    file = '00.txt'
    return np.genfromtxt(file, delimiter=' ', dtype=None)

def getLeftImage(i):
    return cv2.imread('left/{0:010d}.png'.format(i), 0)

def getRightImage(i):
    return cv2.imread('right/{0:010d}.png'.format(i), 0)

def removeDuplicate(queryPoints, refPoints, radius=5):
    # remove duplicate points from new query points,
    for i in range(len(queryPoints)):
        query = queryPoints[i]
        xliml, xlimh = query[0] - radius, query[0] + radius
        yliml, ylimh = query[1] - radius, query[1] + radius
        inside_x_lim_mask = (refPoints[:, 0] > xliml) & (refPoints[:, 0] < xlimh)
        curr_kps_in_x_lim = refPoints[inside_x_lim_mask]

        if curr_kps_in_x_lim.shape[0] != 0:
            inside_y_lim_mask = (curr_kps_in_x_lim[:, 1] > yliml) & (curr_kps_in_x_lim[:, 1] < ylimh)
            curr_kps_in_x_lim_and_y_lim = curr_kps_in_x_lim[inside_y_lim_mask, :]
            if curr_kps_in_x_lim_and_y_lim.shape[0] != 0:
                queryPoints[i] = np.array([0, 0])
    return (queryPoints[:, 0] != 0)
