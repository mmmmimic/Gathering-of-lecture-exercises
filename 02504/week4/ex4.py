from triangulation import triangulate
import numpy as np
from matplotlib import pyplot as plt
import pinholeCamera
##### B #####
K = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
R = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
t = np.array([[-5],[0],[2]])
q1 = np.array([[0.2],[0.4]])
q2 = np.array([[-3/12],[4/12]])
Q = triangulate(q1, q2, K, K, R, t)
print(Q)
q1 = np.array([[2/20],[4/20]])
q2 = np.array([[-3/22],[4/22]])
Q = triangulate(q1, q2, K, K, R, t)
print(Q)
##### C #####
q1 = np.array([[-1/6],[1/3]])
q2 = np.array([[-1/2],[2/7]])
Q = triangulate(q1, q2, K, K, R, t)
print(Q)
##### D #####
q1 = q1+0.1
q2 = q2+0.1
print(q1)
print(q2)
Q = triangulate(q1, q2, K, K, R, t)
print(Q)
q2 = pinholeCamera.projectPoints(K, R, t, [Q])
t = np.array([[0],[0],[0]])
q1 = pinholeCamera.projectPoints(K, R, t, [Q])
print(q1)
print(q2)
#####