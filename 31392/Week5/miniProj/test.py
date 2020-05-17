## Task1
import cv2
import open3d as o3d
from matplotlib import pyplot as plt
import numpy as np
import copy
import scipy
from scipy import spatial
import random
import sys
import math

def compute_transformation(source, target):
    # Kabsch Algorithm
    # Normalization
    number = len(source)
    cs = np.zeros((3,1)) # the centroid of source points
    ct = copy.deepcopy(cs) # the centroid of the target points
    cs[0]=np.mean(source[:][0]);cs[1]=np.mean(source[:][1]);cs[2]=np.mean(source[:][2])
    ct[0]=np.mean(target[:][0]);ct[1]=np.mean(target[:][1]);ct[2]=np.mean(target[:][2])
    cov = np.zeros((3,3)) # covariance matrix
    for i in range(number):
        sources = source[i].reshape(-1,1)-cs
        targets = target[i].reshape(-1,1)-ct
        cov = cov + np.dot(sources,np.transpose(targets))
    # SVD
    u,w,v = np.linalg.svd(cov)
    # rotation matrix
    R = np.dot(u, np.transpose(v))
    # Translation Vector
    T = ct - np.dot(R, cs)
    return R, T

def _transform(source, R, T):
    points = []
    for point in source:
        points.append(np.dot(R, point.reshape(-1,1))+T)
    return points

def compute_rmse(source, target, R, T):
    rmse = 0
    number = len(target)
    points = _transform(source, R, T)
    for i in range(number):
        error = target[i].reshape(-1,1)-points[i]
        rmse = rmse + math.sqrt(error[0]**2+error[1]**2+error[2]**2)
    return rmse


def draw_registrations(source, target, transformation = None, recolor = False):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if(recolor): # recolor the points
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
    if(transformation is not None): # transforma source to targets
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

#draw_registrations(source, target)
def pc2array(pointcloud):
    return np.asarray(pointcloud.points)
def registration_RANSAC(source, target, source_feature, target_feature, ransac_n = 3, max_interation=100000, max_validation=100):
    # the intention of RANSAC is to get the optimal transformation between the source and target
    s = pc2array(source) #(4760, 3)
    t = pc2array(target)
    sf = np.transpose(source_feature.data) # (33,4760), source features
    ssf = copy.deepcopy(sf)
    tf = np.transpose(target_feature.data) # target features
    tree = spatial.KDTree(ssf) # Create a KD tree 
    corres_stock = tree.query(tf)[1]
    for i in range(max_interation):
        # take ransac_n points randomly
        idx = [random.randint(0, t.shape[0]-1) for j in range(ransac_n)]
        corres_idx = corres_stock[idx]
        source_point = s[corres_idx,...]
        target_point = t[idx, ...]
        # estimate transformation
        # use Kabsch Algorithm
        R, T = compute_transformation(source_point, target_point)
        # calculate rmse for all the points
        source_point = s[corres_stock,...]
        target_point = t
        rmse = compute_rmse(source_point, target_point, R, T)
        if not i:
            opt_rmse = rmse
            opt_R = R
            opt_T = T
        else:
            if rmse<opt_rmse:
                opt_rmse = rmse
                opt_R = R
                opt_T = T
    return opt_R, opt_T
# Used for downsampling.
voxel_size = 0.05
def get_fpfh(cp):
    cp = cp.voxel_down_sample(voxel_size)
    cp.estimate_normals()
    return cp, o3d.registration.compute_fpfh_feature(cp, o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=100))



source = o3d.io.read_point_cloud('global_registration/car1.ply')
target = o3d.io.read_point_cloud('global_registration/car2.ply')                   
r1, f1 = get_fpfh(source)
r2, f2 = get_fpfh(target)
#point2point =  o3d.registration.TransformationEstimationPointToPoint(False)
#ransac_result = o3d.registration.registration_ransac_based_on_feature_matching(
#    r1, r2, 
#    f1, f2, 
#    voxel_size*1.5, 
#point2point)
#draw_registrations(r1, r2, ransac_result.transformation, True)
R, T = registration_RANSAC(r1,r2,f1,f2, max_interation=1000)
transformation = np.vstack((np.hstack((np.float64(R), np.float64(T))), np.array([0,0,0,1])))
draw_registrations(r1, r2, transformation, True)
