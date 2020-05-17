import numpy as np
import open3d as o3d
import copy
import random
import math
from matplotlib import pyplot as plt

def draw_labels_on_model(pcl, labels):
    cmap = plt.get_cmap("tab20")
    pcl_temp = copy.deepcopy(pcl)
    max_label = np.int(labels.max())
    colors = cmap(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    colors = colors[...,:3].reshape(-1,3)
    pcl_temp.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcl_temp])

d = 4
mesh = o3d.geometry.TriangleMesh.create_tetrahedron().translate((-d, 0, 0))
mesh += o3d.geometry.TriangleMesh.create_octahedron().translate((0, 0, 0))
mesh += o3d.geometry.TriangleMesh.create_icosahedron().translate((d, 0, 0))
mesh += o3d.geometry.TriangleMesh.create_torus().translate((-d, -d, 0))
mesh += o3d.geometry.TriangleMesh.create_moebius(twists=1).translate((0, -d, 0))
mesh += o3d.geometry.TriangleMesh.create_moebius(twists=2).translate((d, -d, 0))

## apply k means on this
point_cloud = mesh.sample_points_uniformly(int(1e3))
#o3d.visualization.draw_geometries([point_cloud])
def getDist(p1, p2):
    p1 = p1.reshape(-1,1)
    p2 = p2.reshape(-1,1)
    p = p1-p2
    dist = np.dot(p.reshape(1,-1),p)
    return math.sqrt(dist[:])

def clustering(center, points):
    pointNumber = points.shape[0]
    clsNumber = center.shape[0]
    label = np.zeros((pointNumber, 1))
    for i in range(pointNumber):
        # clustering each point
        dist_0 = getDist(center[0], points[i])
        label[i] = 0
        for j in range(1, clsNumber):
            if getDist(center[j], points[i])<dist_0:
                label[i] = j
                dist_0 = getDist(center[j], points[i])
    return label

def getNewCenter(points, label):
    clsNumber = np.int(label.max())+1
    center = np.zeros((clsNumber, 3))
    for i in range(clsNumber):
        idx = np.hstack((label==i, label==i, label==i))
        clsPoint = points.ravel()[idx.ravel()]
        clsPoint = clsPoint.reshape(-1,3)
        center[i, 0] = np.mean(clsPoint[:,0]);center[i, 1] = np.mean(clsPoint[:,1]);center[i, 2] = np.mean(clsPoint[:,2])
    return center

def centerToCenter(center, new_center):
    # here we take the mean distance from the old centers to the new centers
    clsNumber = center.shape[0]
    dist = 0
    for i in range(clsNumber):
        dist = dist+getDist(center[i], new_center[i])
    dsit = dist/clsNumber
    return dist

def getDistortion(center, points, label):
    dist = 0
    for i in range(points.shape[0]):
        dist = dist + getDist(points[i], center[np.int(label[i])])
    return dist


def kMeansClassify(point_cloud, k, auto=False, max_interation=10000, crit=1e0, max_k=10):
    # input is a point cloud
    data = np.asarray(point_cloud.points) # shape = (-1,3)
    ptsNumber = data.shape[0]
    if not auto:
        # initialization
        # select k centers randomly
        idx = [random.randint(0, ptsNumber-1) for i in range(k)]
        center = data[idx, ...]
        for counter in range(max_interation): 
            # clustering
            label = clustering(center, data)
            # calculate the new centers
            new_center = getNewCenter(data, label)
            # measure the distance between the two centers  
            dist = centerToCenter(center, new_center)     
            if dist<=crit:
                # meet the criterion
                break
            else:
                center = new_center
    else:
        # apply the elbow method to get the optimal k
        distortion = np.zeros((max_k-2, 1))
        for k in range(1, max_k):
            label,center = kMeansClassify(point_cloud, k) 
            distortion[k-2] = getDistortion(center, data, label)
        for i in range(distortion.shape[0]-2):
            if (distortion[i]-distortion[i+1])/(distortion[i+2]-distortion[i+1])<1.5:
                k = i+1
        print('The optimal k is '+str(k))
        label,center = kMeansClassify(point_cloud, k)            
    return label, center

label, center = kMeansClassify(point_cloud, 6, True)
draw_labels_on_model(point_cloud, label)
