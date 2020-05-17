import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import copy
        
# Helper function to draw registrations (reccomended)
def draw_registrations(source, target, transformation = None, recolor = False):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        if(recolor):
            source_temp.paint_uniform_color([1, 0.706, 0])
            target_temp.paint_uniform_color([0, 0.651, 0.929])
        if(transformation is not None):
            source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])
# extension
def image2point(imgSrs):
    colorName = "RGBD/color/000"+str(imgSrs)+".jpg"
    depthName = "RGBD/depth/000"+str(imgSrs)+".png"
    # create rgbd images
    color = o3d.io.read_image(colorName)
    depth = o3d.io.read_image(depthName)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity = True)
    camera = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera)
    # flip the point clouds
    pc.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    return pc

def combine(pc1, pc2):
    # combine two point clouds into one and down sample
    pc = pc1+pc2
    return pc.voxel_down_sample(0.05)

def cvt3digit(number):
    mod_0 = number%10
    mod_100 = number//100
    mod_10 = (number-mod_100*100)//10
    return str(mod_100)+str(mod_10)+str(mod_0)

def registration(source, target):
    # input ought to be two cloudPoints
    # Parameters
    threshold = 0.2 #0.02
    trans_init = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0.1, 0.1, 0, 1]]) 
    #Evaluate registration
    evaluation = o3d.registration.evaluate_registration(source, target, threshold, trans_init)
    print("evaluation:"+str(evaluation))
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,
                            max_nn=30),fast_normal_computation=True)    
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,
                            max_nn=30),fast_normal_computation=True)
    point_to_plane =  o3d.registration.TransformationEstimationPointToPlane()                         
    return o3d.registration.registration_icp(source, target, threshold, trans_init, point_to_plane)

if __name__ =="__main__":
    target = image2point(cvt3digit(0))
    for i in range(1, 401):
        source = image2point(cvt3digit(i))
        regis_result = registration(source, target)
        new = source.transform(regis_result.transformation)+target
        target = new.voxel_down_sample(voxel_size=0.05)
    o3d.visualization.draw_geometries([target])
