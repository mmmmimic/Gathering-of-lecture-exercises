#Challenge
#The current implementation only uses features computed at the current timestep. 
# However, as we process more images we potentially have a lot of features from 
# previous timesteps that are still valid. The challenge is to expand the 
# extract_keypoints_surf(..., refPoints) function by giving it old reference 
# points. You should then combine your freshly computed features with the old 
# features and remove all duplicates. This requires you to keep track of old 
# features and 3D points.

#Hint 1: look in helpers.py for removing duplicates.

#Hint 2: you are not interested in points that are behind you, so remember to remove points that are negative in the direction you move.



import numpy as np
import cv2 as cv2
from numpy.linalg import inv, pinv
import matplotlib.pyplot as plt
import time as t
from helpers import *
import copy

def extract_keypoints_surf(img1, img2, K, baseline, old_p, old_l):
    surf = cv2.xfeatures2d_SURF.create()
    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)
    matcher = cv2.FlannBasedMatcher()
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x:x.distance)
    match_points1 = []
    match_points2 = []
    for m in matches:
        match_points1.append(kp1[m.queryIdx].pt)
        match_points2.append(kp2[m.trainIdx].pt)
    p1 = np.array(match_points1).astype(np.float32)
    p2 = np.array(match_points2).astype(np.float32)
    if len(old_p)<1:
        old_p=p1
    else:
        idx = removeDuplicate(copy.deepcopy(p1), old_p, radius=5)
        p1 = p1[idx]
        p2 = p2[idx]
        old_p = np.vstack((old_p, p1))

    ##### ############# ##########
    ##### Do Triangulation #######
    ##### ########################
    #project the feature points to 3D with triangulation
    
    #projection matrix for Left and Right Image
    M_left = K.dot(np.hstack((np.eye(3), np.zeros((3, 1)))))
    M_rght = K.dot(np.hstack((np.eye(3), np.array([[-baseline, 0, 0]]).T)))

    p1_flip = np.vstack((p1.T, np.ones((1, p1.shape[0]))))
    p2_flip = np.vstack((p2.T, np.ones((1, p2.shape[0]))))

    P = cv2.triangulatePoints(M_left, M_rght, p1_flip[:2], p2_flip[:2])

    # Normalize homogeneous coordinates (P->Nx4  [N,4] is the normalizer/scale)
    P = P / P[3]
    land_points = P[:3]
    land_points = land_points.T

    if len(old_l)<1:
        old_l=land_points
    else:
        old_l = np.vstack((old_l, land_points))
    return old_l ,old_p
    
def featureTracking(img_1, img_2, p1, world_points):
    """
    track the features of img_1 in img_2 via optical flow
    p1 and world_points are 2D and 3D points respectively
    return the tracked prev_points(p1), next_points(p2), world_points(3d points)
    """
    params = dict(winSize=(21, 21),maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p2, status,_ = cv2.calcOpticalFlowPyrLK(img_1, img_2, p1, None, **params)
    idx = np.array(status==1).ravel()
    p1 = p1[idx]
    p2 = p2[idx]
    world_points = world_points[idx]
    return world_points, p1, p2

def playImageSequence(left_img, right_img, K):

    baseline = 0.54

    ##### ################################# #######
    ##### Get 3D points Using Triangulation #######
    ##### #########################################
    """
    Implement step 1.2 and 1.3
    Store the features in 'reference_2D' and the 3D points (landmarks) in 'landmark_3D'
    hint: use 'extract_keypoints_surf' above
    """
    landmark_3D = []
    reference_2D = []
    prevPose = []
    landmark_3D,reference_2D = extract_keypoints_surf(left_img, right_img, K, baseline, reference_2D, landmark_3D)
    # reference
    reference_img = left_img

    # Groundtruth for plot
    truePose = getTruePose()
    traj = np.zeros((600, 600, 3), dtype=np.uint8)
    maxError = 0

    for i in range(0, 101):
        print('image: ', i)
        curImage = getLeftImage(i)
        curImage_R = getRightImage(i)

        ##### ############################################################# #######
        ##### Calculate 2D and 3D feature correspndances in t=T-1, and t=T  #######
        ##### #####################################################################
        """
        Implement step 2.2)
        Remember this is a part of a loop, so the initial features are already
        provided in step 1)-1.3) outside the loop in 'reference_2D' and 'landmark_3D'
        """
        landmark_3D, reference_2D, current_2D = featureTracking(reference_img, curImage, 
        reference_2D, landmark_3D)
        ##### ################################# #######
        ##### Calculate relative pose using PNP #######
        ##### #########################################
        """
        Implement step 2.3)
        """
        _, rvec, tvec,_ = cv2.solvePnPRansac(landmark_3D,current_2D, K, None)
        ##### ####################################################### #######
        ##### Get Pose and Tranformation Matrix in world coordionates #######
        ##### ###############################################################
        rot, _ = cv2.Rodrigues(rvec)
        tvec = -rot.T.dot(tvec)  # coordinate transformation, from camera to world. What is the XYZ of the camera wrt World
        inv_transform = np.hstack((rot.T, tvec))  # inverse transform. A tranform projecting points from the camera frame to the world frame
        # abandon rebundant points(behind the direction)
        if not i:
            prevPose = tvec
        else:
            direction = tvec-prevPose
            x = direction[0]/np.abs(direction[0])
            y = direction[1]/np.abs(direction[1])
            z = direction[2]/np.abs(direction[2])
            idx = []
            for j in range(landmark_3D.shape[0]):
                p = landmark_3D[j,:]-tvec
                p[0] = p[0]/np.abs(p[0])
                p[1] = p[1]/np.abs(p[1])
                p[2] = p[2]/np.abs(p[2])
                if np.sum(p-[x,y,z])==0:
                    idx.append(j)
            landmark_3D = landmark_3D[idx]
            reference_2D = reference_2D[idx]
            prevPose = tvec

        ##### ################################# #######
        ##### Get 3D points Using Triangulation #######
        ##### #########################################
        # re-obtain the 3D points
        """
        Implement step 2.4)
        """
        landmark_3D_new, reference_2D_new= extract_keypoints_surf(curImage, curImage_R, K, baseline, reference_2D, landmark_3D)
        #Project the points from camera to world coordinates
        reference_2D = reference_2D_new.astype('float32')
        landmark_3D = inv_transform.dot(np.vstack((landmark_3D_new.T, np.ones((1, landmark_3D_new.shape[0])))))
        landmark_3D = landmark_3D.T

        ##### ####################### #######
        ##### Done, Next image please #######
        ##### ###############################
        reference_img = curImage
        i = max([i-1,0])
        ##### ################################## #######
        ##### START OF Print and visualize stuff #######
        ##### ##########################################
        # draw images
        draw_x, draw_y = int(tvec[0]) + 300, 600-(int(tvec[2]) + 100)
        true_x, true_y = int(truePose[i][3]) + 300, 600-(int(truePose[i][11]) + 100)

        curError = np.sqrt(
            (tvec[0] - truePose[i][3]) ** 2 +
            (tvec[1] - truePose[i][7]) ** 2 +
            (tvec[2] - truePose[i][11]) ** 2)
        
        if (curError > maxError):
            maxError = curError

        print(tvec[0],tvec[1],tvec[2], rvec[0], rvec[1], rvec[2])
        print([truePose[i][3], truePose[i][7], truePose[i][11]])
        
        text = "Coordinates: x ={0:02f}m y = {1:02f}m z = {2:02f}m".format(float(tvec[0]), float(tvec[1]),float(tvec[2]))
        cv2.circle(traj, (draw_x, draw_y), 1, (0, 0, 255), 2)
        cv2.circle(traj, (true_x, true_y), 1, (255, 0, 0), 2)
        cv2.rectangle(traj, (10, 30), (550, 50), (0, 0, 0), cv2.FILLED)
        cv2.putText(traj, text, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

        h1, w1 = traj.shape[:2]
        h2, w2 = curImage.shape[:2]
        vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        vis[:h1, :w1, :3] = traj
        vis[:h2, w1:w1 + w2, :3] = np.dstack((np.dstack((curImage,curImage)),curImage))

        cv2.imshow("Trajectory", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == 27: break


    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('Maximum Error: ', maxError)
    ##### ################################ #######
    ##### END OF Print and visualize stuff #######
    ##### ########################################

if __name__ == '__main__':
    left_img = getLeftImage(0)
    right_img = getRightImage(0)

    K = getK()

    playImageSequence(left_img, right_img, K)