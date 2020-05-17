import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
# 基本思路：
# 我有两个思路，一个是特征提取识别机器人（有一个问题，我怎么提取机器人的特征？就crop？训练集是什么？），一个是纯通过光流来判断，移动比较大的点对应的是机器人
# 其实还有一个最暴力的思路是引入先验知识，先确定一个机器人的区域，以选择feat0
# 对于追踪单个机器人，老师给的hint是，使用SIFT或者SURF等方法，用描述子(descriptor)来识别机器人。
# 但是我觉得因为是处理视频，所以我要追求计算速度，所以使用ORB特征更好些，不为别的，因为快
# 其实我还有一个思路，前提是要能把背景的点消除，考虑到两个机器人一开始离得比较远，用聚类将它们的特征分开，用最简单的k-means就行(是否为凸？)
# 那么现在问题就是怎么去掉背景中的点
# 我先是用了最暴力的方法，发现处理challenge视频的时候两个机器人互相遮挡，果然不行了。应该是feature不太好。
# 换成ORB试试。
# SIFT慢还不靠谱，ORB快但不靠谱
# 决定区域搜索匹配的特征，而不是全局，这需要我创建一个比robotmask稍大的mask。
# 失败了
# 只完成了挑战1（用的还是角点，而不是特征提取）
# 深度学习也许可以解决问题



cap = cv2.VideoCapture('Robots.mp4')
#cap = cv2.VideoCapture('Challenge.mp4')
ret, frame0 = cap.read() # read the first frame
gray0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
mask = np.zeros_like(frame0)# store the former lines
# define a feature extractor
#cl = cv2.xfeatures2d_SIFT.create()
cl = cv2.ORB_create()

# switch
optical = True #是否用opitcal flow
orbon = not optical #是否用特征+matcher

# crop a robot
robotMask = np.zeros_like(gray0)
roi = cv2.selectROI('gray', gray0) # select the first robot I wanna track in the image
x,y,w,h = roi
robotMask[y:y+h, x:x+w] = 1 #the robot in the left
if optical:
    feat0 = cv2.goodFeaturesToTrack(gray0, mask = robotMask, maxCorners=100, qualityLevel=0.2, minDistance=2, blockSize=7)
if orbon:
    kps1, des1 = cl.detectAndCompute(gray0, robotMask)
#roi = cv2.selectROI('gray', gray0) # select the first robot I wanna track in the image
#x,y,w,h = roi
#robotMask[y:y+h, x:x+w] = 1 #the robot in the left
#if optical:
#    feat2 = cv2.goodFeaturesToTrack(gray0, mask = robotMask, maxCorners=100, qualityLevel=0.2, minDistance=2, blockSize=7)


while(ret and optical):
    ret, frame = cap.read() # each loop, read a frame
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    feat1, status, error = cv2.calcOpticalFlowPyrLK(gray0, gray, feat0, None)
    #feat3, status, error = cv2.calcOpticalFlowPyrLK(gray0, gray, feat2, None)
    for i in range(len(feat1)):
        cv2.line(mask, (feat0[i][0][0], feat0[i][0][1]), (feat1[i][0][0], feat1[i][0][1]), (0, 255, 0), 2)
        cv2.circle(frame, (feat0[i][0][0], feat0[i][0][1]), 5, (255, 0, 0), -1)
        cv2.circle(frame, (feat1[i][0][0], feat1[i][0][1]), 5, (0, 0, 255), -1)
    #for i in range(len(feat3)):
    #    cv2.line(mask, (feat2[i][0][0], feat2[i][0][1]), (feat3[i][0][0], feat3[i][0][1]), (255, 0, 0), 2)
    #    cv2.circle(frame, (feat2[i][0][0], feat2[i][0][1]), 5, (255, 0, 0), -1)
    #    cv2.circle(frame, (feat3[i][0][0], feat3[i][0][1]), 5, (0, 0, 255), -1)
    while (1):
        cv2.imshow('feature track', cv2.add(frame, mask))
        k = cv2.waitKey(1)
        break
    feat0 = feat1
    gray0 = gray



matcher = cv2.BFMatcher()
nb_matches = 10
counter = 0
#gap = 30
while(ret and orbon):
    ret, frame = cap.read() # each loop, read a frame
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #searchMask = np.zeros_like(gray) # mask of local search
    #points = [point.pt for point in kps1]
    #minx = np.int(min(points[:][0]))
    #miny = np.int(min(points[:][1]))
    #maxx = np.int(max(points[:][0]))
    #maxy = np.int(max(points[:][1]))
    #searchMask[max([miny-gap,0]):min(maxy+gap,gray.shape[1]), max([minx-gap,0]):min(maxx+gap,gray.shape[0])] = 1
    kps2, des2 = cl.detectAndCompute(gray, searchMask)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key= lambda x: x.distance)
    matches = matches[:nb_matches]
    if not counter:
        feat0 = [point.pt for point in kps1]
        feat0 = [[np.int(feat0[idx][0]), np.int(feat0[idx][1])] for idx in [match.queryIdx for match in matches]]
    feat1 = [point.pt for point in kps2]
    feat1 = [[np.int(feat1[idx][0]), np.int(feat1[idx][1])] for idx in [match.trainIdx for match in matches]]
    for i in range(nb_matches):
        cv2.line(mask, (feat0[i][0], feat0[i][1]), (feat1[i][0], feat1[i][1]), (0, 255, 0), 2)
        cv2.circle(frame, (feat0[i][0], feat0[i][1]), 5, (255, 0, 0), -1)
        cv2.circle(frame, (feat1[i][0], feat1[i][1]), 5, (0, 0, 255), -1)
    while (1):
        cv2.imshow('feature track', cv2.add(frame, mask))
        k = cv2.waitKey(1)
        break
    feat0 = feat1
    counter = counter+1

cv2.destroyAllWindows()
cap.release() # end of the video

