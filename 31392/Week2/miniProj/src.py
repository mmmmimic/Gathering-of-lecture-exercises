import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
# 基本思路：
# 从视频中获取一帧->提取特征点(Shi-Tomashi Corner)->匹配下一帧中对应的特征点，生成optical flow
# 存在问题：特征点的位置飘忽不定，画的线不稳定
# 分析原因：应该是每帧都对图像重新进行corners提取的原因
# 解决方案：画的线不稳定可以通过储存历史数据来解决*history[list]
# 依然存在的问题：点不稳定，还有就是如何用一个mask把不是机器人上的点遮住，甚至遮住另一个机器人上的点(多目标追踪？)
# 突然发现用history太蠢了，我只是要画图而已，使用cv2.add()，用一个空白的mask就好
# 结果果然大大加速了，毕竟避免了一个嵌套的遍历。。。而且点的波动变小了。神奇。。。接下来还是要解决把背景的点去掉的问题
# 我有两个思路，一个是特征提取识别机器人，一个是纯通过光流来判断，移动比较大的点对应的是机器人
# 还有一个思路是引入先验知识，先确定一个机器人的区域，以选择feat0
# 对于追踪机器人，老师给的hint是，使用SIFT或者SURF等方法，来识别机器人。
# 但是我觉得因为是处理视频，所以我要追求计算速度，所以使用ORB或者FAST特征更好些


#def draw_hist(history, img):
#    # input history is a list
#    # input img is an image
#    for i in range(len(history)-1):
#        feat0 = history[i]
#        feat1 = history[i+1]
#        color = (0,255,0) # green line
#        for k in range(len(feat1)):
#            cv2.line(img, (feat0[k][0][0], feat0[k][0][1]), (feat1[k][0][0], feat1[k][0][1]), color, 2)
#    return img 

cap = cv2.VideoCapture('Robots.mp4')
#cap = cv2.VideoCapture('Challenge.mp4')
ret, frame0 = cap.read() # read the first frame
gray0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
feat0 = cv2.goodFeaturesToTrack(gray0, mask = None, maxCorners=300, qualityLevel=0.2, minDistance=2, blockSize=7)
#history = []
mask = np.zeros_like(frame0)# store the former lines
sparse = True # True是sparse，False是dense

while(ret and sparse):
    ret, frame = cap.read() # each loop, read a frame
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # 首先使用saprse optical flow
    feat1, status, error = cv2.calcOpticalFlowPyrLK(gray0, gray, feat0, None)
    #history.append(feat0)
    #history.append(feat1)
    # draw track
    #draw_hist(history, frame)
    for i in range(len(feat1)):
        cv2.line(mask, (feat0[i][0][0], feat0[i][0][1]), (feat1[i][0][0], feat1[i][0][1]), (0, 255, 0), 2)
        cv2.circle(frame, (feat0[i][0][0], feat0[i][0][1]), 5, (255, 0, 0), -1)
        cv2.circle(frame, (feat1[i][0][0], feat1[i][0][1]), 5, (0, 0, 255), -1)
    while (1):
        cv2.imshow('feature track', cv2.add(frame, mask))
        k = cv2.waitKey(1)
        break
    feat0 = feat1
    gray0 = gray

while(ret and not sparse):
    ret, frame = cap.read() # each loop, read a frame
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # 使用dense optical flow
    flow = cv2.calcOpticalFlowFarneback(gray0, gray, None, 0.5, 3, 15, 3, 5, 1.5, 0)
    down_size = 16 #每16个像素点画一次
    for i in range(flow.shape[0]):
        for j in range(flow.shape[1]):
            if i%down_size==1 and j%down_size==1:
                cv2.line(mask, (j, i), (int(j+flow[i][j][1]), int(i+flow[i][j][0])), (255, 0, 0), 2) #轨迹用蓝色
                cv2.circle(frame, (int(j+flow[i][j][0]), int(i+flow[i][j][1])), 2, (0, 255, 0), -1)
    while (1):
        cv2.imshow('feature track', cv2.add(frame, mask))
        k = cv2.waitKey(1)
        break
    gray0 = gray

cv2.destroyAllWindows()
cap.release() # end of the video

