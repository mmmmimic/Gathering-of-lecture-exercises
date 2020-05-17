import cv2
import numpy as np
rvec = np.array([[-0.05], [-1.51],[-0.00]])
tvec = np.array([[87.39],[-2.25],[-24.89]])
R = cv2.Rodrigues(rvec)[0]
R = np.linalg.inv(R)
tvec = -np.dot(R, tvec)
print(np.dot(R, np.array([[-6.71],[0.23],[21.59]]))+tvec)