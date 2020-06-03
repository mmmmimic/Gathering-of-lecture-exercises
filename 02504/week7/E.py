import cv2
from matplotlib import pyplot as plt
import sys

im1 = cv2.imread('TestIm3.png')
im2 = cv2.imread('TestIm4.png')
# detect orb feature
orb = cv2.ORB_create()
kp1, f1 = orb.detectAndCompute(im1, None)
kp2, f2 = orb.detectAndCompute(im2, None)
matcher = cv2.BFMatcher()
match = matcher.match(f1, f2)
match = sorted(match, key = lambda x:x.distance)
match = match[:50]
im3 = cv2.drawMatches(im1, kp1, im2, kp2, match, None)
plt.imshow(im3)
plt.show()

