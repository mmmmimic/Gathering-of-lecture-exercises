# exercise 3.3.1

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from similarity import similarity

# Image to use as query
i = 1

# Similarity: 'SMC', 'Jaccard', 'ExtendedJaccard', 'Cosine', 'Correlation' 
similarity_measure = 'SMC'

# Load the digits
# Load Matlab data file to python dict structure
X = loadmat('../Data/digits.mat')['X']
# You can also try the CBCL faces dataset (remember to change 'transpose')
#X = loadmat('../Data/wildfaces_grayscale.mat')['X']
N, M = X.shape
transpose = False # should the plotted images be transposed? 


# Search the face database for similar faces
# Index of all other images than i
noti = list(range(0,i)) + list(range(i+1,N)) 
# Compute similarity between image i and all others
sim = similarity(X[i,:], X[noti,:], similarity_measure)
sim = sim.tolist()[0]
# Tuples of sorted similarities and their indices
sim_to_index = sorted(zip(sim,noti))


# Visualize query image and 5 most/least similar images
plt.figure(figsize=(12,8))
plt.subplot(3,1,1)

img_hw = int(np.sqrt(len(X[0])))
img = np.reshape(X[i], (img_hw,img_hw))
if transpose: img = img.T
plt.imshow(img, cmap=plt.cm.gray)
plt.xticks([]); plt.yticks([])
plt.title('Query image')
plt.ylabel('image #{0}'.format(i))


for ms in range(5):

    # 5 most similar images found
    plt.subplot(3,5,6+ms)
    im_id = sim_to_index[-ms-1][1]
    im_sim = sim_to_index[-ms-1][0]
    img = np.reshape(X[im_id],(img_hw,img_hw))
    if transpose: img = img.T
    plt.imshow(img, cmap=plt.cm.gray)
    plt.xlabel('sim={0:.3f}'.format(im_sim))
    plt.ylabel('image #{0}'.format(im_id))
    plt.xticks([]); plt.yticks([])
    if ms==2: plt.title('Most similar images')

    # 5 least similar images found
    plt.subplot(3,5,11+ms)
    im_id = sim_to_index[ms][1]
    im_sim = sim_to_index[ms][0]
    img = np.reshape(X[im_id],(img_hw,img_hw))
    if transpose: img = img.T
    plt.imshow(img, cmap=plt.cm.gray)
    plt.xlabel('sim={0:.3f}'.format(im_sim))
    plt.ylabel('image #{0}'.format(im_id))
    plt.xticks([]); plt.yticks([])
    if ms==2: plt.title('Least similar images')
    
plt.show()

print('Ran Exercise 3.3.1')