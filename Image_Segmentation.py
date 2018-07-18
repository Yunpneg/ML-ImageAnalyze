# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt  # for imshow
import matplotlib.colors as colors
import matplotlib.patches as patches
from scipy.cluster import vq     # for k-means and vq
import random
import time


time1 = time.clock() # get the clock
im = plt.imread('/Users/chenyunpeng/Downloads/images/image01.jpg')
im = im/255    # convert to float in interval [0 1]
gim = np.mean(im[:,:,0:3],axis=2)    # find mean of RGB to create a 2D grayscale image (could be RGBA)
#gim = im

# display the image after mean implemention
plt.imshow(gim, cmap='gray')
plt.axis('off')

# perform local contrast normlisation
sgim = ndimage.gaussian_filter(gim,4)    # smooth the intensity image ('reflect' at boundaries)
dev = (gim-sgim)
V = ndimage.gaussian_filter(dev*dev,4)    # smooth the variance
gim = dev / np.maximum(np.sqrt(V), 0.1)

# Gray-levels are no longer in the interval [0 1], so provide normalisation to imshow
plt.figure()
norm = colors.Normalize(vmin=gim.min()/2, vmax=gim.max()/2)
plt.imshow(gim, cmap='gray', norm=norm)
plt.axis('off')


# First Layer
N = 15      # size of patch
K = 50     # number of prototypes for kmeans
P = 20000   # number of randomly selected patches

R = gim.shape[0]    # number of rows in image
C = gim.shape[1]    # number of columns in image

X = np.zeros((P,N*N),dtype=float) # initialise array for random patches

for i in range(0, P):
    r = random.randint(0,R-N)
    c = random.randint(0,C-N)
    patch = gim[r:r+N,c:c+N] 
    X[i,:] = np.reshape(patch,(-1))

codebook, distortion = vq.kmeans(X,K)

spn = np.ceil(np.sqrt(K))    # size of subplot display
norm = colors.Normalize(vmin=codebook.min(), vmax=codebook.max())    # set gray range from minimum to maximum
for i in range(0,K):
    plt.subplot(spn,spn,i+1)
    plt.imshow(np.reshape(codebook[i,:],(N,N)),cmap='gray',norm=norm)
    plt.gca().add_patch(patches.Circle((2,2), radius=1, color=plt.cm.tab20(i)))
    plt.axis('off')    # turn off the axes
    
X = np.zeros(((R-N)*(C-N),N*N),dtype=float)   # initialise array for all patches
i=0
for r in range(0,R-N):
    for c in range(0,C-N):
        X[i,:] = np.reshape(gim[r:r+N,c:c+N],(-1))
        i=i+1

code, dist = vq.vq(X,codebook)
code = np.reshape(code,(R-N,C-N))    # reshape the 1D code array into the original 2D image shape

# to give each label a unique colour, turn off normalisation in order to index directly into discrete colour map
plt.figure()
norm = colors.NoNorm()
plt.imshow(code, cmap='tab20', norm=norm)
plt.axis('off')    # turn off the axes


# Second layer
M = 30      # size of patch for second layer
R1 = code.shape[0]    # number of rows in image
C1 = len(code[0, :])  # number of columns in image
K2 = 7                # the number of features of image

Y1 = np.zeros((P,M*M),dtype=int)   # create the patch

for i in range(0, P):
    r = random.randint(0,R1-M)
    c = random.randint(0,C1-M)
    patch = code[r:r+M,c:c+M] 
    Y1[i,:] = np.reshape(patch,(-1))

# create the histograms
Y2 = np.zeros((P,K))
for j in range(0, P):   # each row is a histogram
    bins = np.bincount(Y1[j, :]) 
    if len(bins) < K:
        Y2[j, :] = np.hstack((bins, np.zeros(K-len(bins))))
    else:
        Y2[j, :] = bins
       
# Run k-means algortihm
codebook, distortion = vq.kmeans(Y2,K2)

# create an empty array for clustered image
Y3 = np.zeros(((R1-M)*(C1-M),K),dtype=int) 
i=0
for r in range(0,R1-M):
    for c in range(0,C1-M):
        # transform M*M pathces to one row, and implement bincount()
        bins = np.bincount(np.reshape(code[r:r+M,c:c+M],(-1))) 
        if len(bins) < K:
            Y3[i, :] = np.hstack((bins, np.zeros(K-len(bins))))
        else:
            Y3[i, :] = bins
        i=i+1
        
code, dist = vq.vq(Y3,codebook)
code = np.reshape(code,(R1-M,C1-M))

plt.figure()
norm = colors.NoNorm()
plt.imshow(code, cmap='tab20', norm=norm)
plt.axis('off')
# compute the running time
elpased = (time.clock() - time1)
print(elpased)