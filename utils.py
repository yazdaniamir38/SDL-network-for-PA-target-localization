import torch.utils.data as data
import torch
import h5py, cv2
import numpy as np
import random
import os

def generate_SCIRD(theta, xAxis, yAxis, k, filter_size):
    xmin = -np.floor(filter_size/2)
    xmax = -xmin
    ymin = xmin
    ymax = xmax
    (x, y) = np.meshgrid(np.arange(xmin, xmax+1), np.arange(ymin,ymax+1))
    a1 = 1
    c1 = xAxis
    c2 = yAxis
    Z = 2*(np.pi)*c1*c2
    constY = 1/((c1 **2)*Z)
    xNew = x*np.cos(theta) + y*np.sin(theta)
    yNew = -x*np.sin(theta) + y*np.cos(theta)
    deriv = (((xNew + k*(yNew **2)) **2)/(c1 **2)) - 1
    gauss = np.exp(- (yNew **2)/(2*c2*c2))
    ridge = np.exp(-((xNew + k*(yNew **2)) **2)/(2*c1*c1))
    f = constY*deriv*gauss*ridge
    #f = (f - f.min())/(f.max() - f.min())
    #cv2.imshow('image',f)
    #cv2.waitKey(0)
    return f

def generate_filters():
    e = np.zeros([4, 4, 4])
    e[0, :, :] = np.concatenate([-np.ones([4, 2]), np.ones([4, 2])], 1)
    e[1, :, :] = np.concatenate([np.ones([4, 2]), -np.ones([4, 2])], 1)
    e[2, :, :] = np.concatenate([-np.ones([2, 4]), np.ones([2, 4])], 0)
    e[3, :, :] = np.concatenate([np.ones([2, 4]), -np.ones([2, 4])], 0)
    e = torch.from_numpy(e).unsqueeze(1).float()
    # Generating the wavefront filters
    W = np.zeros([10, 25, 25])
    for i in range(10):
        W[i, :, :] = generate_SCIRD(3.14, .1 * (i + 3), 20, .1, 25)
    W = torch.from_numpy(W).unsqueeze(1).float()
    # generating the noise filters
    N = np.zeros([20, 50, 15])
    for i in range(20):
        N[i, :, 7] = i * np.ones([1, 50])
    N[5:15, :, :] = np.zeros([10, 50, 15])
    for i in range(10):
        for m in range(15):
            N[i + 5, m, m] = i
            N[i + 5, m, 14 - m] = i
    N = torch.from_numpy(N).unsqueeze(1).float()
    return e,W,N
