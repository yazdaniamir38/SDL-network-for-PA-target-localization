#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 17:58:36 2018

@author: kfj5051
"""


import torch
import torchvision   # throws warning but this import is necessary
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as si
import dsntnn
import h5py
import os
from dataset import DatasetFromHdf5
import simple_enc
#-------------------------------------------------
seed1 = np.random.randint(low = 0,high = 100)
seed2 = np.random.randint(low = 0,high = 100)
seeds = (seed1,seed2)

torch.manual_seed(seed1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(seed2)
#-----------------------------------------------------------------------------
#          -Load Data-
#-----------------------------------------------------------------------------

# pretrained_model=simple_enc.CoordRegressionNetwork(n_locations=1).cuda()
# pretrained_model.load_state_dict(torch.load('./model/theirs/100.pth'))
data_path = './PA_Target_Detection_Dataset/';
patch_path='./PA_Target_Detection_Dataset/'
files='./PA_Target_Detection_Dataset/Codes_to_Share/files/10cm/'
test_files='./PA_Target_Detection_Dataset/Codes_to_Share/'
# samplefile=h5py.File('./PA_Target_Detection_Dataset/samples/samples1.h5','r')
# sample_inp=samplefile.get("Input")
# sample_tar=samplefile.get("Target")
# sam_inp=sample_inp[0:90,:,:].reshape(90,1,256,1024)
# sam_inp=torch.from_numpy(sam_inp).cuda()
# sam_tar=torch.from_numpy(sample_tar[0:90,:,:]).cuda()
# data = si.loadmat(''.join([data_path, 'New.mat']))['New']
# data=np.rollaxis(data,3,2)
# data=torch.from_numpy(data)



# with h5py.File(''.join([data_path, 'Y_Test_20.mat']),'r') as data:
# y=data['Y_Test']

Load_Data = False
if Load_Data == True:
    # train_set = DatasetFromHdf5(os.walk(files),yes=1)
    # training_data_loader =torch.utils.data.DataLoader(dataset=train_set, num_workers=1, batch_size=3, shuffle=True)
    # Load sensor testing data
    with h5py.File(''.join([patch_path, 'Codes_to_Share/X_Test_mixed.mat']), 'r') as data:
        X_Test=data['XTest'][:,:,:,4000:6000]
        # X_Test = np.concatenate((X_Test,data['X_Test'][:, :, :, 2000:3000]),axis=3)
        # X_Test = np.concatenate((X_Test, data['X_Test'][:, :, :, 4000:5000]), axis=3)
    X_Test=np.transpose(X_Test)
    X_Test=torch.from_numpy(X_Test)
    # with h5py.File(''.join([patch_path,'./X_Test_20_Noisy_-9dB.mat']),'r') as data:
    #     X_Test = data['X_Test'][:]
    # X_Test = np.transpose(X_Test)
    #
    # X_Test = torch.from_numpy(X_Test)
    # x=si.loadmat(''.join([data_path,'experimental.mat']))['data']
    # X_Test = X_Test.permute(3,2,1,0)

    # with h5py.File(''.join([data_path,'X_Test_20_noiseless.mat']), 'r') as data:
    #     X_Test_clean = data['X_Test'][:]
    # X_Test_clean=np.transpose(X_Test_clean)
    # X_Test_clean = torch.from_numpy(X_Test_clean).unsqueeze(1)
    # X_Test_clean = X_Test_clean.reshape(X_Test_clean.shape[0], 1, X_Test_clean.shape[1], X_Test_clean.shape[2])
    # Load sensor training data
    with h5py.File(''.join([data_path, 'Codes_to_Share/X_Train_mixed.mat']), 'r') as data:
        X_Train=data['XTrain'][:,:,:,16000:24000]
        # X_Train=np.concatenate((X_Train,data['XTrain'][:,:,:,9000:10000]),axis=3)
        # X_Train = np.concatenate((X_Train, data['XTrain'][:, :, :, 18000:20000]), axis=3)
        X_Clean = data['XTrain'][:, :, :, 0:8000]
        # X_Clean = np.concatenate((X_Clean,X_Clean,X_Clean),axis=3)
        # X_Clean = np.concatenate((X_Clean, data['XTrain'][:, :, :, 2000:4000]), axis=3)
    X_Train=np.transpose(X_Train)
    X_Clean=np.transpose(X_Clean)
    X_Train=torch.from_numpy(X_Train)
    X_Clean=torch.from_numpy(X_Clean)
    # with h5py.File(''.join([data_path,'X_Train_20_Noisy_-9dB.mat']), 'r') as data:
    #     X_Train = data['X_Train'][:]
    # X_Train=np.transpose(X_Train)
    # X_Train = torch.from_numpy(X_Train)
    # X_Train=np.transpose(X_Train)
    # X_Train[7006,:,:,:] = X_Test[4060,0,:,:]   #7006 was blank
    # X_Test = X_Test[0:4060,:,:,:]  #4061 was blank and 4060 is transferred to the training set
    data=si.loadmat(''.join([data_path, 'Codes_to_Share/Y_Test_mixed.mat']))

    # with h5py.File(''.join([data_path, 'Y_Test_20.mat']),'r') as data:
    # y=data['Y_Test']
    Y_te = data['YTest'][0:2000,:,:]
    # # Load label data for test set
    # data = si.loadmat(''.join([data_path, '/Codes_to_Share/Y_Train_high_scattering.mat']))
    # Y_te = data['Y_Train'][:]
    # Y_te=np.reshape(Y_te,[1,1,2])
    # Y_te[0:5,:,:]=y
    # Y_te[np.logical_and(Y_te[:, :, 1] == 30, Y_te[:, :, 0] == 0), :] = [-20, 10]
    Y_te = torch.from_numpy(Y_te).float()
    # Y_te[:,:,[0,1]]=Y_te[:,:,[1,0]]
    # Y_Te = torch.from_numpy(Y_te).unsqueeze(1).float()
    Y_Test = torch.Tensor(2000,4,2)
    # # Normalize so target positions are between -1 and 1 (top left corner is (-1,-1), bottom right corner is (1,1))
    max_ax = 50
    min_ax = 10
    max_normalized_depth = 0.9
    ax = (Y_te[:,:,1]-(max_ax-min_ax)/2-min_ax)/((max_ax-min_ax)/2/max_normalized_depth)
    lat = Y_te[:,:,0]/((max_ax-min_ax)/2/max_normalized_depth)
    Y_Test[:,:,0] = ax
    Y_Test[:,:,1] = lat
    #
    # Load label data for training set
    data=si.loadmat(''.join([data_path, 'Codes_to_Share/Y_Train_mixed.mat']))
    # with h5py.File(''.join([data_path, 'Y_Train_20.mat']),'r') as data:
    Y_tr = data['YTrain'][0:8000,:,:]

    # Y_tr = np.transpose(Y_tr)
    Y_Tr = torch.from_numpy(Y_tr).float()
    Y_Train = torch.Tensor(8000,4,2)
    # Normalize so target positions are between -1 and 1 (top left corner is (-1,-1), bottom right corner is (1,1))
    ax = (Y_Tr[:,:,1]-(max_ax-min_ax)/2-min_ax)/((max_ax-min_ax)/2/max_normalized_depth)
    lat = Y_Tr[:,:,0]/((max_ax-min_ax)/2/max_normalized_depth)
    Y_Train[:,:,0] = ax
    Y_Train[:,:,1] = lat
    # Y_Train[7006,:] = Y_Test[4060,:]  #again, 7006 was blank
    # Y_Test = Y_Test[0:4060,:]   #same as above
    # Y_Train = Y_Train.unsqueeze(1).float()
    # Y_Test = Y_Test.unsqueeze(1).float()

    # with h5py.File(''.join([data_path,'X_Train_20_noiseless.mat']), 'r') as data:
    #   X_Clean = data['X_Train'][:]
    # X_Clean=np.concatenate((X_Clean,X_Clean,X_Clean),axis=3)
    # X_Clean = torch.from_numpy(X_Clean).unsqueeze(1)
    # X_Clean = np.transpose(X_Clean)
    # X_Clean=X_Clean.reshape(X_Clean.shape[0],1,X_Clean.shape[1],X_Clean.shape[2])
    # X_Clean[7006,:,:,:] = X_Test_clean[4060,0,:,:]   #7006 was blank
    # perm=si.loadmat('./model/low_training_tuffc/perm.mat')['perm']
    # perm=torch.squeeze(torch.from_numpy(perm))
    # X_Train=X_Train[perm,:,:,:]
    # X_Train=X_Train[0:8120,:,:,:]
    # X_Clean=X_Clean[perm,:,:,:]
    # X_Clean=X_Clean[0:8120,:,:,:]c
    # Y_Train=Y_Train[perm,:,:]
    # Y_Train=Y_Train[0:8120,:,:]
    # X_Test = X_Test[0:4060,:,:,:]  #4061 was blank and 4060 is transferred to the training set
    # for i in range(24):
    #     h5fw = h5py.File(str(patch_path  + '_' + str(i+1) + '.h5'), 'w')
    #     dset_input = h5fw.create_dataset(name='INPUT', shape=[1000,1,256,1024], data=X_Train[1000*i:1000*(i+1),:,:,:], dtype=np.float32)
    #     if i% 8 == 7:
    #         dset_clean = h5fw.create_dataset(name='Clean', shape=[1000,1,256,1024], data=X_Clean[7000:8000,:,:,:],
    #                                          dtype=np.float32)
    #     else:
    #         dset_clean = h5fw.create_dataset(name='Clean', shape=[1000,1,256,1024], data=X_Clean[(1000*i)%8000:(1000*(i+1))%8000,:,:,:], dtype=np.float32)
    #
    #     dset_target = h5fw.create_dataset(name='TARGET', shape=[1000,4,2],
    #                                      data=Y_Train[1000*i:1000*(i+1), :, :,],
    #                                      dtype=np.float32)
    #     h5fw.close()
Test_data=True
if Test_data:
    # with h5py.File(''.join([patch_path, './X_Test_20_Noisy_-9dB.mat']), 'r') as data:
    #     X_Test = data['X_Test'][:]
    # X_Test = np.transpose(X_Test)
    with h5py.File(''.join([patch_path, 'Codes_to_Share/X_Test_mixed.mat']), 'r') as data:
        X_Test=data['XTest'][:]
    X_Test = np.transpose(X_Test)
    X_Test = torch.from_numpy(X_Test)
    data = si.loadmat(''.join([data_path, 'Codes_to_Share/Y_Test_mixed.mat']))

    # with h5py.File(''.join([data_path, 'Y_Test_20.mat']),'r') as data:
    # y=data['Y_Test']
    Y_te = data['YTest'][:]
    # # Load label data for test set
    # data = si.loadmat(''.join([data_path, '/Codes_to_Share/Y_Train_high_scattering.mat']))
    # Y_te = data['Y_Train'][:]
    # Y_te=np.reshape(Y_te,[1,1,2])
    # Y_te[0:5,:,:]=y
    # Y_te[np.logical_and(Y_te[:, :, 1] == 30, Y_te[:, :, 0] == 0), :] = [-20, 10]
    Y_te = torch.from_numpy(Y_te).float()
    # Y_te[:,:,[0,1]]=Y_te[:,:,[1,0]]
    # Y_Te = torch.from_numpy(Y_te).unsqueeze(1).float()
    Y_Test = torch.Tensor(6000, 4, 2)
    # # Normalize so target positions are between -1 and 1 (top left corner is (-1,-1), bottom right corner is (1,1))
    max_ax = 50
    min_ax = 10
    max_normalized_depth = 0.9
    ax = (Y_te[:, :, 1] - (max_ax - min_ax) / 2 - min_ax) / ((max_ax - min_ax) / 2 / max_normalized_depth)
    lat = Y_te[:, :, 0] / ((max_ax - min_ax) / 2 / max_normalized_depth)
    Y_Test[:, :, 0] = ax
    Y_Test[:, :, 1] = lat
#     #
#     X_Test = torch.from_numpy(X_Test)
#     X_Test = X_Test[0:4060, :, :, :]
#     with h5py.File(''.join([data_path, 'Y_Test_20.mat']), 'r') as data:
#         # y=data['Y_Test']
#         Y_te = data['Y_Test'][:]
#         # # Load label data for test set
        # data = si.loadmat(''.join([data_path, '/Codes_to_Share/Y_Train_high_scattering.mat']))
#         # Y_te = data['Y_Train'][:]
#         # Y_te=np.reshape(Y_te,[1,1,2])
#         # Y_te[0:5,:,:]=y
#         # Y_te[np.logical_and(Y_te[:, :, 1] == 30, Y_te[:, :, 0] == 0), :] = [-20, 10]
#     Y_te = np.transpose(Y_te)
    # Y_te[:,:,[0,1]]=Y_te[:,:,[1,0]]
    # Y_Te = torch.from_numpy(Y_te).unsqueeze(1).float()
    # Y_Test = torch.Tensor(4062, 1, 2)
    # # Normalize so target positions are between -1 and 1 (top left corner is (-1,-1), bottom right corner is (1,1))
    # max_ax = 50
    # min_ax = 10
    # max_normalized_depth = 0.9
    # ax = (Y_Te[:, :, 1] - (max_ax - min_ax) / 2 - min_ax) / ((max_ax - min_ax) / 2 / max_normalized_depth)
    # lat = Y_Te[:, :, 0] / ((max_ax - min_ax) / 2 / max_normalized_depth)
    # Y_Test[:, :, 0] = ax
    # Y_Test[:, :, 1] = lat
    # Y_Test = Y_Test[0:4060, :]
    #
bs =4# batch size
# train = torch.utils.data.TensorDataset(X_Train, Y_Train,X_Clean)
# trainloader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=False)
test=torch.utils.data.TensorDataset(X_Test, Y_Test)
testloader = torch.utils.data.DataLoader(test, batch_size=bs, shuffle=False)

# test = DatasetFromHdf5(os.walk(test_files),yes=0)
# testloader =torch.utils.data.DataLoader(dataset=test, batch_size=4, shuffle=False)
train_set = DatasetFromHdf5(os.walk(files),yes=1)
trainloader =torch.utils.data.DataLoader(dataset=train_set, num_workers=1, batch_size=3, shuffle=True)
#
#-----------------------------------------------------------------------------
#          -Model definition-
#-----------------------------------------------------------------------------

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, expanding=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.expanding = expanding
        self.planes = planes
        self.resconv = conv3x3(inplanes, planes)
        

    def forward(self, x):
        if self.expanding:
            residual = x[:,0:self.planes,:,:]
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out
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

class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3,stride=1,padding=1)

    def forward(self, present, former):
        present = self.up(present)
        present = self.conv(present)
        x = torch.cat([present, former], dim=1)
        return x


class ResNet(nn.Module):

    def __init__(self, block, num_classes=9):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(7,7), stride=(1,1), dilation=(1,4),padding=(3,12), bias=False)
        # self.convs=nn.Conv2d(1,1,kernel_size=7,stride=(4,16),padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pool1d = nn.MaxPool2d(kernel_size=(1,2))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, 16, blocks=1)
        self.layer2 = self._make_layer(block, 16, 32, blocks=1, stride=2)
        self.layer3 = self._make_layer(block, 32, 64, blocks=1, stride=2)
        self.layer4 = self._make_layer(block, 64, 128, blocks=1, stride=2)
        self.layer5 = self._make_layer(block, 128, 256, blocks=1, stride=2)
        self.classify = nn.Conv2d(256, 256, kernel_size=(5,5), padding=(2,2))
        self.bn2 = nn.BatchNorm2d(256)
        self.layer6 = self._make_layer(block, 256, 256, blocks=1, stride=2) #subsample w/o adding features
        self.classify2 = nn.Conv2d(256, 256, kernel_size=(5,5),padding=(2,2)) #8x8
        self.bn3 = nn.BatchNorm2d(256)
        self.layer7 = self._make_layer(block, 256, 256, blocks=1, stride=1)
        self.layer8 = self._make_layer(block, 256, 256, blocks=1, stride=1)
        self.up1 = up(256, 256)
        self.up6=up(256,256)
        self.layer9= self._make_layer(block, 512, 256, blocks=1, stride=1, expanding=True)
        self.dec9=self._make_layer(block, 512, 256, blocks=1, stride=1, expanding=True)
        self.up2 = up(256, 128)
        self.up7 = up(256, 128)
        self.layer10 = self._make_layer(block, 256, 128, blocks=1, stride=1, expanding=True)
        self.dec10 = self._make_layer(block, 256, 128, blocks=1, stride=1, expanding=True)
        self.up3 = up(128, 64)
        self.up8 = up(128, 64)
        self.layer11 = self._make_layer(block, 128, 64, blocks=1, stride=1, expanding=True)
        self.dec11 = self._make_layer(block, 128, 64, blocks=1, stride=1, expanding=True)
        self.up4 = up(64, 32)
        self.up9 = up(64, 32)
        self.layer12 = self._make_layer(block, 64, 32, blocks=1, stride=1, expanding=True)
        self.dec12 = self._make_layer(block, 64, 32, blocks=1, stride=1, expanding=True)
        self.up5 = up(32, 16)
        self.up10 = up(32, 16)
        self.layer13 = self._make_layer(block, 32, 16, blocks=1, stride=1, expanding=True)
        self.dec13 = self._make_layer(block, 32, 16, blocks=1, stride=1, expanding=True)
        self.up11=nn.UpsamplingNearest2d(scale_factor=2)
        self.dec14=self._make_layer(block, 16, 8, blocks=1, stride=1, expanding=True)
        # self.up12 = nn.UpsamplingNearest2d(scale_factor=4)
        # self.dec15 = self._make_layer(block, 8, 4, blocks=1, stride=1, expanding=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, expanding=False):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )            
        layers = []
        layers.append(block(inplanes, planes, stride, downsample, expanding))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
            
        return nn.Sequential(*layers)
    
    def display(self, x, vis):
        if vis == True:
            plt.imshow(x[0,0,:,:].detach().cpu().numpy()) #first in the batch
            plt.show()
            
    def forward(self, x, remember=False,sample=False):
        # xs = self.convs(x)

            x1 = self.conv1(x)
            xhist = {'x1': x}
            x2 = self.bn1(x1)
            xhist['x2'] = x2
            x3 = self.relu(x2)
            xhist['x3'] = x3
            x4 = self.pool1d(x3)       #256x256x16  <-outputs
            xhist['x4'] = x4
            x5 = self.layer1(x4)       #256x256x16
            xhist['x5'] = x5
            x6 = self.layer2(x5)       #128x128x32
            xhist['x6'] = x6
            x7 = self.layer3(x6)       #64x64x64
            xhist['x7'] = x7

            if sample:
                return (x7[:,0,:,:])
            else:
                x8 = self.layer4(x7)       #32x32x128
                xhist['x8'] = x8
                x9 = self.layer5(x8)       #16x16x256
                xhist['x9'] = x9
                x10 = self.classify(x9)    #16x16x256
                xhist['x10'] = x10
                x11 = self.bn2(x10)        #BN
                xhist['x11'] = x11
                x12 = self.relu(x11)       #ReLU
                xhist['x12'] = x12
                x13 = self.layer6(x12)     #8x8x256
                xhist['x13'] = x13
                x14 = self.classify2(x13)  #8x8x256
                xhist['x14'] = x14
                x15 = self.bn3(x14)        #BN
                xhist['x15'] = x15
                x16 = self.relu(x15)       #ReLU
                xhist['x16'] = x16
                x17 = self.layer7(x16)     #8x8x256
                xhist['x17'] = x17
                x18 = self.layer8(x17)     #8x8x256
                xhist['x18'] = x18
                x19 = self.up1(x18,x10)    #16x16x512
                xhist['x19'] = x19
                x20 = self.layer9(x19)     #16x16x256
                xhist['x20'] = x20

                xdec19 = self.up6(x18, x10)  # 16x16x512
                # xhist['x19'] = x19
                xdec20 = self.dec9(xdec19)  # 16x16x256


                x21 = self.up2(x20,x8)     #32x32x256
                xhist['x21'] = x21
                x22 = self.layer10(x21)    #32x32x128
                xhist['x22'] = x22

                xdec21 = self.up7(xdec20, x8)  # 32x32x256
                # xhist['x21'] = x21
                xdec22 = self.dec10(xdec21)  # 32x32x128

                x23 = self.up3(x22,x7)     #64x64x128
                xhist['x23'] = x23
                x24 = self.layer11(x23)    #64x64x64
                xhist['x24'] = x24

                xdec23 = self.up8(xdec22, x7)  # 64x64x128
                # xhist['x23'] = x23
                xdec24 = self.dec11(xdec23)  # 64x64x64
                # xhist['x24'] = x24

                x25 = self.up4(x24,x6)     #128x128x64
                xhist['x25'] = x25
                x26 = self.layer12(x25)    #128x128x32
                xhist['x26'] = x26

                xdec25 = self.up9(xdec24, x6)  # 128x128x64
                # xhist['x25'] = x25
                xdec26 = self.dec12(xdec25)  # 128x128x32

                x27 = self.up5(x26,x5)     #256x256x32
                xhist['x27'] = x27
                x28 = self.layer13(x27)    #256x256x16

                xdec27 = self.up10(xdec26, x5)  # 256x256x32
                # xhist['x27'] = xdec27
                xdec28 = self.dec13(xdec27)  # 256x256x16

                xdec29 = self.up11(xdec28)  # 256x256x32
                # xhist['x27'] = xdec27
                xdec30 = self.dec14(xdec29)  # 256x256x16
                #
                # xdec31 = self.up12(xdec30)  # 256x256x32
                # # xhist['x27'] = xdec27
                # xdec31 = self.dec15(xdec31)  # 256x256x16

                xhist['x28'] = x28
                return x28, xdec30, xhist


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, **kwargs)
    return model

class CoordRegressionNetwork(nn.Module):
    def __init__(self,n_locations):
        super(CoordRegressionNetwork, self).__init__()
        self.fcn = resnet18()
        self.hm_conv = nn.Conv2d(16, n_locations, kernel_size=1,bias=False)
        self.hm_conv2 = nn.Conv2d(8, 1, kernel_size=1,stride=(2,1), bias=False)
        # self.ridge=nn.Conv2d(1,5,kernel_size=25,bias=False)
        # self.noise=nn.Conv2d(1,10,kernel_size=[50,15],bias=False)
        # self.maxpool1=nn.MaxPool2d(5,stride=[10,50])
        # self.maxpool2=nn.MaxPool2d(4,stride=[4,50])
        self.relu=nn.ReLU(inplace=True)
        # self.f=f
        # self.f1=f1
        # self.f2=f2
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, images,sample=False):
        if sample:
            xs=self.fcn(images,sample=sample)
            return xs
        else:
            fcn_out,dec_out, xhist = self.fcn(images)
            unnormalized_heatmaps = self.hm_conv(fcn_out)
            clean_output=self.hm_conv2(dec_out)
            heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
            coords = dsntnn.dsnt(heatmaps)
            # input_edges=self.relu(nn.functional.conv2d(images,self.f))
            # input_edges=torch.add(torch.add(input_edges[:,0:1,:,:],input_edges[:,1:2,:,:]),torch.add(input_edges[:,2:3,:,:],input_edges[:,3:4,:,:]))
            # clean_edges=self.relu(nn.functional.conv2d(clean_output,self.f))
            # clean_edges=torch.add(torch.add(clean_edges[:,0:1,:,:],clean_edges[:,1:2,:,:]),torch.add(clean_edges[:,2:3,:,:],clean_edges[:,3:4,:,:]))
            # mult=torch.mul(clean_edges,input_edges)
            # out_noise=(torch.cat([self.noise(mult),nn.functional.conv2d(mult,self.f2)],dim=1))
            # out_ridge=(torch.cat([self.ridge(mult),nn.functional.conv2d(mult,self.f1)],dim=1))
            # ridge=self.maxpool1(out_ridge)
            # noise=self.maxpool2(out_noise)
            return coords, heatmaps, clean_output,xhist#,ridge,noise

class final(nn.Module):
    def __init__(self,f,f1,f2):
        super(final, self).__init__()
        self.main_model=CoordRegressionNetwork(n_locations=4)
        self.f=f
        self.f1=f1
        self.f2=f2
        self.ridge1 = nn.Conv2d(1, 5, kernel_size=25, bias=False)
        self.ridge2= nn.Conv2d(1, 5, kernel_size=25, bias=False)
        self.ridge3 = nn.Conv2d(1, 5, kernel_size=25, bias=False)
        self.ridge4 = nn.Conv2d(1, 5, kernel_size=25, bias=False)
        self.noise=nn.Conv2d(1,10,kernel_size=[50,15],bias=False)
        self.maxpool1=nn.MaxPool2d(5,stride=[5,25])
        self.maxpool2=nn.MaxPool2d(4,stride=[4,50])
        self.relu = nn.ReLU(inplace=True)
    def forward(self, images):
        coords, heatmaps, clean_output,_=self.main_model(images)
        # input_edges=self.relu(nn.functional.conv2d(images,self.f))
        # input_edges=torch.add(torch.add(input_edges[:,0:1,:,:],input_edges[:,1:2,:,:]),torch.add(input_edges[:,2:3,:,:],input_edges[:,3:4,:,:]))
        clean_edges=self.relu(nn.functional.conv2d(clean_output,self.f))
        clean_edges=torch.add(torch.add(clean_edges[:,0:1,:,:],clean_edges[:,1:2,:,:]),torch.add(clean_edges[:,2:3,:,:],clean_edges[:,3:4,:,:]))
        # mult=torch.mul(clean_edges,input_edges)
        out_noise=(torch.cat([self.noise(clean_edges),nn.functional.conv2d(clean_edges,self.f2)],dim=1))
        out_ridge1=(torch.cat([self.ridge1(clean_edges),nn.functional.conv2d(clean_edges,self.f1)],dim=1))
        out_ridge2 = (torch.cat([self.ridge2(clean_edges), nn.functional.conv2d(clean_edges, self.f1)], dim=1))
        out_ridge3 = (torch.cat([self.ridge3(clean_edges), nn.functional.conv2d(clean_edges, self.f1)], dim=1))
        out_ridge4 = (torch.cat([self.ridge4(clean_edges), nn.functional.conv2d(clean_edges, self.f1)], dim=1))
        out_ridge=torch.cat([out_ridge1,out_ridge2,out_ridge3,out_ridge4],dim=1)
        ridge=self.maxpool1(out_ridge)
        noise=self.maxpool2(out_noise)
        return coords, heatmaps, clean_output,ridge,noise



f=np.zeros([4,4,4])
f[0,:,:]=np.concatenate([-np.ones([4,2]),np.ones([4,2])],1)
f[1,:,:]=np.concatenate([np.ones([4,2]),-np.ones([4,2])],1)
f[2,:,:]=np.concatenate([-np.ones([2,4]),np.ones([2,4])],0)
f[3,:,:]=np.concatenate([np.ones([2,4]),-np.ones([2,4])],0)
f=torch.from_numpy(f).unsqueeze(1).cuda().float()
f1=np.zeros([10,25,25])
for i in range(10):
  f1[i,:,:]=generate_SCIRD(3.14,.1*(i+3),20,.1,25)
f1=torch.from_numpy(f1).unsqueeze(1).float()
f2=np.zeros([20,50,15])
for i in range(20):
    # index=np.arange(3)
    f2[i,:,7]=i*np.ones([1,50])
f2[5:15,:,:]=np.zeros([10,50,15])
for i in range(10):
    for m in range(15):
          f2[i+5,m,m]=i
          f2[i+5,m,14-m]=i
f2=torch.from_numpy(f2).unsqueeze(1).float()
# model = final(f,f1[5:10,:,:,:].cuda(),f2[10:,:,:,:].cuda()).cuda()
model = CoordRegressionNetwork(n_locations=4).cuda()
# model = model
optimizer = optim.RMSprop(model.parameters(), lr=0.0001)
#optimizer = optim.Adam(model.parameters(),lr=0.0002,betas=(0.9,0.999),eps=1e-8)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
load =True# If you want to load a pre-trained model
model_path= './model/singledec_onmixed/20.pth'
              # Path to pretrained model
# model.load_state_dict(torch.load(model_path))
# data=si.loadmat(''.join([data_path,'Codes_to_Share/experimental.mat']))['data'][4,:,:,:]
# data=h5py.File('./PA_Target_Detection_Dataset/Codes_to_Share/experimental/ours/20cm_new/_2.h5', 'r')['INPUT'][2502,:,:,:]
# data=torch.from_numpy(data)
# heats=np.zeros([2,4,128,512])
# cleans=np.zeros([2,1,128,1024])
# for i in range(2):
# #     # get the inputs
# inputs = data[i:i+1,:,:,:]
# # #     # load to GPU
# inputs = data.unsqueeze(0)
# _,heat,clean,_,_,=model(inputs.cuda().float())
# clean=clean.detach().cpu().numpy()
#     _, heat, _, _ = mode/l(inputs)
#     heats[i,:,:,:]=heat.detach().cpu().numpy()
# heat=heat
# heat=heat.detach().cpu().numpy()
# plt.imshow(np.sum(heats[2,:,:,:],0))
# plt.show()
if load == True:
    model.load_state_dict(torch.load(model_path))
    # model.ridge.weight.data = f1[0:5, :, :, :].cuda()
    # model.noise.weight.data = f2[0:10, :, :, :].cuda()
    # model.ridge1.weight.data = f1[0:5, :, :, :].cuda()
    # model.ridge2.weight.data = f1[0:5, :, :, :].cuda()
    # model.ridge3.weight.data = f1[0:5, :, :, :].cuda()
    # model.ridge4.weight.data = f1[0:5, :, :, :].cuda()
    # model.noise.weight.data = f2[0:10, :, :, :].cuda()
    # model.ridge1.weight.requires_grad = False
    # model.ridge2.weight.requires_grad = False
    # model.ridge3.weight.requires_grad = False
    # model.ridge4.weight.requires_grad = False
    # model.noise.weight.requires_grad = False
# else:
#     model.ridge1.weight.data = f1[0:5,:,:,:].cuda()
#     model.ridge2.weight.data = f1[0:5, :, :, :].cuda()
#     model.ridge3.weight.data = f1[0:5, :, :, :].cuda()
#     model.ridge4.weight.data = f1[0:5, :, :, :].cuda()
#     model.noise.weight.data = f2[0:10,:,:,:].cuda()
#     model.ridge1.weight.requires_grad=False
#     model.ridge2.weight.requires_grad = False
#     model.ridge3.weight.requires_grad = False
#     model.ridge4.weight.requires_grad = False
#     model.noise.weight.requires_grad=False

# with h5py.File(''.join(['./PA_Target_Detection_Dataset/', 'X_Train_20_Noisy_-9dB.mat']), 'r') as data:
#  X_Test = data['X_Train'][:,:,:,11:12]
#  X_Test=np.transpose(X_Test)
# data=si.loadmat(''.join([data_path,'meeting_stuff/sensor_data_needed20.mat']))['sensor_data_needed']
# data=np.transpose(data)
# data=torch.from_numpy(data)
# heats=np.zeros([16,4,128,512])
# for i in range(16):
# #     # get the inputs
# inputs, _, labels = data
# # #     # load to GPU
# inputs = data.cuda().unsqueeze(0).unsqueeze(0)
# _,heat,Clean,_=model(inputs.float())
# clean=Clean.detach().cpu().numpy()
    # _, heat, _, _ = model(inputs)
    # heats[i,:,:,:]=heat.detach().cpu().numpy()
# heat=heat
# heat=heat.detach().cpu().numpy()
# plt.show()
# plt.imshow(heat[4,0,:,:])
# import cv2 as cv
# cv.imshow('kir',heat[0,3,:,:]/np.max(heat[0,3,:,:]))
# cv.waitKey()
#os.mkdir('./model/theirs/')
# train = torch.utils.data.TensorDataset(X_Train,torch.cat((X_Clean,X_Clean,X_Clean),0), Y_Train)
# trainloader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=False)
# for i, data in enumerate(trainloader, 0):
#     _,_,input = data
#     input=input.cuda()
#     temp = model(input)
#     temp = temp[2].detach().cpu()
#     X_clean[bs*i:bs*(i+1), :, :, :] = temp
#     print(i)
# traiheats[6,3,:,:]n = torch.utils.data.TensorDataset(X_Train, Y_Train,X_clean)
# trainloader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=False)
# for param in pretrained_model.parameters():
#     param.requires_grad=False
# sio.savemat(os.path.join('./model/theirs_denoise_nonyq/',(str(1)+'9db_d.mat')), {'cords': denoised[0, 0, :, :].cpu().data.numpy()})
#-----------------------------------------------------------------------------
#          -Training-
#-----------------------------------------------------------------------------
# plt.imshow(heats[6,0,:,:]+heats[6,1,:,:]+heats[6,2,:,:]+heats[6,3,:,:])
Train = False
criterion = nn.MSELoss()
if Train:
    # si.savemat('./model/low_training_tuffc/perm.mat', {'perm': perm.numpy()})
    train_size = 24000   # test_size = X_Test.size(0)
    n = 60#number of epochs
    Train_loss_hist = torch.Tensor(n+1).numpy()
    # Test_loss_hist = torch.Tensor(n+1).numpy()
    Train_acc_hist = torch.Tensor(n+1).numpy()
    # Test_acc_hist = torch.Tensor(n+1).numpy()
    count=0
    for epoch in range(n+1):
        model.train()
        # print('Epoch: %d' % epoch)
        running_train_loss = 0.0
        # running_test_loss = 0.0
        train_correct = 0 #running count of correct predictions on training set
        # test_correct = 0 #running count of correct predictions on test set

        total = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs,clean,labels= data
            # load to GPU
            inputs = inputs.cuda()
            labels = labels.cuda()
            clean=clean.cuda()
            # wrap them in Variable
            inputs,labels,clean= Variable(inputs),Variable(labels),Variable(clean)
            #forward pass
            coords, heatmaps,_,_= model(inputs.float())
            # old_coords,old_heatmaps,_=pretrained_model(clean_outputs)
            # loss_sam=0
            # for j in range(45):
            # sam_out=model(sam_inp,sample=True)
            # loss_sam=criterion(sam_out.reshape(90,64,64),sam_tar.float())
            #Per-location euclidean losses
            euc_losses = dsntnn.euclidean_losses(coords, labels)
            # noise_loss=criterion(clean_outputs,clean.float())
            #Per-location regularization losses

            reg_losses = dsntnn.js_reg_losses(heatmaps,labels,sigma_t=1)
            # Coord_loss= dsntnn.average_loss(euc_losses+reg_losses)
            # loss_sam=loss_sam
            #Combine losses into an overall loss

            # ridge_tar=torch.zeros([3,4,ridge.shape[2],ridge.shape[3]])
            # nplabels=torch.zeros([3,4,2])
            # for l in range(3):
            #     for m in range(3):
            #         nplabels[l, m, 0] = labels[l, m, 0] * ((max_ax - min_ax) / 2 / 1) + (
            #                 (max_ax - min_ax) / 2 + min_ax)
            #         nplabels[l, m, 1] = labels[l, m, 1] * ((40) / 2 / 1)
            #         if labels[l,m,0]==0 and labels[l,m,1]==0:
            #             ridge_tar[l, m, :, :] =ridge_tar[l,m-1,:,:]
            #         else:
            #             ridge_tar[l,m,:,:]=clean[l,0,int(128*(nplabels[l,0,1]+27.5)/55)-(ridge.shape[2])//2:int(128*(nplabels[l,0,1]+27.5)/55)+(ridge.shape[2])//2,max(int(1024*(nplabels[l,0,0])/55)-(ridge.shape[3])//2-40,0):max(int(1024*(nplabels[l,0,0])/55)+(ridge.shape[3])//2-40,45)]
            #
            # # if count%5==0:
            # #     loss=-.000001 * criterion(ridge, torch.mul(ridge_tar,torch.ones(ridge.shape)).cuda()) + .0001 * criterion(noise, torch.zeros(noise.shape).cuda())
            # # else:
            # ridge_pattern = torch.cat([torch.mul(ridge_tar[:, 0:1, :, :], torch.ones(3, 10, 20, 40)),
            #                            torch.mul(ridge_tar[:, 1:2, :, :], torch.ones(3, 10, 20, 40)),
            #                            torch.mul(ridge_tar[:, 2:3, :, :], torch.ones(3, 10, 20, 40)),
            #                            torch.mul(ridge_tar[:, 3:4, :, :], torch.ones(3, 10, 20, 40))], dim=1)
            loss = dsntnn.average_loss(euc_losses+reg_losses)#+.5*noise_loss#+10**-4*criterion(ridge, ridge_pattern.cuda()) + 10**-9* criterion(noise, torch.zeros(noise.shape).cuda())#-10^-8*criterion(ridge,torch.zeros(ridge.shape).cuda())+.0001*criterion(noise,torch.zeros(noise.shape).cuda())
            # else:
            # loss=dsntnn.average_loss(euc_losses+reg_losses)+.5/3*noise_loss+0.5/3*dsntnn.average_loss(dsntnn.euclidean_losses(old_coords, labels)+dsntnn.js_reg_losses(old_heatmaps,labels,sigma_t=1.0))-.0001*criterion(ridge,torch.zeros(ridge.shape).cuda())+.0001*criterion(noise,torch.zeros(noise.shape).cuda())#+0.5*loss_sam
            # #Calculate gradients
            # count+=1
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5)
            loss.backward()
            #Update model parameters
            optimizer.step()
            # accrue loss
            running_train_loss += loss.item()
            # print('Epoch: {},loss:{:.10f}' .format(epoch,loss))
            # for data in testloader:
            #     testin, testlabels = data
            #     # Load to GPU
            #     testin = testin.cuda()
            #     testlabels = testlabels.cuda()
            #     # Wrap in Variable
            #     testin = Variable(testin)
            #     testlabels = Variable(testlabels)
            #     # Evaluate
            #     model.eval()
            #     coords, heatmaps, xhist = model(testin)
            #     euc_losses = dsntnn.euclidean_losses(coords, testlabels)
            #     reg_losses = dsntnn.js_reg_losses(heatmaps,testlabels, sigma_t=1.0)
            #     loss = dsntnn.average_loss(euc_losses+reg_losses)
            #     running_test_loss += loss.item()

            
        
        Train_loss_hist[epoch] = running_train_loss
        # Test_loss_hist[epoch] = running_test_loss
        # scheduler.step(running_test_loss)
        print('loss: %.10f' % running_train_loss)
        # print('noise loss:%.10f'% noise_loss)
        # print('coord_loss:%.10f'% Coord_loss)
        if (epoch%5==0 ):            # Saves model every epoch after 50 in case of over-training
            dir_name = './model/singledec_on10cm/'
            torch.save(model.state_dict(), dir_name+str(epoch+95)+'.pth')
            #savemat(dir_name+'/Train_loss_hist',mdict = {'Train_loss_hist':Train_loss_hist})
            #savemat(dir_name+'/Test_loss_hist_',mdict = {'Test_loss_hist':Test_loss_hist})

    

print('Finished Training')

#-----------------------------------------------------------------------------
#                   -Evaluation-
#-----------------------------------------------------------------------------


# if not load:
#     print('\n\n Training Loss vs Epoch')
#     plt.figure(2)
#     plt.plot(Train_loss_hist)
#     plt.ylabel('Training Loss')
#     plt.show()
    # print('\n\n Test Loss vs Epoch')
    # plt.figure(3)
    # plt.plot(Test_loss_hist)
    # plt.ylabel('Test Loss')
    # plt.show()


#-----------------------------------------------------------------------------
#                   -Visualization-
#-----------------------------------------------------------------------------
# for param_group in optimizer.param_groups:
#     print(param_group['lr'])
    
batch = 0
errors = torch.Tensor(6000,12,2)
heat = torch.Tensor(6000,4,256,512)
nplabels = torch.Tensor(bs,4,2)
npcoords = torch.Tensor(bs,4,2)
bw = bs #batch width to decrease at last batch

# si.savemat('./PA_Target_Detection_Dataset/meeting_stuff/heat_sim2_10cm.mat',{'heat':heatmaps[1,:,:,:]})

for data in testloader:
    images, labels = data    
    coords,heatmaps,cleans,_= model(images.cuda().float()) #imagehist is a dictionary with keys 'x0','x1','x2','x3'
    # coords,heatmaps,_=pretrained_model(denoised)
    if batch*bs+bw >= errors.shape[0]:
        # cv.imwrite(os.path.join('./model/theirs_denoise_nonyq_12db/', str(str(1) + ".png")),
        #            denoised[0, 0, :, :].cpu().data.numpy())
        # cleans = cleans.detach().cpu().numpy()
        bw = errors.shape[0]-batch*bs
        nplabels = torch.Tensor(bw,4,2)
        npcoords = torch.Tensor(bw,4,2)
    
    for i in range(bw):
        print(1+i+batch*bs)
        nplabels[i,:,0] = labels[i,:,0]*((max_ax-min_ax)/2/max_normalized_depth)+((max_ax-min_ax)/2+min_ax)
        nplabels[i,:,1] = labels[i,:,1]*((max_ax-min_ax)/2/max_normalized_depth)
        npcoords[i,:,0] = coords.detach().cpu()[i,:,0]*((max_ax-min_ax)/2/max_normalized_depth)+((max_ax-min_ax)/2+min_ax)
        npcoords[i,:,1] = coords.detach().cpu()[i,:,1]*((max_ax-min_ax)/2/max_normalized_depth)
        print(nplabels[i])
        print(npcoords[i])
        print(bw)
            
    heat[batch*bs:batch*bs+bw,:,:,:] = heatmaps.detach().cpu()
    errors[batch*bs:batch*bs+bw,0:4,:] = nplabels
    errors[batch*bs:batch*bs+bw,4:8,:] = npcoords
    errors[batch*bs:batch*bs+bw,8:12,:] = abs(npcoords-nplabels)
    batch += 1
axlow=np.mean(errors[:,8:12,:].numpy()[np.where(errors[:,0:4,0]<35),0])
latlow=np.mean(errors[:,8:12,:].numpy()[np.where(errors[:,0:4,0]<35),1])
axhigh=np.mean(errors[:,8:12,:].numpy()[np.where(errors[:,0:4,0]>=35),0])
lathigh=np.mean(errors[:,8:12,:].numpy()[np.where(errors[:,0:4,0]>=35),1])
euclow=np.mean(np.sqrt(np.power(errors[:,8:12,:].numpy()[np.where(errors[:,0:4,0]<35),0],2)+np.power(errors[:,8:12,:].numpy()[np.where(errors[:,0:4,0]<35),1],2)))
euchigh=np.mean(np.sqrt(np.power(errors[:,8:12,:].numpy()[np.where(errors[:,0:4,0]>=35),0],2)+np.power(errors[:,8:12,:].numpy()[np.where(errors[:,0:4,0]>=35),1],2)))
# plt.plot(Test_loss_hist)
latmean = np.mean(errors[:,8:12,1].numpy())
latstd = np.std(errors[:,8:12,1].numpy())
axmean = np.mean(errors[:,8:12,0].numpy())
axstd = np.std(errors[:,8:12,0].numpy())
eucmean = np.mean(np.sqrt(np.power(errors[:,8:12,0].numpy(),2)+np.power(errors[:,8:12,1].numpy(),2)))
eucstd = np.std(np.sqrt(np.power(errors[:,8:12,0].numpy(),2)+np.power(errors[:,8:12,1].numpy(),2)))

print('\n Mean lateral error: %f' % latmean)
print('\n STD of lateral error: %f' %latstd)
print('\n Mean axial error: %f' % axmean)
print('\n STD of axial error: %f' %axstd)
print('\n Mean euclidian error: %f' % eucmean)
print('\n STD of euclidian error: %f' %eucstd)
print('\n ax low: %f' % axlow)
print('\n lat low: %f' % latlow)
print('\n euc low: %f' % euclow)
print('\n ax high: %f' % axhigh)
print('\n lat high: %f' % lathigh)
print('\n euc high: %f' % euchigh)
print('kir')