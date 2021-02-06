


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as si
import dsntnn
import h5py
import os
import Network
import utils
from Network import CoordRegressionNetwork
from Network import final
from dataset import DatasetFromHdf5
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
#data path should include the h5files for the dataset with 3 dictionares:'INPUT','TARGET','CLEAN'(optional for single mode)
train_files = './SDL_Dir/Train_data';
bs =3# batch size
# train = torch.utils.data.TensorDataset(X_Train, Y_Train,X_Clean)
# trainloader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=False)

#if you don't use clean samples in the network set yes=0
train_set = DatasetFromHdf5(os.walk(train_files),yes=1)
trainloader =torch.utils.data.DataLoader(dataset=train_set, num_workers=1, batch_size=bs, shuffle=True)
#The arhcitecture:
mode='SDL'
if mode='SDL':
  e,W,N=utils.generate_filters()
  model = final(e,W[5:10,:,:,:],N[10:20,:,:,:])
else:
  model = CoordRegressionNetwork(n_locations=4)

optimizer = optim.RMSprop(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
#If you want to load a pre-trained model
load =True
# Path to pretrained model
model_path= './SDL_Dir/checkpoints/SDL_mixed.pth'
# single-decoder:single_mixed.pth
# w/o filters:without_mixed.pth
if load == True:
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
criterion = nn.MSELoss()

#Number of epochs
n = 120
Train_loss_hist = torch.Tensor(n+1).numpy()
Train_acc_hist = torch.Tensor(n+1).numpy()
for epoch in range(n+1):
    model.train()
    print('Epoch: %d' % epoch)
    running_train_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs(Change them acordingly if there is no clean gt in your dataset)'
        inputs,clean,labels= data
        # load to GPU
        # inputs = inputs.cuda()
        # labels = labels.cuda()
        # clean=clean.cuda()
        # wrap them in Variable
        inputs,labels,clean= Variable(inputs),Variable(labels),Variable(clean)
        #forward pass
        if mode=='single':
           coords, heatmaps,_,_= model(inputs.float())
        else:
           if mode=='WN-less':
             coords, heatmaps,clean_outputs,_= model(inputs.float())
           else:
             coords, heatmaps,clean_outputs,ridge,noise= model(inputs.float())
        #Per-location euclidean losses
        euc_losses = dsntnn.euclidean_losses(coords, labels)
        #Per-location regularization losses
        reg_losses = dsntnn.js_reg_losses(heatmaps,labels,sigma_t=1)
        if mode=='single':
            loss = dsntnn.average_loss(euc_losses+reg_losses)
        else:
            #fidelity loss
            fid_loss=criterion(clean_outputs,clean.float())
            if mode=='WN-less':
               loss = dsntnn.average_loss(euc_losses+reg_losses)+.5*fid_loss
            else:
               ridge_tar=torch.zeros([3,4,ridge.shape[2],ridge.shape[3]])
               nplabels=torch.zeros([3,4,2])
                #extracting ridge_clean
               for l in range(3):
                  for m in range(4):
                     nplabels[l, m, 0] = labels[l, m, 0] * ((40) / 2 / .9) + (
                        (40) / 2 + 10)
                     nplabels[l, m, 1] = labels[l, m, 1] * ((40) / 2 / .9)
                     if labels[l,m,0]==0 and labels[l,m,1]==0:
                       ridge_tar[l, m, :, :] =ridge_tar[l,m-1,:,:]
                     else:
                       ridge_tar[l,m,:,:]=clean[l,0,int(256*(nplabels[l,m,1]+25.85)/51.7)-(ridge.shape[2])//2:int(256*(nplabels[l,m,1]+25.85)/51.7)+(ridge.shape[2])//2+1,int(1024*(nplabels[l,m,0])/61)-(ridge.shape[3])//2:int(1024*(nplabels[l,m,0])/61)+(ridge.shape[3])//2]


                 #the gt ridge used for regularization terms
               ridge_pattern = torch.cat([torch.mul(ridge_tar[:, 0:1, :, :], torch.ones(3, 10,ridge.shape[2],ridge.shape[3])),
                                   torch.mul(ridge_tar[:, 1:2, :, :], torch.ones(3, 10, ridge.shape[2],ridge.shape[3])),
                                   torch.mul(ridge_tar[:, 2:3, :, :], torch.ones(3, 10, ridge.shape[2],ridge.shape[3])),
                                   torch.mul(ridge_tar[:, 3:4, :, :], torch.ones(3, 10, ridge.shape[2],ridge.shape[3]))], dim=1)


               loss = dsntnn.average_loss(euc_losses+reg_losses)+.5*fid_loss+10**-4*(criterion(ridge, ridge_pattern) + criterion(noise, torch.zeros(noise.shape)))

        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5)
        loss.backward()
        #Update model parameters
        optimizer.step()
        # accrue loss
        running_train_loss += loss.item()
    Train_loss_hist[epoch] = running_train_loss
    print('loss: %.10f' % running_train_loss)
    if (epoch%5==0 ):            # Saves model every 5 epochs
        dir_name = './SDL_Dir/checkpoints/'
        torch.save(model.state_dict(), dir_name+str(epoch)+'.pth')
            

    

print('Finished Training')

