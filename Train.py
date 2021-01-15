


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
#data path should include the h5files for the dataset with 3 dictionares:'INPUT','TARGET','CLEAN'(optional for single mode)
train_files = './train';
bs =4# batch size
# train = torch.utils.data.TensorDataset(X_Train, Y_Train,X_Clean)
# trainloader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=False)

#if you don't use clean samples in the network set yes=0
train_set = DatasetFromHdf5(os.walk(train_files),yes=1)
trainloader =torch.utils.data.DataLoader(dataset=train_set, num_workers=1, batch_size=3, shuffle=True)
#The arhcitecture:
mode='SDL'
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
            return coords, heatmaps, clean_output,xhist

class final(nn.Module):
    def __init__(self,f,f1,f2):
        super(final, self).__init__()
        self.main_model=CoordRegressionNetwork(n_locations=4)
        self.edges=f
        self.ridge= f1
        self.noise=f2
        self.maxpool1=nn.MaxPool2d(5,stride=[5,25])
        self.maxpool2=nn.MaxPool2d(4,stride=[4,50])
        self.relu = nn.ReLU(inplace=True)
    def forward(self, images):
        coords, heatmaps, clean_output,_=self.main_model(images)
        clean_edges=self.relu(nn.functional.conv2d(clean_output,self.f))
        clean_edges=torch.add(torch.add(clean_edges[:,0:1,:,:],clean_edges[:,1:2,:,:]),torch.add(clean_edges[:,2:3,:,:],clean_edges[:,3:4,:,:]))
        out_noise=nn.functional.conv2d(clean_edges,self.noise)],dim=1)
        out_ridge1=nn.functional.conv2d(clean_edges,self.ridges)],dim=1)
        out_ridge2 = nn.functional.conv2d(clean_edges,self.ridges)],dim=1)
        out_ridge3 =nn.functional.conv2d(clean_edges,self.ridges)],dim=1)
        out_ridge4 = nn.functional.conv2d(clean_edges,self.ridges)],dim=1)
        out_ridge=torch.cat([out_ridge1,out_ridge2,out_ridge3,out_ridge4],dim=1)
        ridge=self.maxpool1(out_ridge)
        noise=self.maxpool2(out_noise)
        return coords, heatmaps,clean_output,ridge,noise


#Generating the edge detection filters
e=np.zeros([4,4,4])
e[0,:,:]=np.concatenate([-np.ones([4,2]),np.ones([4,2])],1)
e[1,:,:]=np.concatenate([np.ones([4,2]),-np.ones([4,2])],1)
e[2,:,:]=np.concatenate([-np.ones([2,4]),np.ones([2,4])],0)
e[3,:,:]=np.concatenate([np.ones([2,4]),-np.ones([2,4])],0)
e=torch.from_numpy(e).unsqueeze(1).cuda().float()
#Generating the wavefront filters
W=np.zeros([10,25,25])
for i in range(10):
  W[i,:,:]=generate_SCIRD(3.14,.1*(i+3),20,.1,25)
W=torch.from_numpy(W).unsqueeze(1).float()
#generating the noise filters
N=np.zeros([20,50,15])
for i in range(20):
    N[i,:,7]=i*np.ones([1,50])
N[5:15,:,:]=np.zeros([10,50,15])
for i in range(10):
    for m in range(15):
          N[i+5,m,m]=i
          N[i+5,m,14-m]=i
N=torch.from_numpy(N).unsqueeze(1).float()
if mode=='SDL':
  model = final(e,W.cuda(),N.cuda()).cuda()
else:
  model = CoordRegressionNetwork(n_locations=4).cuda()

optimizer = optim.RMSprop(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
#If you want to load a pre-trained model
load =True
# Path to pretrained model
model_path= './checkpoints/SDL_mixed.pth'
# single-decoder:single_mixed.pth
# w/o filters:without_mixed.pth
if load == True:
    model.load_state_dict(torch.load(model_path))
Train = False
criterion = nn.MSELoss()
if Train:
    #Number of epochs
    n = 120
    Train_loss_hist = torch.Tensor(n+1).numpy()
    Train_acc_hist = torch.Tensor(n+1).numpy()
        count=0
    for epoch in range(n+1):
        model.train()
        print('Epoch: %d' % epoch)
        running_train_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs(Change them acordingly if there is no clean gt in your dataset)'
            inputs,clean,labels= data
            # load to GPU
            inputs = inputs.cuda()
            labels = labels.cuda()
            clean=clean.cuda()
            # wrap them in Variable
            inputs,labels,clean= Variable(inputs),Variable(labels),Variable(clean)
            #forward pass
            if mode=='single':
               coords, heatmaps,_,_= model(inputs.float())
            else:
               if mode==='WN-less':
                 coords, heatmaps,clean_output,_= model(inputs.float())
               else:
                 coords, heatmaps,clean_output,ridge,noise= model(inputs.float())
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
                  
                  
                    loss = dsntnn.average_loss(euc_losses+reg_losses)+.5*noise_loss+10**-4*(criterion(ridge, ridge_pattern.cuda()) + criterion(noise, torch.zeros(noise.shape).cuda()))
           
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
            dir_name = './model/'
            torch.save(model.state_dict(), dir_name+str(epoch)+'.pth')
            

    

print('Finished Training')

