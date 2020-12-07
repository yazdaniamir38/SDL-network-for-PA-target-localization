import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import dsntnn


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
            residual = x[:, 0:self.planes, :, :]
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
class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, present, former):
        present = self.up(present)
        present = self.conv(present)
        x = torch.cat([present, former], dim=1)
        return x


class ResNet(nn.Module):

    def __init__(self, block, num_classes=9):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(7, 7), stride=(1, 1), dilation=(1, 4), padding=(3, 12), bias=False)
        # self.convs=nn.Conv2d(1,1,kernel_size=7,stride=(4,16),padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pool1d = nn.MaxPool2d(kernel_size=(1, 2))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, 16, blocks=1)
        self.layer2 = self._make_layer(block, 16, 32, blocks=1, stride=2)
        self.layer3 = self._make_layer(block, 32, 64, blocks=1, stride=2)
        self.layer4 = self._make_layer(block, 64, 128, blocks=1, stride=2)
        self.layer5 = self._make_layer(block, 128, 256, blocks=1, stride=2)
        self.classify = nn.Conv2d(256, 256, kernel_size=(5, 5), padding=(2, 2))
        self.bn2 = nn.BatchNorm2d(256)
        self.layer6 = self._make_layer(block, 256, 256, blocks=1, stride=2)  # subsample w/o adding features
        self.classify2 = nn.Conv2d(256, 256, kernel_size=(5, 5), padding=(2, 2))  # 8x8
        self.bn3 = nn.BatchNorm2d(256)
        self.layer7 = self._make_layer(block, 256, 256, blocks=1, stride=1)
        self.layer8 = self._make_layer(block, 256, 256, blocks=1, stride=1)
        self.up1 = up(256, 256)
        self.up6 = up(256, 256)
        self.layer9 = self._make_layer(block, 512, 256, blocks=1, stride=1, expanding=True)
        self.dec9 = self._make_layer(block, 512, 256, blocks=1, stride=1, expanding=True)
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
        self.up11 = nn.UpsamplingNearest2d(scale_factor=2)
        self.dec14 = self._make_layer(block, 16, 8, blocks=1, stride=1, expanding=True)
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
            plt.imshow(x[0, 0, :, :].detach().cpu().numpy())  # first in the batch
            plt.show()

    def forward(self, x, remember=False, sample=False):
        # xs = self.convs(x)

        x1 = self.conv1(x)
        xhist = {'x1': x}
        x2 = self.bn1(x1)
        xhist['x2'] = x2
        x3 = self.relu(x2)
        xhist['x3'] = x3
        x4 = self.pool1d(x3)  # 256x256x16  <-outputs
        xhist['x4'] = x4
        x5 = self.layer1(x4)  # 256x256x16
        xhist['x5'] = x5
        x6 = self.layer2(x5)  # 128x128x32
        xhist['x6'] = x6
        x7 = self.layer3(x6)  # 64x64x64
        xhist['x7'] = x7

        if sample:
            return (x7[:, 0, :, :])
        else:
            x8 = self.layer4(x7)  # 32x32x128
            xhist['x8'] = x8
            x9 = self.layer5(x8)  # 16x16x256
            xhist['x9'] = x9
            x10 = self.classify(x9)  # 16x16x256
            xhist['x10'] = x10
            x11 = self.bn2(x10)  # BN
            xhist['x11'] = x11
            x12 = self.relu(x11)  # ReLU
            xhist['x12'] = x12
            x13 = self.layer6(x12)  # 8x8x256
            xhist['x13'] = x13
            x14 = self.classify2(x13)  # 8x8x256
            xhist['x14'] = x14
            x15 = self.bn3(x14)  # BN
            xhist['x15'] = x15
            x16 = self.relu(x15)  # ReLU
            xhist['x16'] = x16
            x17 = self.layer7(x16)  # 8x8x256
            xhist['x17'] = x17
            x18 = self.layer8(x17)  # 8x8x256
            xhist['x18'] = x18
            x19 = self.up1(x18, x10)  # 16x16x512
            xhist['x19'] = x19
            x20 = self.layer9(x19)  # 16x16x256
            xhist['x20'] = x20

            xdec19 = self.up6(x18, x10)  # 16x16x512
            # xhist['x19'] = x19
            xdec20 = self.dec9(xdec19)  # 16x16x256

            x21 = self.up2(x20, x8)  # 32x32x256
            xhist['x21'] = x21
            x22 = self.layer10(x21)  # 32x32x128
            xhist['x22'] = x22

            xdec21 = self.up7(xdec20, x8)  # 32x32x256
            # xhist['x21'] = x21
            xdec22 = self.dec10(xdec21)  # 32x32x128

            x23 = self.up3(x22, x7)  # 64x64x128
            xhist['x23'] = x23
            x24 = self.layer11(x23)  # 64x64x64
            xhist['x24'] = x24

            xdec23 = self.up8(xdec22, x7)  # 64x64x128
            # xhist['x23'] = x23
            xdec24 = self.dec11(xdec23)  # 64x64x64
            # xhist['x24'] = x24

            x25 = self.up4(x24, x6)  # 128x128x64
            xhist['x25'] = x25
            x26 = self.layer12(x25)  # 128x128x32
            xhist['x26'] = x26

            xdec25 = self.up9(xdec24, x6)  # 128x128x64
            # xhist['x25'] = x25
            xdec26 = self.dec12(xdec25)  # 128x128x32

            x27 = self.up5(x26, x5)  # 256x256x32
            xhist['x27'] = x27
            x28 = self.layer13(x27)  # 256x256x16

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
    def __init__(self, n_locations):
        super(CoordRegressionNetwork, self).__init__()
        self.fcn = resnet18()
        self.hm_conv = nn.Conv2d(16, n_locations, kernel_size=1, bias=False)
        self.hm_conv2 = nn.Conv2d(8, 1, kernel_size=1, stride=(2, 1), bias=False)
        # self.ridge=nn.Conv2d(1,5,kernel_size=25,bias=False)
        # self.noise=nn.Conv2d(1,10,kernel_size=[50,15],bias=False)
        # self.maxpool1=nn.MaxPool2d(5,stride=[10,50])
        # self.maxpool2=nn.MaxPool2d(4,stride=[4,50])
        self.relu = nn.ReLU(inplace=True)
        # self.f=f
        # self.f1=f1
        # self.f2=f2
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, images, sample=False):
        if sample:
            xs = self.fcn(images, sample=sample)
            return xs
        else:
            fcn_out, dec_out, xhist = self.fcn(images)
            unnormalized_heatmaps = self.hm_conv(fcn_out)
            clean_output = self.hm_conv2(dec_out)
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
            return coords, heatmaps, clean_output, xhist


class final(nn.Module):
    def __init__(self, f, f1, f2):
        super(final, self).__init__()
        self.main_model = CoordRegressionNetwork(n_locations=4)
        self.edges = f
        self.ridge = f1
        self.noise = f2
        self.maxpool1 = nn.MaxPool2d(5, stride=[5, 25])
        self.maxpool2 = nn.MaxPool2d(4, stride=[4, 50])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, images):
        coords, heatmaps, clean_output, _ = self.main_model(images)
        clean_edges = self.relu(nn.functional.conv2d(clean_output, self.f))
        clean_edges = torch.add(torch.add(clean_edges[:, 0:1, :, :], clean_edges[:, 1:2, :, :]),
                                torch.add(clean_edges[:, 2:3, :, :], clean_edges[:, 3:4, :, :]))
        out_noise = nn.functional.conv2d(clean_edges, self.noise)], dim = 1)
        out_ridge1 = nn.functional.conv2d(clean_edges, self.ridge)], dim = 1)
        out_ridge2 = nn.functional.conv2d(clean_edges, self.ridge)], dim = 1)
        out_ridge3 = nn.functional.conv2d(clean_edges, self.ridge)], dim = 1)
        out_ridge4 = nn.functional.conv2d(clean_edges, self.ridge)], dim = 1)
        out_ridge = torch.cat([out_ridge1, out_ridge2, out_ridge3, out_ridge4], dim=1)
        ridge = self.maxpool1(out_ridge)
        noise = self.maxpool2(out_noise)
        return coords, heatmaps, ridge, noise
