import torch
import torch.nn as nn
from ResBottleneckBlock import ResBottleneckBlock as resblock
from soft_argmax import SoftArgmax
NUM_OF_CLASSESS = int(68+1)

class FacialLandmark(nn.Module):
    def __init__(self):
        super(FacialLandmark,self).__init__()
        # ResNet-50
        #input 224x224x1 -> 112x112x64 -> 56x56x64
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        #56x56x64 -> 56x56x256
        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv2_1', resblock(64, 256, downsample=False))
        for i in range(1, 3):
                self.layer1.add_module('conv2_%d'%(i+1,), resblock(256, 256, downsample=False))
        #56x56x256 -> 28x28x512
        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv3_1', resblock(256, 512, downsample=True))
        for i in range(1, 4):
                self.layer2.add_module('conv3_%d' % (i+1,), resblock(512, 512, downsample=False))
        #28x28x512 -> 14x14x1024 
        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv4_1', resblock(512, 1024, downsample=True))
        for i in range(1, 6):
            self.layer3.add_module('conv2_%d' % (i+1,), resblock(1024, 1024, downsample=False))
        #14x14x1024 -> 7x7x2048
        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv5_1', resblock(1024, 2048, downsample=True))
        for i in range(1, 3):
            self.layer4.add_module('conv3_%d'%(i+1,), resblock(2048, 2048, downsample=False))
        #7x7x2048 -> 7x7x512
        self.layer5 = nn.Conv2d(2048, 512, kernel_size=(1,1), stride=1, padding=(0,0))
        #7x7x512 -> 7x7x69
        self.layer6 = nn.Conv2d(512, NUM_OF_CLASSESS, kernel_size=(1,1), stride=1, padding=(0,0))
        #upsampling
        #7x7x69 -> 14x14x1024
        self.deconv1 = nn.ConvTranspose2d(NUM_OF_CLASSESS, 1024, kernel_size = (4,4), stride = 2, padding =1)
        #14x14x1024 -> 28x28x512 
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size = (4,4), stride = 2, padding =1)
        #28x28x512 -> 56x56x256
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size = (4,4), stride = 2, padding =1)
        #56x56x256 -> 224x224x69
        self.fuse4 = nn.ConvTranspose2d(256, NUM_OF_CLASSESS, kernel_size = (8,8), stride = 4, padding =2)
    def forward(self,x):  
        x_l0 = self.layer0(x)
        x_l1 = self.layer1(x_l0)
        x_l2 = self.layer2(x_l1)
        x_l3 = self.layer3(x_l2)
        x_l4 = self.layer4(x_l3)
        x_l5 = self.layer5(x_l4)
        x_l6 = self.layer6(x_l5)
        x_dc1 = self.deconv1(x_l6)
        fuse1 = torch.add(x_dc1, x_l3)
        x_dc2 = self.deconv2(fuse1)
        fuse2 = torch.add(x_dc2, x_l2)
        x_dc3 = self.deconv3(fuse2)
        fuse3 = torch.add(x_dc3, x_l1)
        heatmap = self.fuse4(fuse3)
        
        softargmax = SoftArgmax(heatmap.size(2), heatmap.size(3),heatmap.size(1))
        all_pts = softargmax(heatmap)
        back_ground, pred_pts = torch.split(all_pts, [1,68], dim=1)
        pred_pts = pred_pts.view(pred_pts.size(0), -1)
        return pred_pts  