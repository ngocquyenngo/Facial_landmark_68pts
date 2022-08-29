import torch
import torch.nn as nn
from soft_argmax import SoftArgmax
from ResBlock import ResBlock as resblock
NUM_OF_CLASSESS = int(68+1)

class FacialLandmark(nn.Module):
    def __init__(self):
        super(FacialLandmark,self).__init__()
        # ResNet-18
        #input 224x224x3 -> 112x112x64
        self.layer0 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU()
        )
        #112x112x64 -> 56x56x64
        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )
        #56x56x64 -> 28x28x128
        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )
        #28x28x128 -> 14x14x256
        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )
        #14x14x256 -> 7x7x512
        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )
        #7x7x512 -> 7x7x69
        self.layer5 = nn.Conv2d(512, NUM_OF_CLASSESS, kernel_size=(1,1), stride=1, padding=(0,0))
        # Upsampling
        #7x7x69 -> 14x14x256
        self.deconv1 = nn.ConvTranspose2d(NUM_OF_CLASSESS, 256, kernel_size=(4,4), stride=2, padding=1)
        #14x14x256 -> 28x28x128
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=(4,4), stride=2, padding=1)
        #28x28x128 -> 224x224x69
        self.fuse3 = nn.ConvTranspose2d(128, NUM_OF_CLASSESS, kernel_size=(16,16), stride=8, padding=4)

    def forward(self,x):  
        x_l0 = self.layer0(x)
        x_l1 = self.layer1(x_l0)
        x_l2 = self.layer2(x_l1)
        x_l3 = self.layer3(x_l2)
        x_l4 = self.layer4(x_l3)
        x_l5 = self.layer5(x_l4)

        x_dc1 = self.deconv1(x_l5)
        fuse1 = torch.add(x_dc1,x_l3)
        x_dc2 = self.deconv2(fuse1)
        fuse2 = torch.add(x_dc2,x_l2)
        heatmap = self.fuse3(fuse2)
        return heatmap
