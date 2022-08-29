import torch
import torch.nn as nn
from ResBottleneckBlock import ResBottleneckBlock as resblock
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
        #batch_size x7x7x512 -> flatten batch_size x25088 -> batch_size x136
        self.fc = nn.Linear(in_features=25088, out_features=136)

    def forward(self,x):  
        x_l0 = self.layer0(x)
        x_l1 = self.layer1(x_l0)
        x_l2 = self.layer2(x_l1)
        x_l3 = self.layer3(x_l2)
        x_l4 = self.layer4(x_l3)
        x_l5 = self.layer5(x_l4)
        x_l5 = x_l5.view(x_l5.size(0),-1)
        out = self.fc(x_l5)
        return out
