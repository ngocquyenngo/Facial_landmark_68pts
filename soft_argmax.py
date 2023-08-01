import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftArgmax(nn.Module):
    def __init__(self, height, width, channel):
        super(SoftArgmax, self).__init__()
        self.height = height
        self.width = width
        self.channel = channel
        
        pos_x, pos_y= np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width),
                )
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)
        
        
    def forward(self, input):
        # input:  (N, C, H, W)
        # output: (N, C, 2)
        input = input.view(-1, self.height*self.width)
        softmax_attention = F.softmax(input, dim=1)

        self.pos_x = self.pos_x.to(input.device)
        self.pos_y = self.pos_y.to(input.device)
        softmax_attention = softmax_attention.to(input.device)
        
        expected_x = torch.sum(self.pos_x*softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y*softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        coordinates = expected_xy.view(-1, self.channel, 2)
        return coordinates
