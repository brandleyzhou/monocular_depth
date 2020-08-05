from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
#import cv2
from collections import OrderedDict
from layers import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, nonlin='sigm',using_v=0):
        #  num_ch_enc = np.array([64, 64, 128, 256, 512])
        super(DepthDecoder, self).__init__()
        self.using_v=using_v
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1): #i=[4,3,2,1,0]
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)#CONV2D

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
        
        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        nonlin_choices = {'sigm':nn.Sigmoid(),'relu':nn.ReLU()}
        self.nonlin = nonlin_choices[nonlin]

        #down _sampling
        self.maxpool = nn.MaxPool2d(kernel_size = 3,stride = 2,padding = 1)
    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):#[4,3,2,1,0]
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]#this function in layers.py
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.nonlin(self.convs[("dispconv", i)](x))
        ####diferent scale disp maps from final layer
        if self.using_v == 1:
            for i in range(1,4):
                self.outputs[("disp",i)] = self.maxpool(self.outputs[("disp",i-1)])
        elif self.using_v == 2:
            for i in range(1,4):
                self.outputs[("disp",i)] = self.nonlin(self.maxpool(self.outputs[("disp",i-1)]))
        elif self.using_v == 3:
            for i in range(1,4):
                self.outputs[("disp",i)] = cv2.GussianBlur(self.outputs[("disp",i-1)],(3,3),0)
        elif self.using_v == 4:
            for i in range(1,4):
                self.outputs[("disp",i)] = self.outputs[("disp",0)].clone()


        return self.outputs
