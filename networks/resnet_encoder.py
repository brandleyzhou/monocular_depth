from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models # a subpackage containing different models 
import torch.utils.model_zoo as model_zoo#pretrained network

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1, plan="plan1"):
        #block:  block_type = models.resnet.BasicBlock or Bottleneck
        #layers: blocks = [2,2,2,2] if blocktype == BasicBlock else [3,4,6,3]
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.num_ch_enc = [64, 64, 128, 256, 512]
        self.block = block
        self.plan = plan
        self.plan_choices = {
                    "plan0":[
                        nn.Conv2d(num_input_images * 3,64,kernel_size=1,stride=2,padding=0,bias=False),
                        self._make_layer(block,64,layers[0])],
                    "plan1":[
                        nn.Conv2d(num_input_images * 3,self.num_ch_enc[0],kernel_size=7,stride=2,padding=3,bias=False),
                        self._make_layer(block,self.num_ch_enc[1],layers[0])],
                    "plan2":[
                        nn.Conv2d(num_input_images * 3,self.num_ch_enc[0],kernel_size=1,stride=2,padding=0,bias=False),
                        self._make_layer(block,self.num_ch_enc[1],layers[0])],
                    "plan3":[
                        nn.Conv2d(num_input_images * 3,self.num_ch_enc[0],kernel_size=1,stride=2,padding=0,bias=False),
                        nn.Sequential(
                            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
                            nn.ReLU()
                            )],
                    "plan4":[
                        nn.Conv2d(num_input_images * 3,self.num_ch_enc[0],kernel_size=7,stride=2,padding=3,bias=False),
                        nn.Sequential(
                            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
                            nn.ReLU()
                        )],
                    "plan5":[
                        nn.Sequential(
                            nn.Conv2d(num_input_images*3,self.num_ch_enc[0],kernel_size=1,stride=1,padding=0,bias=False),
                            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
                            ),
                        self._make_layer(block,64,layers[0])]
                    } 
        self.conv1 = self.plan_choices[self.plan][0]
        #because the output of self.conv1 is a list having an element, so here we use [0] to extract the element
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.plan_choices[self.plan][1]
        self.layer2 = self._make_layer(block, self.num_ch_enc[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.num_ch_enc[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.num_ch_enc[4], layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1,name ="plan0"):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks,num_input_images=num_input_images,plan = name)

    if pretrained:
        print('------Using a pretained model-------')
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained=False, num_input_images=1, plan = 'plan0'):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4
        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, True, num_input_images,plan)
        else:
            self.encoder = resnet_multiimage_input(num_layers, True, num_input_images,plan)
    
    def forward(self, input_image):
        #print("In model input_size",input_image.size())
        self.features = []
        x = (input_image - 0.45) / 0.225 # normalizetion?
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))
        return self.features# feature has 5 elements
