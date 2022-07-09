import os
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor, Type
import torch

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=64):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.))

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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


class ResNet(nn.Module):

    def __init__(self, taskcla, block, layers, num_classes=1000, base_width=64):
        self.taskcla = taskcla
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.base_width = base_width

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #nn.AvgPool2d(7, stride=1) for 32x32

        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.last=nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(nn.Linear(512 * block.expansion, n))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, base_width=self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        y = []
        for t,i in self.taskcla:
            y.append(self.last[t](x))

        return y



def load_pretrained_pkl(name, model_path='pretrained_models'):
    pretrained_model_path = {
        'imageNet': 'resnet50-19c8e357.pth',
        'moco': 'moco_v1_200ep_pretrain.pth',
        'maskrcnn': 'maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
        'deeplab': 'deeplabv3_resnet50_coco-cd0a2569.pth',
        'keyPoint': 'keypointrcnn_resnet50_fpn_coco-fc266e95.pth',
    }

    pkl = torch.load(os.path.join(model_path, pretrained_model_path[name]))
    state_dict = {}

    if name == 'imageNet':
        for k, v in pkl.items():
            if k.startswith("fc."):
                continue
            state_dict[k] = v
    elif name == 'moco':
        pkl = pkl['state_dict']
        state_dict = {}
        for k, v in pkl.items():
            if not k.startswith("module.encoder_q."):
                continue
            k = k.replace("module.encoder_q.", "")
            if k.startswith("fc."):
                continue
            state_dict[k] = v
    elif name == 'maskrcnn':
        for k, v in pkl.items():
            if not k.startswith("backbone.body."):
                continue
            k = k.replace("backbone.body.", "")
            state_dict[k] = v
    elif name == 'deeplab':
        for k, v in pkl.items():
            if not k.startswith("backbone."):
                continue
            k = k.replace("backbone.", "")
            state_dict[k] = v
    elif name == 'keyPoint':
        for k, v in pkl.items():
            if not k.startswith("backbone.body."):
                continue
            k = k.replace("backbone.body.", "")
            state_dict[k] = v

    return state_dict

def Net(pt, taskcla, **kwargs):
    model = ResNet(taskcla, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pt != 'scratch':
        state_dict = load_pretrained_pkl(pt)
        model.load_state_dict(state_dict, False)

    return model
