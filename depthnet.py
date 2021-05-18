import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.model_zoo as model_zoo


__all__ = ['Bottleneck', 'ResNet', 'resnet18', 'resnet50']

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride = 1, dilation = 1, downsample = None, skip_relu = False):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels = inplanes,
            out_channels = planes,
            kernel_size = 3,
            stride = stride,
            dilation = dilation,
            padding = dilation,
            bias = False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            in_channels = planes,
            out_channels = planes,
            kernel_size = 3,
            padding = 1,
            bias = False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        res = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            res = self.downsample(res)

        if skip_relu:
            return out + res
        else:
            return F.relu(out + res)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride = 1, dilation = 1, downsample = None, skip_relu = False):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels = inplanes,
            out_channels = planes,
            kernel_size = 1,
            bias = False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            in_channels = planes,
            out_channels = planes,
            kernel_size = 3,
            stride = stride,
            padding = dilation,
            dilation = dilation,
            bias = False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(
            in_channels = planes,
            out_channels = planes * 4,
            kernel_size = 1,
            bias = False
        )
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        res = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            res = self.downsample(res)

        if skip_relu:
            return out + res
        else:
            return F.relu(out + res)


class ResNet(nn.Module):

    def __init__(self, block, layers, args):
        
        assert args.stride in [4, 8, 16, 32]

        super(ResNet, self).__init__()

        self.early_dist = args.early_dist
        self.skip_relu = args.skip_relu
        
        stride2 = int(np.minimum(np.maximum(np.log2(args.stride), 2), 3) - 1)
        stride3 = int(np.minimum(np.maximum(np.log2(args.stride), 3), 4) - 2)
        stride4 = int(np.minimum(np.maximum(np.log2(args.stride), 4), 5) - 3)

        dilate2 = (3 - stride2)
        dilate3 = (3 - stride2) * (3 - stride3)
        dilate4 = (3 - stride2) * (3 - stride3) * (3 - stride4)

        self.conv1 = nn.Conv2d(1 if args.depth_only else 3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.inplanes = 64
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride = stride2, dilation = dilate2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = stride3, dilation = dilate3, skip_relu = args.skip_relu)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = stride4, dilation = dilate4, skip_relu = args.skip_relu)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.regressor = nn.Conv2d(512 * block.expansion, args.depth * args.num_joints, 3, padding = 1)

    def freeze_batchnorm(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def _make_layer(self, block, planes, blocks, stride = 1, dilation = 1, skip_relu = False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels = self.inplanes,
                    out_channels = planes * block.expansion,
                    kernel_size = 1,
                    stride = stride,
                    bias = False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks - 1):
            layers.append(block(self.inplanes, planes))

        layers.append(block(self.inplanes, planes, skip_relu = skip_relu))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        m = self.layer3(x)
        n = self.layer4(F.relu(m) if self.skip_relu else m)
        z = self.regressor(F.relu(n) if self.skip_relu else n)

        return z, m if self.early_dist else n


def build_resnet(block, layers, args, pretrain):
    if pretrain:
        model = ResNet(block, layers, args)
        toy_dict = torch.load(args.host_path)['model'] if args.depth_host else torch.load(args.model_path)
        model_dict = model.state_dict()

        keys = list(toy_dict.keys())

        if args.depth_only:
            tensor = toy_dict['conv1.weight'].data
            toy_dict['conv1.weight'].data = torch.clone(tensor[:, :1])

        if args.depth_host:
            tensor = toy_dict['conv1.weight'].data
            toy_dict['conv1.weight'].data = (tensor / 3).repeat(1, 3, 1, 1)

        for key in keys:
            if key not in model_dict:
                print('key [', key, '] deleted')
                del toy_dict[key]

        model_dict.update(toy_dict)
        model.load_state_dict(model_dict)

        return model

    return ResNet(block, layers, args)


def resnet18(args, pretrain):
    return build_resnet(BasicBlock, [2, 2, 2, 2], args, pretrain)


def resnet50(args, pretrain):
    return build_resnet(Bottleneck, [3, 4, 6, 3], args, pretrain)
