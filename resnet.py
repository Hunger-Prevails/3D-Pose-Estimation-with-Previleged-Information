import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.model_zoo as model_zoo


__all__ = ['Bottleneck', 'ResNet', 'resnet50', 'resnet101']


def conv3x3(in_planes, out_planes, stride = 1):
    return nn.Conv2d(
        in_channels = in_planes,
        out_channels = out_planes,
        kernel_size = 3,
        stride = stride,
        padding = 1,
        bias = False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
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
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(out + residual)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
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
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(out + residual)


class ResNet(nn.Module):

    def __init__(self, block, layers, args):
        
    	assert args.stride in [4, 8, 16, 32]

        super(ResNet, self).__init__()

        self.inplanes = 64
        
        stride2 = int(np.minimum(np.maximum(np.log2(args.stride), 2), 3) - 1)
        stride3 = int(np.minimum(np.maximum(np.log2(args.stride), 3), 4) - 2)
        stride4 = int(np.minimum(np.maximum(np.log2(args.stride), 4), 5) - 3)

        side_out = (args.side_in - 1) / args.stride + 1

        self.conv1 = nn.Conv2d(
            in_channels = 4 if args.extra_channel else 3,
            out_channels = 64,
            kernel_size = 7,
            stride = 2,
            padding = 3,
            bias = False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride = stride2, dilation = 3 - stride2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = stride3, dilation = 3 - stride3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = stride4, dilation = 3 - stride4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2.0 / n) ** 0.5)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.cam_regressor = nn.Conv2d(
            in_channels = 512 * block.expansion,
            out_channels = args.depth * args.num_joints,
            kernel_size = 3,
            padding = 1
        )

        self.mat_regressor = nn.Conv2d(
            in_channels = 512 * block.expansion,
            out_channels = args.num_joints,
            kernel_size = 3,
            padding = 1
        ) if args.joint_space else None

        self.atn_regressor = nn.Conv2d(
            in_channels = 512 * block.expansion,
            out_channels = args.num_joints,
            kernel_size = 3,
            padding = 1
        ) if args.do_attention else None

        self.atn_pool = nn.AvgPool2d((side_out, side_out))

    def _make_layer(self, block, planes, blocks, stride = 1, dilation = 1):
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
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.atn_regressor:
            attention = self.atn_regressor(x)
            attention = self.atn_pool(attention).squeeze(-1).squeeze(-1)

            max_val = torch.max(attention, dim = -1, keepdim = True)[0]
            attention = torch.exp(attention - max_val)
            attention = attention / torch.sum(attention, dim = -1, keepdim = True)

            return self.cam_regressor(x), self.mat_regressor(x), attention

        elif self.mat_regressor:
            return self.cam_regressor(x), self.mat_regressor(x)

        else:
            return self.cam_regressor(x)


def resnet18(args):
    if args.pretrain:
        model = ResNet(BasicBlock, [2, 2, 2, 2], args)
        toy_dict = torch.load(args.model_path)
        model_dict = model.state_dict()
        
        keys = toy_dict.keys()

        if args.extra_channel:
            tensor = model_dict['conv1.weight']
            tensor[:, :3] = toy_dict['conv1.weight'].data
            toy_dict['conv1.weight'].data = tensor

        for key in keys:
            if key not in model_dict:
                print 'key [', key, '] deleted'
                del toy_dict[key]

        model_dict.update(toy_dict)
        model.load_state_dict(model_dict)

        return model

    return ResNet(BasicBlock, [2, 2, 2, 2], args)


def resnet50(args):
    if args.pretrain:
        model = ResNet(Bottleneck, [3, 4, 6, 3], args)
        toy_dict = torch.load(args.model_path)
        model_dict = model.state_dict()
        
        keys = toy_dict.keys()

        if args.extra_channel:
            tensor = model_dict['conv1.weight']
            tensor[:, :3] = toy_dict['conv1.weight'].data
            toy_dict['conv1.weight'].data = tensor

        for key in keys:
            if key not in model_dict:
                print 'key [', key, '] deleted'
                del toy_dict[key]
        
        model_dict.update(toy_dict)
        model.load_state_dict(model_dict)

        return model

    return ResNet(Bottleneck, [3, 4, 6, 3], args)
