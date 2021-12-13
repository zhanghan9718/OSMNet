"""
SSLCNet (Short-term Shortcut and Long-term Concatenation)
"""

import torch
import torch.nn as nn


class L2NormDense(nn.Module):
    def __init__(self):
        super(L2NormDense,self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x = x / norm.unsqueeze(1).expand_as(x)
        return x


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

        #

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class SSLCNet(nn.Module):

    def __init__(self, block, num_block):
        super().__init__()

        self.in_channels = 32

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.conv2_x = self._make_layer(block, 32, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 64, num_block[1], 1)
        self.conv4_x = self._make_layer(block, 128, num_block[2], 1)

        self.conv2 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(160, 9, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(9))

        self.conv_cat = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        """

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(
            -1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, x):
        output1 = self.conv1(self.input_norm(x))
        output2 = self.conv2_x(output1)
        output = self.conv3_x(output2)
        output = self.conv_cat(torch.cat([output, output1], 1))
        output = self.conv4_x(output)
        output = self.conv2(torch.cat([output, output2], 1))

        return output


class SSLCNetPseudo(nn.Module):
    """SSLCNetPseudo model definition
    """
    def __init__(self):
        super(SSLCNetPseudo, self).__init__()
        self.ResNet_Opt = SSLCNet(BasicBlock, [1, 1, 1, 1])
        self.ResNet_Sar = SSLCNet(BasicBlock, [1, 1, 1, 1])

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(
            -1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input_opt, input_sar):
        input_opt = self.input_norm(input_opt)
        input_sar = self.input_norm(input_sar)
        features_opt = self.ResNet_Opt(input_opt)
        features_sar = self.ResNet_Sar(input_sar)

        return L2NormDense()(features_opt), L2NormDense()(features_sar)

