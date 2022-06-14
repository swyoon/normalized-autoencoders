import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable

import re
import math
import collections
from torch.utils import model_zoo


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class Fire_UP(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire_UP, self).__init__()
        self.inplanes = inplanes
        self.upsample = nn.ConvTranspose2d(inplanes, inplanes,
                                           kernel_size=4, stride=2, padding=1)
        self.squeeze = nn.ConvTranspose2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.ConvTranspose2d(squeeze_planes, expand1x1_planes,
                                            kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.ConvTranspose2d(squeeze_planes, expand3x3_planes,
                                            kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
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


class BasicBlock_UP(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, upsample=2):
        super(BasicBlock_UP, self).__init__()

        kernel = 2 * upsample
        stride = upsample
        pad = int(upsample / 2)

        self.conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=kernel, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = nn.ConvTranspose2d(inplanes, planes, kernel_size=stride, stride=stride, padding=0, bias=False)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False,
                              dilation=d)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


class UnitDilated8(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        n = int(nOut / 8)
        # n1 = nOut - 5 * n
        n_mid = int(nIn / 2)
        self.c1 = nn.Conv2d(nIn, n_mid, 3, 1, 1)
        self.d1 = CDilated(n_mid, n, 3, 1, 1)
        self.d2 = CDilated(n_mid, n, 3, 1, 2)
        self.d3 = CDilated(n_mid, n, 3, 1, 3)
        self.d4 = CDilated(n_mid, n, 3, 1, 4)
        self.d6 = CDilated(n_mid, n, 3, 1, 6)
        self.d8 = CDilated(n_mid, n, 3, 1, 8)
        # self.d11 = CDilated(n_mid, n, 3, 1, 11)
        # self.d16 = CDilated(n_mid, n, 3, 1, 16)
#        self.d1 = CDilated(nOut, n, 3, 1, 1)
#        self.d2 = CDilated(nOut, n, 3, 1, 2)
#        self.d3 = CDilated(nOut, n, 3, 1, 3)
#        self.d4 = CDilated(nOut, n, 3, 1, 4)
#        self.d6 = CDilated(nOut, n, 3, 1, 6)
#        self.d8 = CDilated(nOut, n, 3, 1, 8)
#        self.d11 = CDilated(nOut, n, 3, 1, 11)
#        self.d16 = CDilated(nOut, n, 3, 1, 16)

    def forward(self, input):
        output1 = self.c1(input)
        # d1 = self.d1(output1)
        # d2 = self.d2(output1)
        d3 = self.d3(output1)
        d4 = self.d4(output1)
        d6 = self.d6(output1)
        d8 = self.d8(output1)
        # d11 = self.d11(output1)
        # d16 = self.d16(output1)

        # d2 = d2 + d1
        # d3 = d2 + d3
        d4 = d3 + d4
        d6 = d4 + d6
        d8 = d6 + d8
        # d11 = d8 + d11
        # d16 = d11 + d16

        output = torch.cat([d3, d4, d6, d8],1)
        # output = torch.cat([d1, d2, d3, d4, d6, d8, d11, d16], 1)
        return output


class UnitDilated(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        n = int(nOut / 8)
        n_mid = int(nIn / 2)
        self.c1 = nn.Conv2d(nIn, n_mid, 3, 1, 1)
        self.d1 = CDilated(n_mid, n, 3, 1, 1)
        self.d2 = CDilated(n_mid, n, 3, 1, 2)
        self.d3 = CDilated(n_mid, n, 3, 1, 3)
        self.d4 = CDilated(n_mid, n, 3, 1, 4)
        self.d6 = CDilated(n_mid, n, 3, 1, 6)
        self.d8 = CDilated(n_mid, n, 3, 1, 8)
        self.d11 = CDilated(n_mid, n, 3, 1, 11)
        self.d16 = CDilated(n_mid, n, 3, 1, 16)
#        self.d1 = CDilated(nOut, n, 3, 1, 1)
#        self.d2 = CDilated(nOut, n, 3, 1, 2)
#        self.d3 = CDilated(nOut, n, 3, 1, 3)
#        self.d4 = CDilated(nOut, n, 3, 1, 4)
#        self.d6 = CDilated(nOut, n, 3, 1, 6)
#        self.d8 = CDilated(nOut, n, 3, 1, 8)
#        self.d11 = CDilated(nOut, n, 3, 1, 11)
#        self.d16 = CDilated(nOut, n, 3, 1, 16)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)
        d4 = self.d4(output1)
        d6 = self.d6(output1)
        d8 = self.d8(output1)
        d11 = self.d11(output1)
        d16 = self.d16(output1)

        d2 = d2 + d1
        d3 = d2 + d3
        d4 = d3 + d4
        d6 = d4 + d6
        d8 = d6 + d8
        d11 = d8 + d11
        d16 = d11 + d16

        # output = torch.cat([d1, d2, d3, d4, d6, d8],1)
        output = torch.cat([d1, d2, d3, d4, d6, d8, d11, d16], 1)
        return output


class conv2DBatchNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(conv2DBatchNorm, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cb_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_filters)))
        else:
            self.cb_unit = nn.Sequential(conv_mod)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class conv2DGroupNorm(nn.Module):
    def __init__(
        self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, n_groups=16
    ):
        super(conv2DGroupNorm, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        self.cg_unit = nn.Sequential(conv_mod, nn.GroupNorm(n_groups, int(n_filters)))

    def forward(self, inputs):
        outputs = self.cg_unit(inputs)
        return outputs


class deconv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True):
        super(deconv2DBatchNorm, self).__init__()

        self.dcb_unit = nn.Sequential(
            nn.ConvTranspose2d(
                int(in_channels),
                int(n_filters),
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                bias=bias,
            ),
            nn.BatchNorm2d(int(n_filters)),
        )

    def forward(self, inputs):
        outputs = self.dcb_unit(inputs)
        return outputs


class conv2DBatchNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class conv2DGroupNormRelu(nn.Module):
    def __init__(
        self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, n_groups=16
    ):
        super(conv2DGroupNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        self.cgr_unit = nn.Sequential(
            conv_mod, nn.GroupNorm(n_groups, int(n_filters)), nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        outputs = self.cgr_unit(inputs)
        return outputs


class deconv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True):
        super(deconv2DBatchNormRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(
            nn.ConvTranspose2d(
                int(in_channels),
                int(n_filters),
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                bias=bias,
            ),
            nn.BatchNorm2d(int(n_filters)),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, is_dropout=False, p=0.5):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1), nn.ReLU())
            if is_dropout:
                self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1), nn.Dropout2d(p=p), nn.ReLU())
            else:
                self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1), nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class segnetDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetDown3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetUp2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp2, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class segnetUp3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp3, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class residualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, n_filters, stride=1, downsample=None):
        super(residualBlock, self).__init__()

        self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3, stride, 1, bias=False)
        self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1, bias=False)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.convbnrelu1(x)
        out = self.convbn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class residualBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, n_filters, stride=1, downsample=None):
        super(residualBottleneck, self).__init__()
        self.convbn1 = nn.Conv2DBatchNorm(in_channels, n_filters, k_size=1, bias=False)
        self.convbn2 = nn.Conv2DBatchNorm(
            n_filters, n_filters, k_size=3, padding=1, stride=stride, bias=False
        )
        self.convbn3 = nn.Conv2DBatchNorm(n_filters, n_filters * 4, k_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.convbn1(x)
        out = self.convbn2(out)
        out = self.convbn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class linknetUp(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(linknetUp, self).__init__()

        # B, 2C, H, W -> B, C/2, H, W
        self.convbnrelu1 = conv2DBatchNormRelu(
            in_channels, n_filters / 2, k_size=1, stride=1, padding=1
        )

        # B, C/2, H, W -> B, C/2, H, W
        self.deconvbnrelu2 = nn.deconv2DBatchNormRelu(
            n_filters / 2, n_filters / 2, k_size=3, stride=2, padding=0
        )

        # B, C/2, H, W -> B, C, H, W
        self.convbnrelu3 = conv2DBatchNormRelu(
            n_filters / 2, n_filters, k_size=1, stride=1, padding=1
        )

    def forward(self, x):
        x = self.convbnrelu1(x)
        x = self.deconvbnrelu2(x)
        x = self.convbnrelu3(x)
        return x


class FRRU(nn.Module):
    """
    Full Resolution Residual Unit for FRRN
    """

    def __init__(self, prev_channels, out_channels, scale, group_norm=False, n_groups=None):
        super(FRRU, self).__init__()
        self.scale = scale
        self.prev_channels = prev_channels
        self.out_channels = out_channels
        self.group_norm = group_norm
        self.n_groups = n_groups

        if self.group_norm:
            conv_unit = conv2DGroupNormRelu
            self.conv1 = conv_unit(
                prev_channels + 32,
                out_channels,
                k_size=3,
                stride=1,
                padding=1,
                bias=False,
                n_groups=self.n_groups,
            )
            self.conv2 = conv_unit(
                out_channels,
                out_channels,
                k_size=3,
                stride=1,
                padding=1,
                bias=False,
                n_groups=self.n_groups,
            )

        else:
            conv_unit = conv2DBatchNormRelu
            self.conv1 = conv_unit(
                prev_channels + 32, out_channels, k_size=3, stride=1, padding=1, bias=False
            )
            self.conv2 = conv_unit(
                out_channels, out_channels, k_size=3, stride=1, padding=1, bias=False
            )

        self.conv_res = nn.Conv2d(out_channels, 32, kernel_size=1, stride=1, padding=0)

    def forward(self, y, z):
        x = torch.cat([y, nn.MaxPool2d(self.scale, self.scale)(z)], dim=1)
        y_prime = self.conv1(x)
        y_prime = self.conv2(y_prime)

        x = self.conv_res(y_prime)
        upsample_size = torch.Size([_s * self.scale for _s in y_prime.shape[-2:]])
        x = F.upsample(x, size=upsample_size, mode="nearest")
        z_prime = z + x

        return y_prime, z_prime


class RU(nn.Module):
    """
    Residual Unit for FRRN
    """

    def __init__(self, channels, kernel_size=3, strides=1, group_norm=False, n_groups=None):
        super(RU, self).__init__()
        self.group_norm = group_norm
        self.n_groups = n_groups

        if self.group_norm:
            self.conv1 = conv2DGroupNormRelu(
                channels,
                channels,
                k_size=kernel_size,
                stride=strides,
                padding=1,
                bias=False,
                n_groups=self.n_groups,
            )
            self.conv2 = conv2DGroupNorm(
                channels,
                channels,
                k_size=kernel_size,
                stride=strides,
                padding=1,
                bias=False,
                n_groups=self.n_groups,
            )

        else:
            self.conv1 = conv2DBatchNormRelu(
                channels, channels, k_size=kernel_size, stride=strides, padding=1, bias=False
            )
            self.conv2 = conv2DBatchNorm(
                channels, channels, k_size=kernel_size, stride=strides, padding=1, bias=False
            )

    def forward(self, x):
        incoming = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + incoming


class residualConvUnit(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(residualConvUnit, self).__init__()

        self.residual_conv_unit = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=kernel_size),
        )

    def forward(self, x):
        input = x
        x = self.residual_conv_unit(x)
        return x + input


class multiResolutionFusion(nn.Module):
    def __init__(self, channels, up_scale_high, up_scale_low, high_shape, low_shape):
        super(multiResolutionFusion, self).__init__()

        self.up_scale_high = up_scale_high
        self.up_scale_low = up_scale_low

        self.conv_high = nn.Conv2d(high_shape[1], channels, kernel_size=3)

        if low_shape is not None:
            self.conv_low = nn.Conv2d(low_shape[1], channels, kernel_size=3)

    def forward(self, x_high, x_low):
        high_upsampled = F.upsample(
            self.conv_high(x_high), scale_factor=self.up_scale_high, mode="bilinear"
        )

        if x_low is None:
            return high_upsampled

        low_upsampled = F.upsample(
            self.conv_low(x_low), scale_factor=self.up_scale_low, mode="bilinear"
        )

        return low_upsampled + high_upsampled


class chainedResidualPooling(nn.Module):
    def __init__(self, channels, input_shape):
        super(chainedResidualPooling, self).__init__()

        self.chained_residual_pooling = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(5, 1, 2),
            nn.Conv2d(input_shape[1], channels, kernel_size=3),
        )

    def forward(self, x):
        input = x
        x = self.chained_residual_pooling(x)
        return x + input


class pyramidPooling(nn.Module):
    def __init__(
        self, in_channels, pool_sizes, model_name="pspnet", fusion_mode="cat", is_batchnorm=True
    ):
        super(pyramidPooling, self).__init__()

        bias = not is_batchnorm

        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(
                conv2DBatchNormRelu(
                    in_channels,
                    int(in_channels / len(pool_sizes)),
                    1,
                    1,
                    0,
                    bias=bias,
                    is_batchnorm=is_batchnorm,
                )
            )

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes
        self.model_name = model_name
        self.fusion_mode = fusion_mode

    def forward(self, x):
        h, w = x.shape[2:]

        if self.training or self.model_name != "icnet":  # general settings or pspnet
            k_sizes = []
            strides = []
            for pool_size in self.pool_sizes:
                k_sizes.append((int(h / pool_size), int(w / pool_size)))
                strides.append((int(h / pool_size), int(w / pool_size)))
        else:  # eval mode and icnet: pre-trained for 1025 x 2049
            k_sizes = [(8, 15), (13, 25), (17, 33), (33, 65)]
            strides = [(5, 10), (10, 20), (16, 32), (33, 65)]

        if self.fusion_mode == "cat":  # pspnet: concat (including x)
            output_slices = [x]

            for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                # out = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
                if self.model_name != "icnet":
                    out = module(out)
                out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
                output_slices.append(out)

            return torch.cat(output_slices, dim=1)
        else:  # icnet: element-wise sum (including x)
            pp_sum = x

            for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                # out = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
                if self.model_name != "icnet":
                    out = module(out)
                out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
                pp_sum = pp_sum + out

            return pp_sum


class bottleNeckPSP(nn.Module):
    def __init__(
        self, in_channels, mid_channels, out_channels, stride, dilation=1, is_batchnorm=True
    ):
        super(bottleNeckPSP, self).__init__()

        bias = not is_batchnorm

        self.cbr1 = conv2DBatchNormRelu(
            in_channels, mid_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm
        )
        if dilation > 1:
            self.cbr2 = conv2DBatchNormRelu(
                mid_channels,
                mid_channels,
                3,
                stride=stride,
                padding=dilation,
                bias=bias,
                dilation=dilation,
                is_batchnorm=is_batchnorm,
            )
        else:
            self.cbr2 = conv2DBatchNormRelu(
                mid_channels,
                mid_channels,
                3,
                stride=stride,
                padding=1,
                bias=bias,
                dilation=1,
                is_batchnorm=is_batchnorm,
            )
        self.cb3 = conv2DBatchNorm(
            mid_channels, out_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm
        )
        self.cb4 = conv2DBatchNorm(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )

    def forward(self, x):
        conv = self.cb3(self.cbr2(self.cbr1(x)))
        residual = self.cb4(x)
        return F.relu(conv + residual, inplace=True)


class bottleNeckIdentifyPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, stride, dilation=1, is_batchnorm=True):
        super(bottleNeckIdentifyPSP, self).__init__()

        bias = not is_batchnorm

        self.cbr1 = conv2DBatchNormRelu(
            in_channels, mid_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm
        )
        if dilation > 1:
            self.cbr2 = conv2DBatchNormRelu(
                mid_channels,
                mid_channels,
                3,
                stride=1,
                padding=dilation,
                bias=bias,
                dilation=dilation,
                is_batchnorm=is_batchnorm,
            )
        else:
            self.cbr2 = conv2DBatchNormRelu(
                mid_channels,
                mid_channels,
                3,
                stride=1,
                padding=1,
                bias=bias,
                dilation=1,
                is_batchnorm=is_batchnorm,
            )
        self.cb3 = conv2DBatchNorm(
            mid_channels, in_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm
        )

    def forward(self, x):
        residual = x
        x = self.cb3(self.cbr2(self.cbr1(x)))
        return F.relu(x + residual, inplace=True)


class residualBlockPSP(nn.Module):
    def __init__(
        self,
        n_blocks,
        in_channels,
        mid_channels,
        out_channels,
        stride,
        dilation=1,
        include_range="all",
        is_batchnorm=True,
    ):
        super(residualBlockPSP, self).__init__()

        if dilation > 1:
            stride = 1

        # residualBlockPSP = convBlockPSP + identityBlockPSPs
        layers = []
        if include_range in ["all", "conv"]:
            layers.append(
                bottleNeckPSP(
                    in_channels,
                    mid_channels,
                    out_channels,
                    stride,
                    dilation,
                    is_batchnorm=is_batchnorm,
                )
            )
        if include_range in ["all", "identity"]:
            for i in range(n_blocks - 1):
                layers.append(
                    bottleNeckIdentifyPSP(
                        out_channels, mid_channels, stride, dilation, is_batchnorm=is_batchnorm
                    )
                )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class cascadeFeatureFusion(nn.Module):
    def __init__(
        self, n_classes, low_in_channels, high_in_channels, out_channels, is_batchnorm=True
    ):
        super(cascadeFeatureFusion, self).__init__()

        bias = not is_batchnorm

        self.low_dilated_conv_bn = conv2DBatchNorm(
            low_in_channels,
            out_channels,
            3,
            stride=1,
            padding=2,
            bias=bias,
            dilation=2,
            is_batchnorm=is_batchnorm,
        )
        self.low_classifier_conv = nn.Conv2d(
            int(low_in_channels),
            int(n_classes),
            kernel_size=1,
            padding=0,
            stride=1,
            bias=True,
            dilation=1,
        )  # Train only
        self.high_proj_conv_bn = conv2DBatchNorm(
            high_in_channels,
            out_channels,
            1,
            stride=1,
            padding=0,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )

    def forward(self, x_low, x_high):
        x_low_upsampled = F.interpolate(
            x_low, size=get_interp_size(x_low, z_factor=2), mode="bilinear", align_corners=True
        )

        low_cls = self.low_classifier_conv(x_low_upsampled)

        low_fm = self.low_dilated_conv_bn(x_low_upsampled)
        high_fm = self.high_proj_conv_bn(x_high)
        high_fused_fm = F.relu(low_fm + high_fm, inplace=True)

        return high_fused_fm, low_cls


def get_interp_size(input, s_factor=1, z_factor=1):  # for caffe
    ori_h, ori_w = input.shape[2:]

    # shrink (s_factor >= 1)
    ori_h = (ori_h - 1) / s_factor + 1
    ori_w = (ori_w - 1) / s_factor + 1

    # zoom (z_factor >= 1)
    ori_h = ori_h + (ori_h - 1) * (z_factor - 1)
    ori_w = ori_w + (ori_w - 1) * (z_factor - 1)

    resize_shape = (int(ori_h), int(ori_w))
    return resize_shape


def interp(input, output_size, mode="bilinear"):
    n, c, ih, iw = input.shape
    oh, ow = output_size

    # normalize to [-1, 1]
    h = torch.arange(0, oh, dtype=torch.float, device=input.device) / (oh - 1) * 2 - 1
    w = torch.arange(0, ow, dtype=torch.float, device=input.device) / (ow - 1) * 2 - 1

    grid = torch.zeros(oh, ow, 2, dtype=torch.float, device=input.device)
    grid[:, :, 0] = w.unsqueeze(0).repeat(oh, 1)
    grid[:, :, 1] = h.unsqueeze(0).repeat(ow, 1).transpose(0, 1)
    grid = grid.unsqueeze(0).repeat(n, 1, 1, 1)  # grid.shape: [n, oh, ow, 2]
    grid = Variable(grid)
    if input.is_cuda:
        grid = grid.cuda()

    return F.grid_sample(input, grid, mode=mode)


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


########################################################################
############## UTILS FOR EfficientNet##############
########################################################################


########################################################################
############### HELPERS FUNCTIONS FOR MODEL ARCHITECTURE ###############
########################################################################


# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate',])


# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])


# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def relu_fn(x):
    """ Swish activation function """
    return x * torch.sigmoid(x)


def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype)  # uniform [0,1)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


class Conv2dSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]]*2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


########################################################################
############## HELPERS FUNCTIONS FOR LOADING MODEL PARAMS ##############
########################################################################


def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }
    return params_dict[model_name]


class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.

        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.

        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet(width_coefficient=None, depth_coefficient=None,
                 dropout_rate=0.2, drop_connect_rate=0.2, n_classes=2):
    """ Creates a efficientnet model. """

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        # data_format='channels_last',  # removed, this is always true in PyTorch
        num_classes=n_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None
    )

    return blocks_args, global_params


def get_model_params(model_name, override_params, num_classes=2):
    """ Get the block args and global params for a given model """
    if model_name.startswith('efficientnet'):
        w, d, _, p = efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = efficientnet(width_coefficient=w, depth_coefficient=d, dropout_rate=p, n_classes=num_classes)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


url_map = {
    'efficientnet-b0': 'http://storage.googleapis.com/public-models/efficientnet-b0-08094119.pth',
    'efficientnet-b1': 'http://storage.googleapis.com/public-models/efficientnet-b1-dbc7070a.pth',
    'efficientnet-b2': 'http://storage.googleapis.com/public-models/efficientnet-b2-27687264.pth',
    'efficientnet-b3': 'http://storage.googleapis.com/public-models/efficientnet-b3-c8376fa2.pth',
}

def load_pretrained_weights(model, model_name):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    state_dict = model_zoo.load_url(url_map[model_name])
    model.load_state_dict(state_dict)
    print('Loaded pretrained weights for {}'.format(model_name))


def weight_norm(net):
    norm = 0
    for param in net.parameters():
        norm += (param ** 2).sum()
    return norm

