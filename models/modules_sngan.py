"""
ResNet architectures from SNGAN(Miyato et al., 2018)
Brought this code from:
    https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
"""
# ResNet generator and discriminator
from torch import nn
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F

import numpy as np


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)




channels = 3

class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockGeneratorNoBN(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGeneratorNoBN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.ReLU(),
            self.conv2
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockGeneratorGN(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, num_groups=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(),
            self.conv2
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)



class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                SpectralNorm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
            # if in_channels == out_channels:
            #     self.bypass = nn.AvgPool2d(2, stride=stride, padding=0)
            # else:
            #     self.bypass = nn.Sequential(
            #         SpectralNorm(nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)),
            #         nn.AvgPool2d(2, stride=stride, padding=0)
            #     )


    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            SpectralNorm(self.conv1),
            nn.ReLU(),
            SpectralNorm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            SpectralNorm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

GEN_SIZE=128
DISC_SIZE=128

class Generator(nn.Module):
    def __init__(self, z_dim, channels, hidden_dim=128, out_activation=None):
        super().__init__()
        self.z_dim = z_dim

        self.dense = nn.ConvTranspose2d(z_dim, hidden_dim, 4)
        self.final = nn.Conv2d(hidden_dim, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        l_layers = [
            ResBlockGenerator(hidden_dim, hidden_dim, stride=2),
            ResBlockGenerator(hidden_dim, hidden_dim, stride=2),
            ResBlockGenerator(hidden_dim, hidden_dim, stride=2),
            # nn.BatchNorm2d(hidden_dim),  # should be uncommented. temporarily commented for backward compatibility
            nn.ReLU(),
            self.final,
            ]

        if out_activation == 'sigmoid':
            l_layers.append(nn.Sigmoid())
        elif out_activation == 'tanh':
            l_layers.append(nn.Tanh())
        self.model = nn.Sequential(*l_layers)

    def forward(self, z):
        return self.model(self.dense(z))


class GeneratorNoBN(nn.Module):
    """remove batch normalization
    z vector is assumed to be 4-dimensional (fully-convolutional)"""
    def __init__(self, z_dim, channels, hidden_dim=128, out_activation=None):
        super().__init__()
        self.z_dim = z_dim

        # self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE)
        self.dense = nn.ConvTranspose2d(z_dim, hidden_dim, 4)
        self.final = nn.Conv2d(hidden_dim, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        l_layers = [
            ResBlockGeneratorNoBN(hidden_dim, hidden_dim, stride=2),
            ResBlockGeneratorNoBN(hidden_dim, hidden_dim, stride=2),
            ResBlockGeneratorNoBN(hidden_dim, hidden_dim, stride=2),
            nn.ReLU(),
            self.final]

        if out_activation == 'sigmoid':
            l_layers.append(nn.Sigmoid())
        elif out_activation == 'tanh':
            l_layers.append(nn.Tanh())
        self.model = nn.Sequential(*l_layers)

    def forward(self, z):
        # return self.model(self.dense(z).view(-1, GEN_SIZE, 4, 4))
        return self.model(self.dense(z))


class GeneratorNoBN64(nn.Module):
    """remove batch normalization
    z vector is assumed to be 4-dimensional (fully-convolutional)
    for generating 64x64 output"""
    def __init__(self, z_dim, channels, hidden_dim=128, out_activation=None):
        super().__init__()
        self.z_dim = z_dim

        # self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE)
        self.dense = nn.ConvTranspose2d(z_dim, hidden_dim, 4)
        self.final = nn.Conv2d(hidden_dim, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        l_layers = [
            ResBlockGeneratorNoBN(hidden_dim, hidden_dim, stride=2),
            ResBlockGeneratorNoBN(hidden_dim, hidden_dim, stride=2),
            ResBlockGeneratorNoBN(hidden_dim, hidden_dim, stride=2),
            ResBlockGeneratorNoBN(hidden_dim, hidden_dim, stride=2),
            nn.ReLU(),
            self.final]

        if out_activation == 'sigmoid':
            l_layers.append(nn.Sigmoid())
        elif out_activation == 'tanh':
            l_layers.append(nn.Tanh())
        self.model = nn.Sequential(*l_layers)

    def forward(self, z):
        # return self.model(self.dense(z).view(-1, GEN_SIZE, 4, 4))
        return self.model(self.dense(z))


class GeneratorGN(nn.Module):
    """replace Batch Normalization to Group Normalization
    z vector is assumed to be 4-dimensional (fully-convolutional)"""
    def __init__(self, z_dim, channels, hidden_dim=128, out_activation=None, num_groups=1):
        super().__init__()
        self.z_dim = z_dim

        # self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE)
        self.dense = nn.ConvTranspose2d(z_dim, hidden_dim, 4)
        self.final = nn.Conv2d(hidden_dim, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        l_layers = [
            ResBlockGeneratorGN(hidden_dim, hidden_dim, stride=2),
            ResBlockGeneratorGN(hidden_dim, hidden_dim, stride=2),
            ResBlockGeneratorGN(hidden_dim, hidden_dim, stride=2),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim),
            nn.ReLU(),
            self.final]

        if out_activation == 'sigmoid':
            l_layers.append(nn.Sigmoid())
        elif out_activation == 'tanh':
            l_layers.append(nn.Tanh())
        self.model = nn.Sequential(*l_layers)

    def forward(self, z):
        # return self.model(self.dense(z).view(-1, GEN_SIZE, 4, 4))
        return self.model(self.dense(z))




class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
                FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                nn.ReLU(),
                nn.AvgPool2d(8),
            )
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)

    def forward(self, x):
        return self.fc(self.model(x).view(-1,DISC_SIZE))
