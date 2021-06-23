import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch import optim
from torch.distributions import Normal 
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from models.spectral_norm import spectral_norm
from models import igebm


class DummyDistribution(nn.Module):
    """ Function-less class introduced for backward-compatibility of model checkpoint files. """
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer('sigma', torch.tensor(0., dtype=torch.float))

    def forward(self, x):
        return self.net(x)


class IsotropicGaussian(nn.Module):
    """Isotripic Gaussian density function paramerized by a neural net.
    standard deviation is a free scalar parameter"""
    def __init__(self, net, sigma=1., sigma_trainable=False, error_normalize=True, deterministic=False):
        super().__init__()
        self.net = net
        self.sigma_trainable = sigma_trainable
        self.error_normalize = error_normalize
        self.deterministic = deterministic
        if sigma_trainable:
            # self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float))
            self.register_parameter('sigma', nn.Parameter(torch.tensor(sigma, dtype=torch.float)))
        else:
            self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float))

    def log_likelihood(self, x, z):
        decoder_out = self.net(z)
        if self.deterministic:
            return - ((x - decoder_out)**2).view((x.shape[0], -1)).sum(dim=1) 
        else:
            D = torch.prod(torch.tensor(x.shape[1:]))
            # sig = torch.tensor(1, dtype=torch.float32)
            sig = self.sigma
            const = - D * 0.5 * torch.log(2 * torch.tensor(np.pi, dtype=torch.float32)) - D * torch.log(sig)
            loglik = const - 0.5 * ((x - decoder_out)**2).view((x.shape[0], -1)).sum(dim=1) / (sig ** 2)
            return loglik

    def error(self, x, x_hat):
        if not self.error_normalize:
            return (((x - x_hat) / self.sigma) ** 2).view(len(x), -1).sum(-1)
        else:
            return ((x - x_hat) ** 2).view(len(x), -1).mean(-1)

    def forward(self, z):
        """returns reconstruction"""
        return self.net(z)

    def sample(self, z):
        if self.deterministic:
            return self.mean(z)
        else:
            x_hat = self.net(z)
            return x_hat + torch.randn_like(x_hat) * self.sigma

    def mean(self, z):
        return self.net(z)

    def max_log_likelihood(self, x):
        if self.deterministic:
            return torch.tensor(0., dtype=torch.float, device=x.device)
        else:
            D = torch.prod(torch.tensor(x.shape[1:]))
            sig = self.sigma
            const = - D * 0.5 * torch.log(2 * torch.tensor(np.pi, dtype=torch.float32)) - D * torch.log(sig)
            return const

class IsotropicLaplace(nn.Module):
    """Isotropic Laplace density function -- equivalent to using L1 error """
    def __init__(self, net, sigma=0.1, sigma_trainable=False):
        super().__init__()
        self.net = net
        self.sigma_trainable = sigma_trainable
        if sigma_trainable:
            self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float))
        else:
            self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float))

    def log_likelihood(self, x, z):
        # decoder_out = self.net(z)
        # D = torch.prod(torch.tensor(x.shape[1:]))
        # sig = torch.tensor(1, dtype=torch.float32)
        # const = - D * 0.5 * torch.log(2 * torch.tensor(np.pi, dtype=torch.float32)) - D * torch.log(sig)
        # loglik = const - 0.5 * (torch.abs(x - decoder_out)).view((x.shape[0], -1)).sum(dim=1) / (sig ** 2)
        # return loglik
        raise NotImplementedError

    def error(self, x, x_hat):
        if self.sigma_trainable:
            return ((torch.abs(x - x_hat) / self.sigma)).view(len(x), -1).sum(-1)
        else:
            return (torch.abs(x - x_hat)).view(len(x), -1).mean(-1)

    def forward(self, z):
        """returns reconstruction"""
        return self.net(z)

    def sample(self, z):
        # x_hat = self.net(z) 
        # return x_hat + torch.randn_like(x_hat) * self.sigma
        raise NotImplementedError


class ConvNet2FC(nn.Module):
    """additional 1x1 conv layer at the top"""
    def __init__(self, in_chan=1, out_chan=64, nh=8, nh_mlp=512, out_activation='linear', use_spectral_norm=False):
        """nh: determines the numbers of conv filters"""
        super(ConvNet2FC, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, nh * 4, kernel_size=3, bias=True)
        self.conv2 = nn.Conv2d(nh * 4, nh * 8, kernel_size=3, bias=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.conv4 = nn.Conv2d(nh * 8, nh * 16, kernel_size=3, bias=True)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(nh * 16, nh_mlp, kernel_size=4, bias=True)
        self.conv6 = nn.Conv2d(nh_mlp, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = get_activation(out_activation)

        if use_spectral_norm:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            self.conv3 = spectral_norm(self.conv3)
            self.conv4 = spectral_norm(self.conv4)
            self.conv5 = spectral_norm(self.conv5)

        layers = [self.conv1,
                  nn.ReLU(),
                  self.conv2,
                  nn.ReLU(),
                  self.max1,
                  self.conv3,
                  nn.ReLU(),
                  self.conv4,
                  nn.ReLU(),
                  self.max2,
                  self.conv5,
                  nn.ReLU(),
                  self.conv6,]
        if self.out_activation is not None:
            layers.append(self.out_activation)


        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeConvNet2(nn.Module):
    def __init__(self, in_chan=1, out_chan=1, nh=8, out_activation='linear',
                 use_spectral_norm=False):
        """nh: determines the numbers of conv filters"""
        super(DeConvNet2, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_chan, nh * 16, kernel_size=4, bias=True)
        self.conv2 = nn.ConvTranspose2d(nh * 16, nh * 8, kernel_size=3, bias=True)
        self.conv3 = nn.ConvTranspose2d(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.conv4 = nn.ConvTranspose2d(nh * 8, nh * 4, kernel_size=3, bias=True)
        self.conv5 = nn.ConvTranspose2d(nh * 4, out_chan, kernel_size=3, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = get_activation(out_activation) 

        if use_spectral_norm:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            self.conv3 = spectral_norm(self.conv3)
            self.conv4 = spectral_norm(self.conv4)
            self.conv5 = spectral_norm(self.conv5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x


'''
ConvNet for CIFAR10, following architecture in (Ghosh et al., 2019)
but excluding batch normalization
'''

class ConvNet64(nn.Module):
    """ConvNet architecture for CelebA64 following Ghosh et al., 2019"""
    def __init__(self, in_chan=3, out_chan=64, nh=32, out_activation='linear', activation='relu',
                 num_groups=None, use_bn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chan, nh * 4, kernel_size=5, bias=True, stride=2)
        self.conv2 = nn.Conv2d(nh * 4, nh * 8, kernel_size=5, bias=True, stride=2)
        self.conv3 = nn.Conv2d(nh * 8, nh * 16, kernel_size=5, bias=True, stride=2)
        self.conv4 = nn.Conv2d(nh * 16, nh * 32, kernel_size=5, bias=True, stride=2)
        self.fc1 = nn.Conv2d(nh * 32, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.num_groups = num_groups
        self.use_bn = use_bn

        layers = []
        layers.append(self.conv1)
        if num_groups is not None:
            layers.append(self.get_norm_layer(num_channels=nh * 4))
        layers.append(get_activation(activation))
        layers.append(self.conv2)
        if num_groups is not None:
            layers.append(self.get_norm_layer(num_channels=nh * 8))
        layers.append(get_activation(activation))
        layers.append(self.conv3)
        if num_groups is not None:
            layers.append(self.get_norm_layer(num_channels=nh * 16))
        layers.append(get_activation(activation))
        layers.append(self.conv4)
        if num_groups is not None:
            layers.append(self.get_norm_layer(num_channels=nh * 32))
        layers.append(get_activation(activation))
        layers.append(self.fc1)
        out_activation = get_activation(out_activation)
        if out_activation is not None:
            layers.append(out_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def get_norm_layer(self, num_channels):
        if self.num_groups is not None:
            return nn.GroupNorm(num_groups=self.num_groups, num_channels=num_channels)
        elif self.use_bn:
            return nn.BatchNorm2d(num_channels)


class DeConvNet64(nn.Module):
    """ConvNet architecture for CelebA64 following Ghosh et al., 2019"""
    def __init__(self, in_chan=64, out_chan=3, nh=32, out_activation='linear', activation='relu',
                 num_groups=None, use_bn=False):
        super().__init__()
        self.fc1 = nn.ConvTranspose2d(in_chan, nh * 32, kernel_size=8, bias=True)
        self.conv1 = nn.ConvTranspose2d(nh * 32, nh * 16, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv2 = nn.ConvTranspose2d(nh * 16, nh * 8, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv3 = nn.ConvTranspose2d(nh * 8, nh * 4, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv4 = nn.ConvTranspose2d(nh * 4, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.num_groups = num_groups
        self.use_bn = use_bn

        layers = []
        layers.append(self.fc1)
        if num_groups is not None:
            layers.append(self.get_norm_layer(num_channels=nh * 32))
        layers.append(get_activation(activation))
        layers.append(self.conv1)
        if num_groups is not None:
            layers.append(self.get_norm_layer(num_channels=nh * 16))
        layers.append(get_activation(activation))
        layers.append(self.conv2)
        if num_groups is not None:
            layers.append(self.get_norm_layer(num_channels=nh * 8))
        layers.append(get_activation(activation))
        layers.append(self.conv3)
        if num_groups is not None:
            layers.append(self.get_norm_layer(num_channels=nh * 4))
        layers.append(get_activation(activation))
        layers.append(self.conv4)
        out_activation = get_activation(out_activation)
        if out_activation is not None:
            layers.append(out_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def get_norm_layer(self, num_channels):
        if self.num_groups is not None:
            return nn.GroupNorm(num_groups=self.num_groups, num_channels=num_channels)
        elif self.use_bn:
            return nn.BatchNorm2d(num_channels)


class ConvMLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None, out_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
        if out_dim is None:
            out_dim = dim

        self.block = nn.Sequential(
                nn.Conv2d(dim, hidden_dim, kernel_size=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1))

    def forward(self, x):
        return self.block(x)


class DeConvNet3(nn.Module):
    def __init__(self, in_chan=1, out_chan=1, nh=32, out_activation='linear',
                 activation='relu', num_groups=None):
        """nh: determines the numbers of conv filters"""
        super(DeConvNet3, self).__init__()
        self.num_groups = num_groups
        self.fc1 = nn.ConvTranspose2d(in_chan, nh * 32, kernel_size=8, bias=True)
        self.conv1 = nn.ConvTranspose2d(nh * 32, nh * 16, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv2 = nn.ConvTranspose2d(nh * 16, nh * 8, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv3 = nn.ConvTranspose2d(nh * 8, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan

        layers = [self.fc1,] 
        layers += [] if self.num_groups is None else [self.get_norm_layer(nh*32)]
        layers += [get_activation(activation), self.conv1,]
        layers += [] if self.num_groups is None else [self.get_norm_layer(nh*16)]
        layers += [get_activation(activation), self.conv2,]
        layers += [] if self.num_groups is None else [self.get_norm_layer(nh*8)]
        layers += [get_activation(activation), self.conv3] 
        out_activation = get_activation(out_activation)
        if out_activation is not None:
            layers.append(out_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def get_norm_layer(self, num_channels):
        if self.num_groups is not None:
            return nn.GroupNorm(num_groups=self.num_groups, num_channels=num_channels)
        # elif self.use_bn:
        #     return nn.BatchNorm2d(num_channels)
        else:
            return None


class IGEBMEncoder(nn.Module):
    """Neural Network used in IGEBM"""
    def __init__(self, in_chan=3, out_chan=1, n_class=None, use_spectral_norm=False, keepdim=True):
        super().__init__()
        self.keepdim = keepdim
        self.use_spectral_norm = use_spectral_norm

        if use_spectral_norm:
            self.conv1 = spectral_norm(nn.Conv2d(in_chan, 128, 3, padding=1), std=1)
        else:
            self.conv1 = nn.Conv2d(in_chan, 128, 3, padding=1)

        self.blocks = nn.ModuleList(
            [
                igebm.ResBlock(128, 128, n_class, downsample=True, use_spectral_norm=use_spectral_norm),
                igebm.ResBlock(128, 128, n_class, use_spectral_norm=use_spectral_norm),
                igebm.ResBlock(128, 256, n_class, downsample=True, use_spectral_norm=use_spectral_norm),
                igebm.ResBlock(256, 256, n_class, use_spectral_norm=use_spectral_norm),
                igebm.ResBlock(256, 256, n_class, downsample=True, use_spectral_norm=use_spectral_norm),
                igebm.ResBlock(256, 256, n_class, use_spectral_norm=use_spectral_norm),
            ]
        )

        if keepdim:
            self.linear = nn.Conv2d(256, out_chan, 1, 1, 0)
        else:
            self.linear = nn.Linear(256, out_chan)
        if use_spectral_norm:
            self.linear = spectral_norm(self.linear)

    def forward(self, input, class_id=None):
        out = self.conv1(input)

        out = F.leaky_relu(out, negative_slope=0.2)

        for block in self.blocks:
            out = block(out, class_id)

        out = F.relu(out)
        if self.keepdim:
            out = F.adaptive_avg_pool2d(out, (1,1))
        else:
            out = out.view(out.shape[0], out.shape[1], -1).sum(2)

        out = self.linear(out)

        return out


class SphericalActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / x.norm(p=2, dim=1, keepdim=True)


# Fully Connected Network
def get_activation(s_act):
    if s_act == 'relu':
        return nn.ReLU(inplace=True)
    elif s_act == 'sigmoid':
        return nn.Sigmoid()
    elif s_act == 'softplus':
        return nn.Softplus()
    elif s_act == 'linear':
        return None
    elif s_act == 'tanh':
        return nn.Tanh()
    elif s_act == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif s_act == 'softmax':
        return nn.Softmax(dim=1)
    elif s_act == 'spherical':
        return SphericalActivation()
    else:
        raise ValueError(f'Unexpected activation: {s_act}')


class FCNet(nn.Module):
    """fully-connected network"""
    def __init__(self, in_dim, out_dim, l_hidden=(50,), activation='sigmoid', out_activation='linear',
                 use_spectral_norm=False):
        super().__init__()
        l_neurons = tuple(l_hidden) + (out_dim,)
        if isinstance(activation, str):
            activation = (activation,) * len(l_hidden)
        activation = tuple(activation) + (out_activation,)

        l_layer = []
        prev_dim = in_dim
        for i_layer, (n_hidden, act) in enumerate(zip(l_neurons, activation)):
            if use_spectral_norm and i_layer < len(l_neurons) - 1:  # don't apply SN to the last layer
                l_layer.append(spectral_norm(nn.Linear(prev_dim, n_hidden)))
            else:
                l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)
        self.in_dim = in_dim
        self.out_shape = (out_dim,) 

    def forward(self, x):
        return self.net(x)


class ConvMLP(nn.Module):
    def __init__(self, in_dim, out_dim, l_hidden=(50,), activation='sigmoid', out_activation='linear',
                 likelihood_type='isotropic_gaussian'):
        super(ConvMLP, self).__init__()
        self.likelihood_type = likelihood_type
        l_neurons = tuple(l_hidden) + (out_dim,)
        activation = (activation,) * len(l_hidden)
        activation = tuple(activation) + (out_activation,)

        l_layer = []
        prev_dim = in_dim
        for i_layer, (n_hidden, act) in enumerate(zip(l_neurons, activation)):
            l_layer.append(nn.Conv2d(prev_dim, n_hidden, 1, bias=True))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)
        self.in_dim = in_dim

    def forward(self, x):
        return self.net(x)


class FCResNet(nn.Module):
    """FullyConnected Residual Network
    Input - Linear - (ResBlock * K) - Linear - Output"""
    def __init__(self, in_dim, out_dim, res_dim, n_res_hidden=100, n_resblock=2, out_activation='linear', use_spectral_norm=False):
        super().__init__()
        l_layer = []
        block = nn.Linear(in_dim, res_dim)
        if use_spectral_norm:
            block = spectral_norm(block)
        l_layer.append(block)

        for i_resblock in range(n_resblock):
            block = FCResBlock(res_dim, n_res_hidden, use_spectral_norm=use_spectral_norm)
            l_layer.append(block)
        l_layer.append(nn.ReLU())

        block = nn.Linear(res_dim, out_dim)
        if use_spectral_norm:
            block = spectral_norm(block)
        l_layer.append(block)
        out_activation = get_activation(out_activation)
        if out_activation is not None:
            l_layer.append(out_activation)
        self.net = nn.Sequential(*l_layer)

    def forward(self, x):
        return self.net(x)


class FCResBlock(nn.Module):
    def __init__(self, res_dim, n_res_hidden, use_spectral_norm=False):
        super().__init__()
        if use_spectral_norm:
            self.net = nn.Sequential(nn.ReLU(),
                                     spectral_norm(nn.Linear(res_dim, n_res_hidden)),
                                     nn.ReLU(),
                                     spectral_norm(nn.Linear(n_res_hidden, res_dim)))
        else:
            self.net = nn.Sequential(nn.ReLU(),
                                     nn.Linear(res_dim, n_res_hidden),
                                     nn.ReLU(),
                                     nn.Linear(n_res_hidden, res_dim))

    def forward(self, x):
        return x + self.net(x)
