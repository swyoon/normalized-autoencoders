import pytest
import torch
from torch.optim import Adam
from models.modules import FCNet, FCResNet, IsotropicGaussian
from models.modules import ConvNet64, DeConvNet64
from models.modules_sngan import GeneratorNoBN

@pytest.mark.parametrize('out_activation', ['linear', 'spherical'])
def test_fcnet(out_activation):
    net = FCNet(in_dim=1, out_dim=5, l_hidden=(50,), activation='sigmoid', out_activation=out_activation)
    X = torch.rand(10, 1)
    Z = net(X)



def test_sngan_generator():
    z_dim = 4
    channel = 3
    z = torch.rand(3, z_dim, 1, 1, dtype=torch.float)

    net = GeneratorNoBN(z_dim, channel, hidden_dim=32, out_activation='sigmoid')
    out = net(z)
    assert out.shape == (3, 3, 32, 32)


def test_fcresnet():
    net = FCResNet(in_dim=2, out_dim=2, res_dim=20, n_res_hidden=100, n_resblock=2)

    x = torch.rand((20, 2))
    out = net(x)
    assert out.shape == (20, 2)


@pytest.mark.parametrize('distribution', ['IsotropicGaussian'])
def test_distribution_modules(distribution):
    if distribution == 'IsotropicGaussian':
        dist = IsotropicGaussian

    N = 10; z_dim = 3; x_dim = 5
    z = torch.rand(N, z_dim)
    x = torch.rand(N, x_dim)
    net = FCNet(in_dim=z_dim, out_dim=x_dim, l_hidden=(50,), activation='sigmoid', out_activation='sigmoid')
    net = dist(net)
    lik = net.log_likelihood(x, z)
    assert lik.shape == (N,)
    samples = net.sample(z)
    assert samples.shape == (N, x_dim)


@pytest.mark.parametrize('num_groups', [None, 2])
def test_convnet64(num_groups):
    x = torch.rand(2, 3, 64, 64)
    encoder = ConvNet64(num_groups=num_groups)
    decoder = DeConvNet64(num_groups=num_groups)
    recon = decoder(encoder(x))
    assert x.shape == recon.shape

