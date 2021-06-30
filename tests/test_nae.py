import pytest
import torch
from torch.optim import Adam
from models.modules import FCNet
from models.nae import NAE, FFEBM
from models import get_model
from omegaconf import OmegaConf


def test_ffebm():
    net = FCNet(2, 1, out_activation='linear')
    model = FFEBM(net, x_step=3, x_stepsize=1., sampling='x')
    X = torch.randn((10, 2), dtype=torch.float)
    opt = Adam(model.parameters(), lr=1e-4)

    # forward
    lik = model.predict(X)

    # sample
    model._set_x_shape(X)
    d_sample = model.sample_x(10, 'cpu:0')

    # training
    model.train_step(X, opt)


@pytest.mark.parametrize('sampling', ['x', 'on_manifold', 'cd'])
def test_nae(sampling):
    encoder = FCNet(2, 1)
    decoder = FCNet(1, 2)
    nae = NAE(encoder, decoder, initial_dist='gaussian', sampling=sampling)
    opt = Adam(nae.parameters(), lr=1e-4)

    X = torch.randn((10, 2), dtype=torch.float)

    # forward
    lik = nae.predict(X)

    # sample
    nae._set_x_shape(X)
    nae._set_z_shape(X)
    d_sample = nae.sample(X)

    # training
    nae.train_step(X, opt)


def test_cifar():
    cfg = OmegaConf.load('configs/cifar_ood_nae/z32gn.yml')
    model = get_model(cfg)
    xx = torch.rand(2, 3, 32, 32)
    recon = model.reconstruct(xx)
    error = model(xx)

    assert recon.shape == (2, 3, 32, 32)
    assert error.shape == (2,)

