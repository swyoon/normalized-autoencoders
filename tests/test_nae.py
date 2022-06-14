import pytest
import torch
from torch.optim import Adam
from models.modules import FCNet
from models.nae import NAE, FFEBM, FFEBMV2, NAE_L2_CD, NAE_L2_NCE, NAE_L2_OMI
from models import get_model
from models.mcmc import LangevinSampler, MHSampler, NoiseSampler
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


@pytest.mark.parametrize('sampling', ['cd', 'x'])
@pytest.mark.parametrize('sampler', ['mh', 'langevin'])
def test_ffebm_v2(sampling, sampler):
    net = FCNet(2, 1, out_activation='linear')
    if sampler == 'mh':
        sampler = MHSampler(n_step=2, stepsize=0.1, bound=(-5, 5), buffer_size=10000, replay_ratio=0.95, reject_boundary=False,
                            mh=True, initial_dist='uniform')
    elif sampler == 'langevin':
        sampler = LangevinSampler(n_step=2, stepsize=0.1, noise_std=0.1, noise_anneal=1.,
                                  bound=(-5, 5), buffer_size=10000, replay_ratio=0.95, reject_boundary=False,
                                  mh=True, initial_dist='uniform')
    model = FFEBMV2(net, sampler, gamma=1, sampling=sampling)
    X = torch.randn((10, 2), dtype=torch.float)
    opt = Adam(model.parameters(), lr=1e-4)

    # forward
    lik = model.predict(X)

    # sample
    model._set_x_shape(X)
    d_sample = model.sample(n_sample=10, device='cpu:0', sampling='x')

    # training
    model.train_step(X, opt)


@pytest.mark.parametrize('sampling', ['cd', 'x'])
def test_nae_l2_cd(sampling):
    encoder = FCNet(2, 1, out_activation='spherical')
    decoder = FCNet(1, 2)
    sampler = LangevinSampler(n_step=2, stepsize=0.1, noise_std=0.1, noise_anneal=1.,
                              bound=(-5, 5), buffer_size=10000, replay_ratio=0.95, reject_boundary=False,
                              mh=True, initial_dist='uniform')
    nae = NAE_L2_CD(encoder, decoder, sampler, sampling=sampling)
    opt = Adam(nae.parameters(), lr=1e-4)

    X = torch.randn((10, 2), dtype=torch.float)

    # forward
    lik = nae.predict(X)

    # sample
    nae._set_x_shape(X)
    # nae._set_z_shape(X)
    d_sample = nae.sample(X)

    # training
    nae.train_step(X, opt)


def test_nae_l2_nce():
    encoder = FCNet(2, 1, out_activation='spherical')
    decoder = FCNet(1, 2)
    sampler = NoiseSampler(dist='gaussian', shape=(2,), offset=1., scale=1.)
    nae = NAE_L2_NCE(encoder, decoder, sampler, T=0.5, T_trainable=True)
    opt = Adam(nae.parameters(), lr=1e-4)

    X = torch.randn((10, 2), dtype=torch.float)

    # forward
    lik = nae.predict(X)

    # training
    nae.train_step(X, opt)


@pytest.mark.parametrize('spherical', [True, False])
def test_nae_l2_omi(spherical):
    encoder = FCNet(2, 3, out_activation='linear')
    decoder = FCNet(3, 2)
    sampler_x = LangevinSampler(n_step=2, stepsize=0.1, noise_std=0.1, noise_anneal=1.,
                                bound=(-5, 5), buffer_size=10000, replay_ratio=0.,
                                reject_boundary=False, mh=True, initial_dist='uniform')

    sampler_z = LangevinSampler(n_step=2, stepsize=0.1, noise_std=0.1, noise_anneal=None,
                                bound=(-5, 5), buffer_size=10000, replay_ratio=0.95,
                                reject_boundary=False, mh=True, initial_dist='uniform')
 
    nae = NAE_L2_OMI(encoder, decoder, sampler_z, sampler_x, T=0.5, T_trainable=True, spherical=spherical)
    opt = Adam(nae.parameters(), lr=1e-4)

    X = torch.randn((10, 2), dtype=torch.float)

    # forward
    lik = nae.predict(X)

    # training
    nae.train_step(X, opt)


