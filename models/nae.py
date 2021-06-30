import numpy as np
import torch
from torch import nn
from torchvision.utils import make_grid 
from models.ae import AE
from models.modules import IsotropicLaplace, DummyDistribution
from models.energybased import SampleBuffer, SampleBufferV2
from models.langevin import sample_langevin, sample_langevin_v2

class FFEBM(nn.Module):
    """feed-forward energy-based model"""
    def __init__(self, net, x_step=None, x_stepsize=None, x_noise_std=None, x_noise_anneal=None,
                 x_bound=None, x_clip_langevin_grad=None, l2_norm_reg=None,
                 buffer_size=10000, replay_ratio=0.95, replay=True, gamma=1, sampling='x',
                 initial_dist='gaussian', temperature=1., temperature_trainable=False,
                 mh=False, reject_boundary=False):
        super().__init__()
        self.net = net

        self.x_bound = x_bound
        self.l2_norm_reg = l2_norm_reg
        self.gamma = gamma
        self.sampling = sampling

        self.x_step = x_step
        self.x_stepsize = x_stepsize
        self.x_noise_std = x_noise_std
        self.x_noise_anneal = x_noise_anneal
        self.x_bound = x_bound
        self.x_clip_langevin_grad = x_clip_langevin_grad
        self.mh = mh
        self.reject_boundary = reject_boundary

        self.buffer_size = buffer_size
        self.replay_ratio = replay_ratio
        self.replay = replay

        self.buffer = SampleBufferV2(max_samples=buffer_size, replay_ratio=replay_ratio)

        self.x_shape = None
        self.initial_dist = initial_dist
        temperature = np.log(temperature)
        self.temperature_trainable = temperature_trainable
        if temperature_trainable:
            self.register_parameter('temperature_', nn.Parameter(torch.tensor(temperature, dtype=torch.float)))
        else:
            self.register_buffer('temperature_', torch.tensor(temperature, dtype=torch.float))

    @property
    def temperature(self):
        return torch.exp(self.temperature_)

    @property
    def sample_shape(self):
        return self.x_shape

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        return self.forward(x)

    def energy(self, x):
        return self.forward(x)

    def energy_T(self,x):
        return self.energy(x) / self.temperature

    def sample(self, x0=None, n_sample=None, device=None, replay=None):
        """sampling factory function. takes either x0 or n_sample and device
        """
        if x0 is not None:
            n_sample = len(x0)
            device = x0.device
        if replay is None:
            replay = self.replay

        if self.sampling == 'x':
            return self.sample_x(n_sample, device, replay=replay)
        elif self.sampling == 'cd':
            return self.sample_x(n_sample, device, x0=x0, replay=False)
        elif self.sampling == 'on_manifold':
            return self.sample_omi(n_sample, device, replay=replay)

    def sample_x(self, n_sample=None, device=None, x0=None, replay=False):
        if x0 is None:
            x0 = self.initial_sample(n_sample, device=device)
        d_sample_result = sample_langevin_v2(x0.detach(), self.energy, stepsize=self.x_stepsize, n_steps=self.x_step,
                                        noise_scale=self.x_noise_std,
                                        clip_x=self.x_bound, noise_anneal=self.x_noise_anneal,
                                        clip_grad=self.x_clip_langevin_grad, spherical=False,
                                        mh=self.mh, temperature=self.temperature, reject_boundary=self.reject_boundary)
        sample_result = d_sample_result['sample']
        if replay:
            self.buffer.push(sample_result)
        d_sample_result['sample_x'] = sample_result
        d_sample_result['sample_x0'] = x0
        return d_sample_result

    def initial_sample(self, n_samples, device):
        l_sample = []
        if not self.replay or len(self.buffer) == 0:
            n_replay = 0
        else:
            n_replay = (np.random.rand(n_samples) < self.replay_ratio).sum()
            l_sample.append(self.buffer.get(n_replay))

        shape = (n_samples - n_replay,) + self.sample_shape
        if self.initial_dist == 'gaussian':
            x0_new = torch.randn(shape, dtype=torch.float)
        elif self.initial_dist == 'uniform':
            x0_new = torch.rand(shape, dtype=torch.float)
            if self.sampling != 'on_manifold' and self.x_bound is not None:
                x0_new = x0_new * (self.x_bound[1] - self.x_bound[0]) + self.x_bound[0]
            elif self.sampling == 'on_manifold' and self.z_bound is not None:
                x0_new = x0_new * (self.z_bound[1] - self.z_bound[0]) + self.z_bound[0]

        l_sample.append(x0_new)
        return torch.cat(l_sample).to(device)

    def _set_x_shape(self, x):
        if self.x_shape is not None:
            return
        self.x_shape = x.shape[1:]

    def weight_norm(self, net):
        norm = 0
        for param in net.parameters():
            norm += (param ** 2).sum()
        return norm

    def train_step(self, x, opt):
        self._set_x_shape(x)

        # negative sample
        d_sample = self.sample(x)
        x_neg = d_sample['sample_x']

        opt.zero_grad()
        neg_e = self.energy(x_neg)

        # ae recon pass
        pos_e = self.energy(x)

        loss = (pos_e.mean() - neg_e.mean()) / self.temperature

        if self.gamma is not None:
            loss += self.gamma * (pos_e ** 2 + neg_e ** 2).mean()

        # weight regularization
        l2_norm = self.weight_norm(self.net)
        if self.l2_norm_reg is not None:
            loss = loss + self.l2_norm_reg * l2_norm

        loss.backward()
        opt.step()

        d_result = {'pos_e': pos_e.mean().item(), 'neg_e': neg_e.mean().item(),
                    'x_neg': x_neg.detach().cpu(), 'x_neg_0': d_sample['sample_x0'].detach().cpu(),
                    'loss': loss.item(), 'sample': x_neg.detach().cpu(),
                    'l2_norm': l2_norm.item()}
        return d_result

    def validation_step(self, x, y=None):
        pos_e = self.energy(x)
        loss = pos_e.mean().item()
        predict = pos_e.detach().cpu().flatten()
        return {'loss': pos_e, 'predict':predict}



class NAE(FFEBM):
    """Normalized Autoencoder"""
    def __init__(self, encoder, decoder,
                 z_step=50, z_stepsize=0.2, z_noise_std=0.2, z_noise_anneal=None,
                 x_step=50, x_stepsize=10, x_noise_std=0.05, x_noise_anneal=None,
                 x_bound=(0, 1), z_bound=None,
                 z_clip_langevin_grad=None, x_clip_langevin_grad=None, 
                 l2_norm_reg=None, l2_norm_reg_en=None, spherical=True, z_norm_reg=None,
                 buffer_size=10000, replay_ratio=0.95, replay=True,
                 gamma=None, sampling='on_manifold',
                 temperature=1., temperature_trainable=True,
                 initial_dist='gaussian', 
                 mh=False, mh_z=False, reject_boundary=False, reject_boundary_z=False):
        """
        encoder: An encoder network, an instance of nn.Module.
        decoder: A decoder network, an instance of nn.Module.

        **Sampling Parameters**
        sampling: Sampling methods.
                  'on_manifold' - on-manifold initialization.
                  'cd' - Contrastive Divergence.
                  'x' - Persistent CD.

        z_step: The number of steps in latent chain.
        z_stepsize: The step size of latent chain
        z_noise_std: The standard deviation of noise in latent chain
        z_noise_anneal: Noise annealing parameter in latent chain. If None, no annealing.
        mh_z: If True, use Metropolis-Hastings rejection in latent chain.
        z_clip_langevin_grad: Clip the norm of gradient in latent chain.
        z_bound: [z_min, z_max]

        x_step: The number of steps in visible chain.
        x_stepsize: The step size of visible chain
        x_noise_std: The standard deviation of noise in visible chain
        x_noise_anneal: Noise annealing parameter in visible chain. If None, no annealing.
        mh: If True, use Metropolis-Hastings rejection in latent chain.
        x_clip_langevin_grad: Clip the norm of gradient in visible chain.
        x_bouond: [x_min, x_bound]. 

        replay: Whether to use the replay buffer.
        buffer_size: The size of replay buffer.
        replay_ratio: The probability of applying persistent CD. A chain is re-initialized with the probability of
                      (1 - replay_ratio).
        initial_dist: The distribution from which initial samples are generated.
                      'Gaussian' or 'uniform'



        **Regularization Parameters**
        gamma: The coefficient for regularizing the negative sample energy.
        l2_norm_reg: The coefficient for L2 norm of decoder weights.
        l2_norm_reg_en: The coefficient for L2 norm of encoder weights.
        z_norm_reg: The coefficient for regularizing the L2 norm of Z vector.


        """
        super(NAE, self).__init__(net=None, x_step=x_step, x_stepsize=x_stepsize, x_noise_std=x_noise_std,
                                  x_noise_anneal=x_noise_anneal, x_bound=x_bound,
                                  x_clip_langevin_grad=x_clip_langevin_grad, l2_norm_reg=l2_norm_reg,
                                  buffer_size=buffer_size, replay_ratio=replay_ratio, replay=replay,
                                  gamma=gamma, sampling=sampling, initial_dist=initial_dist,
                                  temperature=temperature, temperature_trainable=temperature_trainable,
                                  mh=mh, reject_boundary=reject_boundary)
        self.encoder = encoder
        self.decoder = DummyDistribution(decoder)
        self.z_step = z_step
        self.z_stepsize = z_stepsize
        self.z_noise_std = z_noise_std
        self.z_noise_anneal = z_noise_anneal
        self.z_clip_langevin_grad = z_clip_langevin_grad
        self.mh_z = mh_z
        self.reject_boundary_z = reject_boundary_z

        self.z_bound = z_bound
        self.l2_norm_reg = l2_norm_reg  # decoder
        self.l2_norm_reg_en = l2_norm_reg_en
        self.spherical = spherical
        self.sampling = sampling
        self.z_norm_reg = z_norm_reg

        self.z_shape = None
        self.x_shape = None

    @property
    def sample_shape(self):
        if self.sampling == 'on_manifold':
            return self.z_shape
        else:
            return self.x_shape

    def error(self, x, recon):
        """L2 error"""
        return ((x - recon) ** 2).view((x.shape[0], -1)).sum(dim=1)

    def forward(self, x):
        """ Computes error per dimension """
        D = np.prod(x.shape[1:])
        z = self.encode(x)
        recon = self.decoder(z)
        return self.error(x, recon) / D

    def energy_with_z(self, x):
        D = np.prod(x.shape[1:])
        z = self.encode(x)
        recon = self.decoder(z)
        return self.error(x, recon) / D, z

    def normalize(self, z):
        """normalize to unit length"""
        if self.spherical:
            if len(z.shape) == 4:
                z = z / z.view(len(z), -1).norm(dim=-1)[:, None, None, None]
            else:
                z = z / z.view(len(z), -1).norm(dim=1, keepdim=True)
            return z
        else:
            return z

    def encode(self, x):
        if self.spherical:
            return self.normalize(self.encoder(x))
        else:
            return self.encoder(x)

    def sample_omi(self, n_sample, device, replay=False):
        """using on-manifold initialization"""
        # Step 1: On-manifold initialization: LMC on Z space 
        z0 = self.initial_sample(n_sample, device)
        if self.spherical:
            z0 = self.normalize(z0)
        d_sample_z = self.sample_z(z0=z0, replay=replay)
        sample_z = d_sample_z['sample']

        sample_x_1 = self.decoder(sample_z).detach()
        if self.x_bound is not None:
            sample_x_1.clamp_(self.x_bound[0], self.x_bound[1])

        # Step 2: LMC on X space
        d_sample_x = self.sample_x(x0=sample_x_1, replay=False)
        sample_x_2 = d_sample_x['sample_x']
        return {'sample_x': sample_x_2, 'sample_z': sample_z.detach(), 'sample_x0': sample_x_1, 'sample_z0': z0.detach()} 

    def sample_z(self, n_sample=None, device=None, replay=False, z0=None):
        if z0 is None:
            z0 = self.initial_sample(n_sample, device)
        energy = lambda z: self.energy(self.decoder(z))
        d_sample_result = sample_langevin_v2(z0, energy, stepsize=self.z_stepsize, n_steps=self.z_step,
                                             noise_scale=self.z_noise_std,
                                             clip_x=self.z_bound, clip_grad=self.z_clip_langevin_grad,
                                             spherical=self.spherical, mh=self.mh_z,
                                             temperature=self.temperature, reject_boundary=self.reject_boundary_z)
        sample_z = d_sample_result['sample']
        if replay:
            self.buffer.push(sample_z)
        return d_sample_result 

    def _set_z_shape(self, x):
        if self.z_shape is not None:
            return
        # infer z_shape by computing forward
        with torch.no_grad():
            dummy_z = self.encode(x[[0]])
        z_shape = dummy_z.shape
        self.z_shape = z_shape[1:]

    def reconstruct(self, x):
        z = self.encode(x)
        return self.decoder(z)

    def train_step_ae(self, x, opt, clip_grad=None):
        opt.zero_grad()
        z = self.encode(x)
        recon = self.decoder(z)
        z_norm = (z ** 2).mean()
        x_dim = np.prod(x.shape[1:])
        recon_error = self.error(x, recon).mean() / x_dim
        loss = recon_error

        # weight regularization
        decoder_norm = self.weight_norm(self.decoder)
        encoder_norm = self.weight_norm(self.encoder)
        if self.l2_norm_reg is not None:
            loss = loss + self.l2_norm_reg * decoder_norm
        if self.l2_norm_reg_en is not None:
            loss = loss + self.l2_norm_reg_en * encoder_norm
        loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], max_norm=clip_grad)
        opt.step()
        d_result = {'loss': loss.item(), 'z_norm': z_norm.item(), 'recon_error_': recon_error.item(),
                    'decoder_norm_': decoder_norm.item(), 'encoder_norm_': encoder_norm.item()}
        return d_result

    def train_step(self, x, opt):
        self._set_z_shape(x)
        self._set_x_shape(x)

        # negative sample
        d_sample = self.sample(x)
        x_neg = d_sample['sample_x']

        opt.zero_grad()
        neg_e, neg_z = self.energy_with_z(x_neg)

        # ae recon pass
        pos_e, pos_z = self.energy_with_z(x)

        loss = (pos_e.mean() - neg_e.mean()) 

        if self.temperature_trainable:
            loss = loss + (pos_e.mean() - neg_e.mean()).detach() / self.temperature


        # regularizing negative sample energy
        if self.gamma is not None:
            gamma_term = ((neg_e) ** 2).mean()
            loss += self.gamma * gamma_term

        # weight regularization
        decoder_norm = self.weight_norm(self.decoder)
        encoder_norm = self.weight_norm(self.encoder)
        if self.l2_norm_reg is not None:
            loss = loss + self.l2_norm_reg * decoder_norm
        if self.l2_norm_reg_en is not None:
            loss = loss + self.l2_norm_reg_en * encoder_norm
        if self.z_norm_reg is not None:
            z_norm = (torch.cat([pos_z, neg_z]) ** 2).mean()
            loss = loss + self.z_norm_reg * z_norm

        loss.backward()
        opt.step()

        # for debugging
        x_neg_0 = d_sample['sample_x0']
        neg_e_x0 = self.energy(x_neg_0)  # energy of samples from latent chain
        recon_neg = self.reconstruct(x_neg)
        d_result = {'pos_e': pos_e.mean().item(), 'neg_e': neg_e.mean().item(),
                    'x_neg': x_neg.detach().cpu(), 'recon_neg': recon_neg.detach().cpu(),
                    'loss': loss.item(), 'sample': x_neg.detach().cpu(),
                    'decoder_norm': decoder_norm.item(), 'encoder_norm': encoder_norm.item(),
                    'neg_e_x0': neg_e_x0.mean().item(), 'x_neg_0': x_neg_0.detach().cpu(),
                    'temperature': self.temperature.item(), 
                    'pos_z': pos_z.detach().cpu(), 'neg_z': neg_z.detach().cpu()}
        if self.gamma is not None:
            d_result['gamma_term'] = gamma_term.item()
        if 'sample_z0' in d_sample:
            x_neg_z0 = self.decoder(d_sample['sample_z0'])
            d_result['neg_e_z0'] = self.energy(x_neg_z0).mean().item()
        return d_result

    def validation_step(self, x, y=None):
        z = self.encode(x)
        recon = self.decoder(z)
        energy = self.error(x, recon)
        loss = energy.mean().item()
        recon_img = make_grid(recon.detach().cpu(), nrow=10, range=(0, 1))
        input_img = make_grid(x.detach().cpu(), nrow=10, range=(0, 1))
        return {'loss': loss, 'pos_e': loss, 'recon@': recon_img, 'input@': input_img}

