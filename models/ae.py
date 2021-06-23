"""
ae.py
=====
Autoencoders
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision.utils import make_grid
from models.modules import IsotropicGaussian, IsotropicLaplace


class AE(nn.Module):
    """autoencoder"""
    def __init__(self, encoder, decoder):
        """
        encoder, decoder : neural networks
        """
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.own_optimizer = False

    def forward(self, x):
        z = self.encode(x)
        recon = self.decoder(z)
        return recon

    def encode(self, x):
        z = self.encoder(x)
        return z

    def predict(self, x):
        """one-class anomaly prediction"""
        recon = self(x)
        if hasattr(self.decoder, 'error'):
            predict = self.decoder.error(x, recon)
        else:
            predict = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        return predict

    def predict_and_reconstruct(self, x):
        recon = self(x)
        if hasattr(self.decoder, 'error'):
            recon_err = self.decoder.error(x, recon)
        else:
            recon_err = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        return recon_err, recon

    def validation_step(self, x, **kwargs):
        recon = self(x)
        if hasattr(self.decoder, 'error'):
            predict = self.decoder.error(x, recon)
        else:
            predict = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        loss = predict.mean()

        if kwargs.get('show_image', True):
            x_img = make_grid(x.detach().cpu(), nrow=10, range=(0, 1))
            recon_img = make_grid(recon.detach().cpu(), nrow=10, range=(0, 1))
        else:
            x_img, recon_img = None, None
        return {'loss': loss.item(), 'predict': predict, 'reconstruction': recon,
                'input@': x_img, 'recon@': recon_img}

    def train_step(self, x, optimizer, clip_grad=None, **kwargs):
        optimizer.zero_grad()
        recon_error = self.predict(x)
        loss = recon_error.mean()
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
        optimizer.step()
        return {'loss': loss.item()}

    def reconstruct(self, x):
        return self(x)

    def sample(self, N, z_shape=None, device='cpu'):
        if z_shape is None:
            z_shape = self.encoder.out_shape

        rand_z = torch.rand(N, *z_shape).to(device) * 2 - 1
        sample_x = self.decoder(rand_z)
        return sample_x



def clip_vector_norm(x, max_norm):
    norm = x.norm(dim=-1, keepdim=True)
    x = x * ((norm < max_norm).to(torch.float) + (norm > max_norm).to(torch.float) * max_norm/norm + 1e-6)
    return x


class DAE(AE):
    """denoising autoencoder"""
    def __init__(self, encoder, decoder, sig=0.0, noise_type='gaussian'):
        super(DAE, self).__init__(encoder, decoder)
        self.sig = sig
        self.noise_type = noise_type

    def train_step(self, x, optimizer, y=None):
        optimizer.zero_grad()
        if self.noise_type == 'gaussian':
            noise = torch.randn(*x.shape, dtype=torch.float32)
            noise = noise.to(x.device)
            recon = self(x + self.sig * noise)
        elif self.noise_type == 'saltnpepper':
            x = self.salt_and_pepper(x)
            recon = self(x)
        else:
            raise ValueError(f'Invalid noise_type: {self.noise_type}')

        loss = torch.mean((recon - x) ** 2)
        loss.backward()
        optimizer.step()
        return {'loss': loss.item()}

    def salt_and_pepper(self, img):
        """salt and pepper noise for mnist"""
        # for salt and pepper noise, sig is probability of occurance of noise pixels.
        img = img.copy()
        prob = self.sig
        rnd = torch.random.rand(*img.shape).to(img.device)
        img[rnd < prob / 2] = 0.
        img[rnd > 1 - prob / 2] = 1.
        return img


class VAE(AE):
    def __init__(self, encoder, decoder, n_sample=1, use_mean=False, pred_method='recon', sigma_trainable=False):
        super(VAE, self).__init__(encoder, IsotropicGaussian(decoder, sigma=1, sigma_trainable=sigma_trainable))
        self.n_sample = n_sample  # the number of samples to generate for anomaly detection
        self.use_mean = use_mean  # if True, does not sample from posterior distribution
        self.pred_method = pred_method  # which anomaly score to use
        self.z_shape = None
        
    def forward(self, x):
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        return self.decoder(z_sample)

    def sample_latent(self, z):
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        if self.use_mean:
            return mu
        eps = torch.randn(*mu.shape, dtype=torch.float32)
        eps = eps.to(z.device)
        return mu + torch.exp(log_sig) * eps

    # def sample_marginal_latent(self, z_shape):
    #     return torch.randn(z_shape)

    def kl_loss(self, z):
        """analytic (positive) KL divergence between gaussians
        KL(q(z|x) | p(z))"""
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        mu_sq = mu ** 2
        sig_sq = torch.exp(log_sig) ** 2
        kl = mu_sq + sig_sq - torch.log(sig_sq) - 1
        # return 0.5 * torch.mean(kl.view(len(kl), -1), dim=1)
        return 0.5 * torch.sum(kl.view(len(kl), -1), dim=1)

    def train_step(self, x, optimizer, y=None, clip_grad=None):
        optimizer.zero_grad()
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        nll = - self.decoder.log_likelihood(x, z_sample)

        kl_loss = self.kl_loss(z)
        loss = nll + kl_loss
        loss = loss.mean()
        nll = nll.mean()

        loss.backward()
        optimizer.step()
        return {'loss': nll.item(), 'vae/kl_loss_': kl_loss.mean(), 'vae/sigma_': self.decoder.sigma.item()}

    def predict(self, x):
        """one-class anomaly prediction using the metric specified by self.anomaly_score"""
        if self.pred_method == 'recon':
            return self.reconstruction_probability(x)
        elif self.pred_method == 'lik':
            return  - self.marginal_likelihood(x)  # negative log likelihood
        else:
            raise ValueError(f'{self.pred_method} should be recon or lik')

    def validation_step(self, x, y=None, **kwargs):
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        recon = self.decoder(z_sample)
        loss = torch.mean((recon - x) ** 2)
        predict = - self.decoder.log_likelihood(x, z_sample)
        
        if kwargs.get('show_image', True):
            x_img = make_grid(x.detach().cpu(), nrow=10, range=(0, 1))
            recon_img = make_grid(recon.detach().cpu(), nrow=10, range=(0, 1))
        else:
            x_img, recon_img = None, None
            
        return {'loss': loss.item(), 'predict': predict, 'reconstruction': recon,
                'input@': x_img, 'recon@': recon_img}

    def reconstruction_probability(self, x):
        l_score = []
        z = self.encoder(x)
        for i in range(self.n_sample):
            z_sample = self.sample_latent(z)
            recon_loss = - self.decoder.log_likelihood(x, z_sample)
            score = recon_loss
            l_score.append(score)
        return torch.stack(l_score).mean(dim=0)

    def marginal_likelihood(self, x, n_sample=None):
        """marginal likelihood from importance sampling
        log P(X) = log \int P(X|Z) * P(Z)/Q(Z|X) * Q(Z|X) dZ"""
        if n_sample is None:
            n_sample = self.n_sample

        # check z shape
        with torch.no_grad():
            z = self.encoder(x)

            l_score = []
            for i in range(n_sample):
                z_sample = self.sample_latent(z)
                log_recon = self.decoder.log_likelihood(x, z_sample)
                log_prior = self.log_prior(z_sample)
                log_posterior = self.log_posterior(z, z_sample)
                l_score.append(log_recon + log_prior - log_posterior)
        score = torch.stack(l_score)
        logN = torch.log(torch.tensor(n_sample, dtype=torch.float, device=x.device))
        return torch.logsumexp(score, dim=0) - logN

    def marginal_likelihood_naive(self, x, n_sample=None):
        if n_sample is None:
            n_sample = self.n_sample

        # check z shape
        z_dummy = self.encoder(x[[0]])
        z = torch.zeros(len(x), *list(z_dummy.shape[1:]), dtype=torch.float).to(x.device)

        l_score = []
        for i in range(n_sample):
            z_sample = self.sample_latent(z)
            recon_loss = - self.decoder.log_likelihood(x, z_sample)
            score = recon_loss
            l_score.append(score)
        score = torch.stack(l_score)
        return - torch.logsumexp(-score, dim=0)

    def elbo(self, x):
        l_score = []
        z = self.encoder(x)
        for i in range(self.n_sample):
            z_sample = self.sample_latent(z)
            recon_loss = - self.decoder.log_likelihood(x, z_sample)
            kl_loss = self.kl_loss(z)
            score = recon_loss + kl_loss
            l_score.append(score)
        return torch.stack(l_score).mean(dim=0)

    def log_posterior(self, z, z_sample):
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]

        log_p = torch.distributions.Normal(mu, torch.exp(log_sig)).log_prob(z_sample)
        log_p = log_p.view(len(z), -1).sum(-1)
        return log_p

    def log_prior(self, z_sample):
        log_p = torch.distributions.Normal(torch.zeros_like(z_sample), torch.ones_like(z_sample)).log_prob(z_sample)
        log_p = log_p.view(len(z_sample), -1).sum(-1)
        return log_p

    def posterior_entropy(self, z):
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        D = mu.shape[1]
        pi = torch.tensor(np.pi, dtype=torch.float32).to(z.device)
        term1 = D / 2
        term2 = D / 2 * torch.log(2 * pi)
        term3 = log_sig.view(len(log_sig), -1).sum(dim=-1)
        return term1 + term2 + term3

    def _set_z_shape(self, x):
        if self.z_shape is not None:
            return
        with torch.no_grad():
            dummy_z = self.encode(x[[0]])
        dummy_z = self.sample_latent(dummy_z)
        z_shape = dummy_z.shape
        self.z_shape = z_shape[1:]

    def sample_z(self, n_sample, device):
        z_shape = (n_sample,) + self.z_shape
        return torch.randn(z_shape, device=device, dtype=torch.float)

    def sample(self, n_sample, device):
        z = self.sample_z(n_sample, device)
        return {'sample_x': self.decoder.sample(z)}


class WAE(AE):
    """Wassertstein Autoencoder with MMD loss"""
    def __init__(self, encoder, decoder, reg=1., bandwidth='median', prior='gaussian'):
        super().__init__(encoder, decoder)
        if not isinstance(bandwidth, str):
            bandwidth = float(bandwidth)
        self.bandwidth = bandwidth
        self.reg = reg  # coefficient for MMD loss
        self.prior = prior

    def train_step(self, x, optimizer, y=None, **kwargs):
        optimizer.zero_grad()
        # forward step
        z = self.encoder(x)
        recon = self.decoder(z)
        # recon_loss = torch.mean(self.decoder.square_error(x, recon))
        recon_loss = torch.mean((x - recon) ** 2)

        # MMD step
        z_prior = self.sample_prior(z)
        mmd_loss = self.mmd(z_prior, z)

        loss = recon_loss + mmd_loss * self.reg
        loss.backward()
        optimizer.step()
        return {'loss': loss.item(), 'recon_loss': recon_loss, 'mmd_loss': mmd_loss.item()}

    def sample_prior(self, z):
        if self.prior == 'gaussian':
            return torch.randn_like(z)
        elif self.prior == 'uniform_tanh':
            return torch.rand_like(z) * 2 - 1
        else:
            raise ValueError(f'invalid prior {self.prior}')

    def mmd(self, X1, X2):
        if len(X1.shape) == 4:
            X1 = X1.view(len(X1), -1)
        if len(X2.shape) == 4:
            X2 = X2.view(len(X2), -1)

        N1 = len(X1)
        X1_sq = X1.pow(2).sum(1).unsqueeze(0)
        X1_cr = torch.mm(X1, X1.t())
        X1_dist = X1_sq + X1_sq.t() - 2 * X1_cr

        N2 = len(X2)
        X2_sq = X2.pow(2).sum(1).unsqueeze(0)
        X2_cr = torch.mm(X2, X2.t())
        X2_dist = X2_sq + X2_sq.t() - 2 * X2_cr

        X12 = torch.mm(X1, X2.t())
        X12_dist = X1_sq.t() + X2_sq - 2 * X12

        # median heuristic to select bandwidth
        if self.bandwidth == 'median':
            X1_triu = X1_dist[torch.triu(torch.ones_like(X1_dist), diagonal=1) == 1]
            bandwidth1 = torch.median(X1_triu)
            X2_triu = X2_dist[torch.triu(torch.ones_like(X2_dist), diagonal=1) == 1]
            bandwidth2 = torch.median(X2_triu)
            bandwidth_sq = ((bandwidth1 + bandwidth2) / 2).detach()
        else:
            bandwidth_sq = (self.bandwidth ** 2)

        C = - 0.5 / bandwidth_sq
        K11 = torch.exp(C * X1_dist)
        K22 = torch.exp(C * X2_dist)
        K12 = torch.exp(C * X12_dist)
        K11 = (1 - torch.eye(N1).to(X1.device)) * K11
        K22 = (1 - torch.eye(N2).to(X1.device)) * K22
        mmd = K11.sum() / N1 / (N1 - 1) + K22.sum() / N2 / (N2 - 1) - 2 * K12.mean()
        return mmd



