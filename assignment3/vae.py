from math import sqrt

import numpy as np
import torch
from torch.distributions import Beta
from torch.nn import Module
from torch.nn.functional import gumbel_softmax, binary_cross_entropy

from assignment3.model import GaussianEncoder, GaussianDecoder, BernoulliEncoder, BernoulliDecoder, BetaDecoder, \
    CategoricalDecoder
from assignment3.utils import binomial_kl, log_beta_pdf


class VAE:
    def __init__(self, encoder: Module, decoder: Module, device: str):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device

    def reparameterize(self, *args):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def loss(self, x):
        raise NotImplementedError

    def sample(self, n_samples: int):
        raise NotImplementedError

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class GaussianVAE(VAE):
    def __init__(self, n_latent: int, n_pixels: int, device: str):
        super().__init__(GaussianEncoder(n_latent), GaussianDecoder(n_pixels, n_latent), device)
        self.n_latent = n_latent

    def reparameterize(self, mu, var):
        return mu + torch.sqrt(var) * torch.randn(var.shape, device=self.device)

    def forward(self, x):
        mu_z, var_z = self.encode(x)
        z = self.reparameterize(mu_z, var_z)
        return self.decode(z)

    def loss(self, x):
        mu_z, var_z = self.encode(x)
        z = self.reparameterize(mu_z, var_z)
        x_decoded = self.decode(z)

        recon = x_decoded + sqrt(self.decoder.var) * torch.randn(x.flatten(1).shape[1], device=self.device)
        mse = (recon - x.flatten(1))**2
        recon_loss = 0.5 * torch.sum(np.log(self.decoder.var * 2 * np.pi) + mse / self.decoder.var)

        kl_loss = torch.mean(0.5 * torch.sum(mu_z**2 + var_z**2 - torch.log(var_z**2) - 1, dim=1))
        return recon_loss + kl_loss

    def sample(self, n_samples: int):
        with torch.no_grad():
            z = torch.randn(n_samples, self.n_latent, device=self.device)
            x = self.decode(z)
            return x, z


class CategoricalVAE(VAE):

    def __init__(self, n_latent: int, n_pixels: int, device: str, n_bins=50):
        super().__init__(GaussianEncoder(n_latent), CategoricalDecoder(n_pixels, n_latent, n_bins), device)
        self.n_latent = n_latent

    def reparameterize(self, mu, var):
        return mu + torch.sqrt(var) * torch.randn(var.shape, device=self.device)

    def forward(self, x):
        mu_z, var_z = self.encode(x)
        z = self.reparameterize(mu_z, var_z)
        return self.decode(z)

    def loss(self, x):
        mu_z, var_z = self.encode(x)
        z = self.reparameterize(mu_z, var_z)
        x_decoded = self.decode(z)

        mse = (x_decoded - x.flatten(1))**2
        # TODO FIX THIS
        recon_loss = mse.sum()

        kl_loss = torch.mean(0.5 * torch.sum(mu_z**2 + var_z**2 - torch.log(var_z**2) - 1, dim=1))
        return recon_loss + kl_loss

    def sample(self, n_samples: int):
        with torch.no_grad():
            z = torch.randn(n_samples, self.n_latent, device=self.device)
            x = self.decode(z)
            # m = Categorical(logits=mu)
            # x = m.sample()
            return x, z


class BernoulliVAE(VAE):
    def __init__(self, n_latent: int, n_pixels: int, device: str):
        super().__init__(BernoulliEncoder(n_latent), BernoulliDecoder(n_pixels, n_latent), device)
        self.n_latent = n_latent

    def reparameterize(self, z):
        return gumbel_softmax(z).flatten(1)

    def forward(self, x):
        x_binary = torch.bernoulli(x)
        z = self.encode(x_binary)
        z = self.reparameterize(z)
        return self.decode(z), z

    def loss(self, x):
        x_decoded, z = self.forward(x)
        entropy_loss = binary_cross_entropy(x_decoded, x.flatten(1))
        kl_loss = binomial_kl(z, self.device)
        # loss = entropy_loss + kl_loss
        return entropy_loss

    def sample(self, n_samples: int):
        with torch.no_grad():
            z = torch.randn(n_samples, self.n_latent, device=self.device)
            z = gumbel_softmax(z).flatten(1)
            x = self.decode(z)
            x = torch.bernoulli(x)
            return x, z


class BetaVAE(VAE):
    def __init__(self, n_latent: int, n_pixels: int, device: str):
        super().__init__(GaussianEncoder(n_latent), BetaDecoder(n_pixels, n_latent), device)
        self.n_latent = n_latent

    def reparameterize(self, mu, var):
        return mu + torch.sqrt(var) * torch.randn(var.shape, device=self.device)

    def forward(self, x):
        mu, var = self.encode(x)
        z = self.reparameterize(mu, var)
        return self.decode(z)

    def loss(self, x):
        mu, var = self.encode(x)
        z = self.reparameterize(mu, var)
        alpha, beta = self.decode(z)

        recon_loss = torch.mean(log_beta_pdf(x.flatten(1), alpha, beta))
        kl_loss = torch.mean(0.5 * torch.sum(mu**2 + var**2 - torch.log(var**2) - 1, dim=1))
        return recon_loss + kl_loss

    def sample(self, n_samples: int):
        with torch.no_grad():
            z = torch.randn(n_samples, self.n_latent, device=self.device)
            alpha, beta = self.decode(z)
            beta_dist = Beta(alpha, beta)
            x = beta_dist.sample()
            return x, z
