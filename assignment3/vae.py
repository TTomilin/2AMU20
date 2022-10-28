import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import Module
from torch.nn.functional import one_hot

from assignment3.model import GaussianEncoder, GaussianDecoder, BernoulliDecoder, BetaDecoder, \
    CategoricalDecoder, BaseDecoder, BaseEncoder
from assignment3.utils import log_beta_pdf, kl_loss


class VAE(Module):
    def __init__(self, encoder: BaseEncoder, decoder: BaseDecoder):
        super(VAE, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

    def elbo(self, x):
        raise NotImplementedError

    def reconstruct(self, x):
        raise NotImplementedError

    def forward(self, x):
        z, _, _ = self.encode(x)
        return self.decode(z)

    def sample(self, n_samples: int):
        with torch.no_grad():
            z = torch.randn(n_samples, self.encoder.n_latent, device=self.device)
            return self.decoder.reconstruct(z)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class GaussianVAE(VAE):
    def __init__(self, n_latent: int, n_pixels: int):
        super().__init__(GaussianEncoder(n_latent), GaussianDecoder(n_pixels, n_latent))
        self.n_latent = n_latent

    def elbo(self, x):
        z, mu_z, var_z = self.encode(x)
        x_decoded = self.decode(z)

        recon = x_decoded
        mse = (recon - x.flatten(1))**2
        recon_loss = 0.5 * torch.sum(np.log(self.decoder.var * 2 * np.pi) + mse / self.decoder.var)

        return recon_loss + kl_loss(mu_z, var_z)

    def reconstruct(self, x):
        return self.forward(x)


class CategoricalVAE(VAE):

    def __init__(self, n_latent: int, n_pixels: int, n_bins: int):
        super().__init__(GaussianEncoder(n_latent), CategoricalDecoder(n_pixels, n_latent, n_bins))
        self.n_latent = n_latent

    def elbo(self, x):
        x = torch.floor(x * (self.decoder.n_bins - 1))
        z, mu_z, var_z = self.encode(x)
        pmf = self.decode(z)

        one_hot_x = one_hot(x.flatten(1).to(torch.int64), num_classes=self.decoder.n_bins)
        log_likelihood = torch.sum(torch.log(pmf) * one_hot_x)
        return kl_loss(mu_z, var_z) - log_likelihood

    def sample(self, n_samples: int):
        with torch.no_grad():
            z = torch.randn(n_samples, self.n_latent, device=self.device)
            return self.decoder.reconstruct(z)

    def reconstruct(self, x):
        pmf = self.forward(x)
        dist = Categorical(pmf)
        return dist.sample() / self.decoder.n_bins


class BernoulliVAE(VAE):
    def __init__(self, n_latent: int, n_pixels: int):
        super().__init__(GaussianEncoder(n_latent), BernoulliDecoder(n_pixels, n_latent))
        self.n_latent = n_latent

    def forward(self, x):
        x_binary = torch.bernoulli(x)
        return super().forward(x_binary)

    def elbo(self, x):
        x_binary = torch.bernoulli(x)
        z, mu_z, var_z = self.encode(x_binary)
        pmf = self.decode(z)
        entropy_loss = torch.mean(
            torch.sum(x.flatten(1) * torch.log(pmf) + (1 - x.flatten(1)) * torch.log(1 - pmf), dim=1))
        return kl_loss(mu_z, var_z) - entropy_loss

    def reconstruct(self, x):
        pmf = self.forward(x)
        return torch.bernoulli(pmf)


class BetaVAE(VAE):
    def __init__(self, n_latent: int, n_pixels: int):
        super().__init__(GaussianEncoder(n_latent), BetaDecoder(n_pixels, n_latent))
        self.n_latent = n_latent

    def elbo(self, x):
        z, mu, var = self.encode(x)
        alpha, beta = self.decode(z)

        log_likelihood = torch.sum(log_beta_pdf(x.flatten(1), alpha, beta))
        return kl_loss(mu, var) - log_likelihood

    def reconstruct(self, x):
        alpha, beta = self.forward(x)
        variance = (alpha * beta) / ((alpha + beta).pow(2) * (alpha + beta + 1))
        return alpha / (alpha + beta) + variance * torch.randn_like(variance)
