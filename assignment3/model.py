from math import sqrt

import torch
from torch import relu
from torch.distributions import Categorical
from torch.nn import Conv2d, Linear, BatchNorm2d, ConvTranspose2d, ELU, Softplus, Module, Sigmoid, Softmax, ModuleList


class BaseEncoder(Module):
    def __init__(self, n_latent: int, in_channels: int, n_conv_blocks: int, n_filters: int):
        super(BaseEncoder, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_latent = n_latent
        conv_head = []
        for i in range(n_conv_blocks):
            conv_head.append(Conv2d(in_channels, n_filters, kernel_size=3, stride=2, padding=1))
            conv_head.append(BatchNorm2d(n_filters))
            conv_head.append(ELU(inplace=True))
            in_channels = n_filters
            n_filters *= 2
        self.conv_head = ModuleList(conv_head)

    def to_latent(self, flattened):
        raise NotImplementedError

    def forward(self, input_x):
        res = input_x
        for layer in self.conv_head:
            res = layer(res)
        res = res.flatten(1)
        return self.to_latent(res)


class GaussianEncoder(BaseEncoder):
    def __init__(self, n_latent: int, in_channels=1, n_conv_blocks=3, n_filters=32, n_fc=2048):
        super(GaussianEncoder, self).__init__(n_latent, in_channels, n_conv_blocks, n_filters)

        # Mean and variance layers
        self.mu = Linear(n_fc, n_latent)
        self.var = Linear(n_fc, n_latent)
        self.var_act = Softplus()

    def reparameterize(self, mu, var):
        return mu + torch.sqrt(var) * torch.randn(var.shape, device=self.device)

    def to_latent(self, flattened):
        mu = self.mu(flattened)
        var = self.var_act(self.var(flattened))
        z = self.reparameterize(mu, var)
        return z, mu, var


class BaseDecoder(Module):
    def __init__(self, n_latent: int, n_deconv_blocks: int, in_channels: int, n_fc: int):
        super(BaseDecoder, self).__init__()
        self.in_channels = in_channels
        self.filter_size = int(sqrt(n_fc / in_channels))
        self.dense = Linear(n_latent, n_fc)

        de_conv_head = []
        for i in range(n_deconv_blocks):
            # Don't use output padding for the first block
            out_padding = 0 if i == 0 else 1
            # The last block should output a single channel
            out_channels = 1 if i == n_deconv_blocks - 1 else in_channels // 2
            de_conv_head.append(ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                                output_padding=out_padding))
            de_conv_head.append(BatchNorm2d(out_channels))
            de_conv_head.append(ELU(inplace=True))
            in_channels = out_channels
        self.de_conv_head = ModuleList(de_conv_head)

    def output(self, deconvoluted):
        raise NotImplementedError

    def reconstruct(self, z):
        raise NotImplementedError

    def forward(self, z):
        # First run the latent vector through a dense layer
        res = relu(self.dense(z))
        # Create a suitable shape for the de-convolution
        res = res.reshape(-1, self.in_channels, self.filter_size, self.filter_size)
        for layer in self.de_conv_head:
            res = layer(res)
        res = res.flatten(1)
        return self.output(res)


class GaussianDecoder(BaseDecoder):

    def __init__(self, n_pixels: int, n_latent: int, n_deconv_blocks=3, in_channels=128, n_fc=2048, var=0.05):
        super(GaussianDecoder, self).__init__(n_latent, n_deconv_blocks, in_channels, n_fc)
        self.n_latent = n_latent
        self.in_channels = in_channels
        self.filter_size = int(sqrt(n_fc / in_channels))
        self.var = var

        # Generate the output layer
        self.mu = Linear(n_pixels, n_pixels)

    def output(self, deconvoluted):
        return self.mu(deconvoluted)

    def reconstruct(self, z):
        return self.forward(z)


class CategoricalDecoder(BaseDecoder):
    def __init__(self, n_pixels: int, n_latent: int, n_bins: int, n_deconv_blocks=3, in_channels=128, n_fc=2048):
        super(CategoricalDecoder, self).__init__(n_latent, n_deconv_blocks, in_channels, n_fc)
        self.n_bins = n_bins
        self.n_pixels = n_pixels
        self.n_latent = n_latent

        self.out = Linear(n_pixels, n_pixels * n_bins)
        self.act = Softmax(dim=2)

    def output(self, deconvoluted):
        out = self.out(deconvoluted)
        out = out.reshape(-1, self.n_pixels, self.n_bins)
        out = self.act(out)
        return out

    def reconstruct(self, z):
        pmf = self.forward(z)
        dist = Categorical(pmf)
        return dist.sample() / self.n_bins


class BernoulliDecoder(BaseDecoder):
    def __init__(self, n_pixels: int, n_latent: int, n_deconv_blocks=3, in_channels=128, n_fc=2048, var=0.05):
        super(BernoulliDecoder, self).__init__(n_latent, n_deconv_blocks, in_channels, n_fc)
        self.n_latent = n_latent
        self.out = Linear(n_pixels, n_pixels)
        self.out_act = Sigmoid()

    def output(self, deconvoluted):
        return self.out_act(self.out(deconvoluted))

    def reconstruct(self, z):
        pmf = self.forward(z)
        return torch.bernoulli(pmf)


class BetaDecoder(BaseDecoder):
    def __init__(self, n_pixels: int, n_latent: int, n_deconv_blocks=3, in_channels=128, n_fc=2048):
        super(BetaDecoder, self).__init__(n_latent, n_deconv_blocks, in_channels, n_fc)
        self.n_pixels = n_pixels
        self.n_latent = n_latent

        self.alpha = Linear(n_pixels, n_pixels)
        self.beta = Linear(n_pixels, n_pixels)
        self.act = Softplus()

    def output(self, deconvoluted):
        # Output the parameters of the Beta Distribution
        alpha = self.act(self.alpha(deconvoluted))
        beta = self.act(self.beta(deconvoluted))
        return alpha, beta

    def reconstruct(self, z):
        alpha, beta = self.forward(z)
        variance = (alpha * beta) / ((alpha + beta).pow(2) * (alpha + beta + 1))
        return alpha / (alpha + beta) + variance * torch.randn_like(variance)
