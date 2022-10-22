from math import sqrt

import numpy as np
import torch
from torch.nn import Conv2d, Linear, BatchNorm2d, ConvTranspose2d, ELU, Softplus, Module


class ConvEncoder(Module):
    def __init__(self, n_latent: int, in_channels=1, n_conv_blocks=3, n_filters=32, n_fc=2048):
        super(ConvEncoder, self).__init__()
        self.n_latent = n_latent
        self.n_fc = n_fc
        conv_head = []
        for i in range(n_conv_blocks):
            conv_head.append(Conv2d(in_channels, n_filters, kernel_size=3, stride=2, padding=1))
            conv_head.append(BatchNorm2d(n_filters))
            conv_head.append(ELU(inplace=True))
            in_channels = n_filters
            n_filters *= 2
        # TODO Do we need tanh?
        # conv_head.append(Tanh())
        self.conv_head = torch.nn.ModuleList(conv_head)

        # Mean and variance layers
        self.mu = Linear(n_fc, n_latent)
        self.var = Linear(n_fc, n_latent)
        self.var_act = Softplus()

    def forward(self, x):
        res = x
        for layer in self.conv_head:
            res = layer(res)
        res = res.view(-1, self.n_fc)
        mu = self.mu(res)
        var = self.var_act(self.var(res))
        return mu, var


class ConvDecoder(Module):
    def __init__(self, n_pixels: int, n_latent: int, n_deconv_blocks=3, in_channels=128, n_fc=2048, var=0.05):
        super(ConvDecoder, self).__init__()
        self.n_pixels = n_pixels
        self.n_latent = n_latent
        self.in_channels = in_channels
        self.n_fc = n_fc
        self.filter_size = int(sqrt(n_fc / in_channels))
        self.var = var
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
        self.de_conv_head = torch.nn.ModuleList(de_conv_head)

        # Generate the output layer
        self.mu = torch.nn.Linear(n_pixels, n_pixels)

    def forward(self, z):
        # First run the latent vector through a dense layer
        res = torch.nn.functional.relu(self.dense(z))
        # Create a suitable shape for the de-convolution
        res = res.reshape(-1, self.in_channels, self.filter_size, self.filter_size)
        for layer in self.de_conv_head:
            res = layer(res)
        res = res.flatten(1)
        mu = self.mu(res)
        return mu

    def sample(self, n_samples, device, convert_to_numpy=False, suppress_noise=True):
        with torch.no_grad():
            z = torch.randn(n_samples, self.n_latent, device=device)
            mu = self.forward(z)
            x = mu
            # The conditional VAE distribution is an isotropic Gaussian, hence we just add noise when sampling it
            # Suppress this for images
            if not suppress_noise:
                x += np.sqrt(self.var) * torch.randn(n_samples, self.n_pixels, device=device)
            if convert_to_numpy:
                z = z.cpu().numpy()
                x = x.cpu().numpy()
            return x, z


class CategoricalEncoder(Module):
    def __init__(self, n_classes: int, n_distributions=50, in_channels=1, n_conv_blocks=3, n_filters=32, n_fc=2048):
        super(CategoricalEncoder, self).__init__()
        self.n_distributions = n_distributions
        self.n_classes = n_classes
        self.n_fc = n_fc
        conv_head = []
        for i in range(n_conv_blocks):
            conv_head.append(Conv2d(in_channels, n_filters, kernel_size=3, stride=2, padding=1))
            conv_head.append(BatchNorm2d(n_filters))
            conv_head.append(ELU(inplace=True))
            in_channels = n_filters
            n_filters *= 2
        self.conv_head = torch.nn.ModuleList(conv_head)

        # Categorical distribution
        self.dist = Linear(n_fc, n_distributions * n_classes)

    def forward(self, x):
        res = x
        for layer in self.conv_head:
            res = layer(res)
        res = res.flatten(1)
        dist = self.dist(res).view(-1, self.n_distributions, self.n_classes)
        return dist


class CategoricalDecoder(Module):
    def __init__(self, n_pixels: int, n_classes: int, n_distributions=50, n_deconv_blocks=3, in_channels=128, n_fc=2048, var=0.05):
        super(CategoricalDecoder, self).__init__()
        self.n_distributions = n_distributions
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.n_fc = n_fc
        self.var = var
        self.filter_size = int(sqrt(n_fc / in_channels))
        self.dense = Linear(n_classes * n_distributions, n_fc)

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
        self.de_conv_head = torch.nn.ModuleList(de_conv_head)

        # Generate the output layer
        self.mu = torch.nn.Linear(n_pixels, n_pixels)

    def forward(self, z):
        assert len(z.shape) == 3  # [batch_size, n_distributions, n_classes]
        assert z.shape[1:] == (self.n_distributions, self.n_classes)

        # First run the latent vector through a dense layer
        res = z.flatten(1)
        res = torch.nn.functional.relu(self.dense(res))
        # Create a suitable shape for the de-convolution
        res = res.reshape(-1, self.in_channels, self.filter_size, self.filter_size)
        for layer in self.de_conv_head:
            res = layer(res)
        res = res.flatten(1)
        mu = self.mu(res)
        return mu

    def sample(self, n_samples, device, convert_to_numpy=False, suppress_noise=True):
        with torch.no_grad():
            # z = torch.randn(n_samples, self.n_latent, device=device)
            z = torch.randn(n_samples, self.n_distributions, self.n_classes, device=device)
            mu = self.forward(z)
            x = mu
            # The conditional VAE distribution is an isotropic Gaussian, hence we just add noise when sampling it
            # Suppress this for images
            if not suppress_noise:
                x += np.sqrt(self.var) * torch.randn(n_samples, self.num_var, device=device)
            if convert_to_numpy:
                z = z.cpu().numpy()
                x = x.cpu().numpy()
            return x, z
