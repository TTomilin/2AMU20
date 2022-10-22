from os import path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import gumbel_softmax

from assignment3 import datasets


def plot_ELBO(ELBO_history: list, save_path: str = None):
    plt.figure(figsize=(12, 10))
    plt.clf()
    plt.plot(ELBO_history)
    plt.show()
    plt.savefig(path.join(save_path, 'elbo.png'))


def plot_latent_space(distribution: str, encoder: Module, test_x: Tensor, test_labels: Tensor):
    """
    Display a 2D plot of the digit classes in the latent space
    :param distribution: type of the latent distribution
    :param encoder: encoder network extending torch.nn.Module
    :param test_x: test image data
    :param test_labels: test image labels [0-9]
    :return:
    """
    if distribution == 'categorical':
        z_mean = encoder(test_x)
    else:
        z_mean, _ = encoder(test_x)
    z_mean = z_mean.cpu().detach().numpy()
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=test_labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


def plot_latent_space_PCA(encoder: Module, test_x: Tensor, test_labels: Tensor):
    """
    Display a 2D plot of the digit classes in the latent space using Principal Component Analysis (PCA)
    :param encoder: encoder network extending torch.nn.Module
    :param test_x: test image data
    :param test_labels: test image labels [0-9]
    :return:
    """
    z_mean, _ = encoder(test_x)
    z_mean = z_mean.cpu().detach().numpy()
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=test_labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


def plot_interpolation(distribution: str, encoder: Module, decoder: Module, test_x: Tensor, test_labels: Tensor,
                       digit_size=28, k=15):
    """
    Plot interpolation between two pair-wise sampled digits in the latent space.
    :param distribution: type of the latent distribution
    :param encoder: encoder network extending torch.nn.Module
    :param decoder: decoder network extending torch.nn.Module
    :param test_x: test image data
    :param test_labels: test image labels [0-9]
    :param digit_size: number of pixels of the width/height of the square MNIST digit image
    :param k: number of digit pairs to interpolate between
    """
    # 1D linearly spaced partition
    lin_space = np.linspace(0, 1, k)
    figure = np.zeros((digit_size * k, digit_size * k))
    row = 0
    while row < k:
        # Sample two digits
        indices = np.random.choice(len(test_x), 2)
        # Re-sample the indices if the labels are the same
        if test_labels[indices[0]] == test_labels[indices[1]]:
            continue

        x1 = test_x[indices[0]].reshape(1, 1, digit_size, digit_size)
        x2 = test_x[indices[1]].reshape(1, 1, digit_size, digit_size)

        for j, lambda_ in enumerate(lin_space):

            if distribution == 'categorical':
                z1 = encoder(x1)
                z2 = encoder(x2)
            else:
                z1, _ = encoder(x1)
                z2, _ = encoder(x2)

            # Interpolate between the two latent vectors
            z = lambda_ * z1 + (1 - lambda_) * z2

            if distribution == 'categorical':
                z = gumbel_softmax(z)

            # Decode the interpolated latent vector
            x = decoder(z).reshape(digit_size, digit_size).cpu().detach().numpy()

            # Plot the interpolated digit
            figure[row * digit_size: (row + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = x
        row += 1

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = k * digit_size - start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(lin_space, 2)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks([])
    plt.xlabel("lambda")
    plt.imshow(figure, cmap='Greys_r')
    plt.show()


def plot_reconstruction(distribution: str, encoder: Module, decoder: Module, device: str, input_x: Tensor,
                        n_samples: int, file_path: str, img_size: int):
    """
    Plot the reconstruction of the input samples
    :param distribution: type of the latent distribution
    :param encoder: encoder network extending torch.nn.Module
    :param decoder: decoder network extending torch.nn.Module
    :param device: cpu/cuda
    :param input_x: input images
    :param n_samples: number of new images to generate
    :param file_path: path to save the plot
    :param img_size: number of pixels of the width/height of the square MNIST digit image
    """
    # Reshape the input
    x_reshaped = input_x.reshape(n_samples, img_size, img_size).unsqueeze(1)

    # Encode the input
    if distribution == 'categorical':
        z = encoder(x_reshaped)
        z = gumbel_softmax(z)
    else:
        mu_z, var_z = encoder(x_reshaped)
        z = mu_z + torch.sqrt(var_z) * torch.randn(n_samples, encoder.n_latent, device=device)

    # Decode the latent vector
    x_decoded = decoder(z)

    # Generate new images by sampling from the latent space
    samples, _ = decoder.sample(n_samples, device)

    # Save images
    n_rows = 3
    plot_img = np.stack((input_x.detach().cpu().numpy(),
                         x_decoded.detach().cpu().numpy(),
                         samples.detach().cpu().numpy()))
    plot_img = np.reshape(plot_img, (n_samples * n_rows, img_size, img_size))
    datasets.save_image_stack(plot_img, n_rows, n_samples, file_path, margin=3)
