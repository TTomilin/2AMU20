from itertools import chain
from os import path

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import Module
from torch.nn.functional import gumbel_softmax

from assignment3 import datasets
from assignment3.utils import log_beta_pdf
from assignment3.vae import VAE


def plot_loss(loss_history: list, save_path: str = None):
    """
    Plot the loss history
    :param loss_history: list of loss values
    :param save_path: path to save the plot to
    """
    plt.figure(figsize=(12, 10))
    plt.clf()
    plt.plot(loss_history)
    plt.show()
    plt.savefig(path.join(save_path, 'elbo.png'))


def plot_latent_space(distribution: str, encoder: Module, test_x: Tensor, test_labels: Tensor, use_pca=False):
    """
    Display a 2D plot of the digit classes in the latent space
    :param distribution: type of the latent distribution
    :param encoder: encoder network extending torch.nn.Module
    :param test_x: test image data
    :param test_labels: test image labels [0-9]
    :param use_pca: whether to use PCA to reduce the dimensionality of the latent space
    :return:
    """
    if distribution == 'bernoulli':
        z_mean = encoder(test_x)
    else:
        z_mean, _ = encoder(test_x)
    values = z_mean.cpu().detach().numpy()
    plt.figure(figsize=(12, 10))

    if use_pca:
        pca = PCA(n_components=2)
        values = pca.fit_transform(values)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
    else:
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")

    c_map = mpl.cm.viridis
    bounds = np.linspace(0, 9, 10)
    norm = mpl.colors.BoundaryNorm(bounds, c_map.N, extend='both')
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=c_map))
    plt.scatter(values[:, 0], values[:, 1], c=test_labels)
    plt.show()


def plot_interpolation(distribution: str, vae: VAE, test_x: Tensor, test_labels: Tensor,
                       digit_size=28, k=15):
    """
    Plot interpolation between two pair-wise sampled digits in the latent space.
    :param distribution: type of the latent distribution
    :param vae: VAE model
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

            if distribution == 'bernoulli':
                z1 = vae.encode(x1)
                z2 = vae.encode(x2)
            else:
                z1, _ = vae.encode(x1)
                z2, _ = vae.encode(x2)

            # Interpolate between the two latent vectors
            z = lambda_ * z1 + (1 - lambda_) * z2

            if distribution == 'bernoulli':
                z = gumbel_softmax(z)

            # Decode the interpolated latent vector
            if distribution == 'beta':
                alpha, beta = vae.decode(z)
                x = log_beta_pdf(x1.reshape(1, digit_size * digit_size), alpha, beta)
                x = x.reshape(1, 1, digit_size, digit_size)
            else:
                x = vae.decode(z).reshape(digit_size, digit_size)

            if distribution == 'bernoulli':
                x = torch.bernoulli(x)

            # Plot the interpolated digit
            figure[row * digit_size: (row + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = x.detach().cpu().numpy()
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


def save_reconstruction(dist_type: str, vae: VAE, x_input: Tensor, n_samples: int, file_path: str, img_size: int):
    """
    Plot the reconstruction of the input samples
    :param dist_type: type of the latent distribution
    :param vae: VAE model
    :param x_input: input images
    :param n_samples: number of new images to generate
    :param file_path: path to save the plot
    :param img_size: number of pixels of the width/height of the square MNIST digit image
    """
    # Reshape the input
    x_reshaped = x_input.reshape(n_samples, img_size, img_size).unsqueeze(1)

    if dist_type == 'gaussian':
        x_decoded = vae.forward(x_reshaped)
    elif dist_type == 'bernoulli':
        x_decoded = vae.forward(x_reshaped)
        x_decoded = torch.bernoulli(x_decoded)
        x_input = torch.bernoulli(x_input)
    elif dist_type == 'beta':
        alpha, beta = vae.forward(x_reshaped)
        x_decoded = log_beta_pdf(x_input, alpha, beta)
        x_decoded = torch.sigmoid(x_decoded)
    elif dist_type == 'categorical':
        pixels_per_bin = 5
        # x_reshaped = torch.floor(x_reshaped / pixels_per_bin)
        x_decoded = vae.forward(x_reshaped)
        x_one_hot = torch.nn.functional.one_hot(x_reshaped.flatten(1).to(torch.int64), num_classes=vae.decoder.n_bins)
        x_decoded = torch.sum(x_one_hot * torch.log(x_decoded), dim=1)
        # x_decoded = gumbel_softmax(x_decoded)
        # x_input = gumbel_softmax(x_input)
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")

    # Generate new images by sampling from the latent space
    samples, _ = vae.sample(n_samples)

    # Save images
    n_rows = 3
    plot_img = np.stack((x_input.detach().cpu().numpy(),
                         x_decoded.detach().cpu().numpy(),
                         samples.detach().cpu().numpy()))
    plot_img = np.reshape(plot_img, (n_samples * n_rows, img_size, img_size))
    datasets.save_image_stack(plot_img, n_rows, n_samples, file_path, margin=3)


def plot_reconstruction(vae: VAE, dist_type: str, n_samples: int, img_size: int, samples=None, reconstruct=False):
    # Reconstruct the input images
    if reconstruct:
        samples = samples.reshape(n_samples, img_size, img_size)
        images = samples.unsqueeze(1)

        if dist_type == 'gaussian':
            x_hat = vae.forward(images)
        elif dist_type == 'beta':
            alpha, beta = vae.forward(images)
            variance = (alpha * beta) / ((alpha + beta).pow(2) * (alpha + beta + 1))
            x_hat = alpha / (alpha + beta) + variance * torch.randn_like(variance)
        elif dist_type == 'categorical':
            pmf = vae.forward(images)
            dist = Categorical(pmf)
            x_hat = dist.sample() / vae.decoder.n_bins
        elif dist_type == 'bernoulli':
            pmf = vae.forward(images)
            x_hat = torch.bernoulli(pmf)
            samples = torch.bernoulli(samples)
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

        x_hat = x_hat.reshape(n_samples, img_size, img_size).detach().cpu().numpy()

        images = images.squeeze().detach().cpu().numpy()
        plot_images = list(chain.from_iterable(zip(images, x_hat)))
    else:
        samples = vae.sample(n_samples)
        samples = samples.reshape(n_samples, img_size, img_size)
        plot_images = samples.detach().cpu().numpy()

    # Plot images
    height = samples.shape[1]
    width = samples.shape[2]

    num_columns = 8
    num_rows = len(plot_images) // num_columns
    frame = 2
    frame_gray_val = 1.0
    margin = 5
    margin_gray_val = 1.0

    img = margin_gray_val * np.ones((height * num_rows + (num_rows - 1) * margin,
                                     width * num_columns + (num_columns - 1) * margin))
    counter = 0
    for h in range(num_rows):
        for w in range(num_columns):
            img[h * (height + margin): h * (height + margin) + height,
            w * (width + margin): w * (width + margin) + width] = plot_images[counter]
            counter += 1

    framed_img = frame_gray_val * np.ones((img.shape[0] + 2 * frame, img.shape[1] + 2 * frame))
    framed_img[frame:(frame + img.shape[0]), frame:(frame + img.shape[1])] = img

    plt.imshow(framed_img)
    plt.show()
