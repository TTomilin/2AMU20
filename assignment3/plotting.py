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


def plot_latent_space(encoder: Module, test_x: Tensor, test_labels: Tensor, use_pca=False):
    """
    Display a 2D plot of the digit classes in the latent space
    :param encoder: encoder network extending torch.nn.Module
    :param test_x: test image data
    :param test_labels: test image labels [0-9]
    :param use_pca: whether to use PCA to reduce the dimensionality of the latent space
    :return:
    """
    z, _, _ = encoder(test_x)
    values = z.detach().cpu().numpy()
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


def plot_interpolation(vae: VAE, test_x: Tensor, test_labels: Tensor, img_size: int, k=15):
    """
    Plot interpolation between two pair-wise sampled digits in the latent space.
    :param vae: VAE model
    :param test_x: test image data
    :param test_labels: test image labels [0-9]
    :param img_size: number of pixels of the width/height of the square MNIST digit image
    :param k: number of digit pairs to interpolate between
    """
    # 1D linearly spaced partition
    lin_space = np.linspace(0, 1, k)
    figure = np.zeros((img_size * k, img_size * k))
    row = 0
    while row < k:
        # Sample two digits
        indices = np.random.choice(len(test_x), 2)
        # Re-sample the indices if the labels are the same
        if test_labels[indices[0]] == test_labels[indices[1]]:
            continue

        x1 = test_x[indices[0]].reshape(1, 1, img_size, img_size)
        x2 = test_x[indices[1]].reshape(1, 1, img_size, img_size)

        for j, lambda_ in enumerate(lin_space):

            z1, _, _ = vae.encode(x1)
            z2, _, _ = vae.encode(x2)

            # Interpolate between the two latent vectors
            z = lambda_ * z1 + (1 - lambda_) * z2

            x_hat = vae.decoder.reconstruct(z)
            x_hat = x_hat.reshape(img_size, img_size).detach().cpu().numpy()

            # Plot the interpolated digit
            figure[row * img_size: (row + 1) * img_size, j * img_size: (j + 1) * img_size] = x_hat
        row += 1

    plt.figure(figsize=(10, 10))
    start_range = img_size // 2
    end_range = k * img_size - start_range + 1
    pixel_range = np.arange(start_range, end_range, img_size)
    sample_range_x = np.round(lin_space, 2)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks([])
    plt.xlabel("lambda")
    plt.imshow(figure, cmap='Greys_r')
    plt.show()


def save_reconstruction(vae: VAE, x_input: Tensor, n_samples: int, file_path: str, img_size: int):
    """
    Plot the reconstruction of the input samples
    :param vae: VAE model
    :param x_input: input images
    :param n_samples: number of new images to generate
    :param file_path: path to save the plot
    :param img_size: number of pixels of the width/height of the square MNIST digit image
    """
    # Reshape the input
    x_reshaped = x_input.reshape(n_samples, img_size, img_size).unsqueeze(1)

    x_hat = vae.reconstruct(x_reshaped)
    x_hat = x_hat.reshape(n_samples, img_size, img_size).detach().cpu().numpy()

    # Generate new images by sampling from the latent space
    samples, _ = vae.sample(n_samples)

    # Save images
    n_rows = 3
    plot_img = np.stack((x_input.detach().cpu().numpy(),
                         x_hat.detach().cpu().numpy(),
                         samples.detach().cpu().numpy()))
    plot_img = np.reshape(plot_img, (n_samples * n_rows, img_size, img_size))
    datasets.save_image_stack(plot_img, n_rows, n_samples, file_path, margin=3)


def plot_reconstruction(vae: VAE, n_samples: int, img_size: int, samples=None, reconstruct=False):
    if reconstruct:
        samples = samples.reshape(n_samples, img_size, img_size)
        x = samples.unsqueeze(1)

        x_hat = vae.reconstruct(x)
        x_hat = x_hat.reshape(n_samples, img_size, img_size).detach().cpu().numpy()

        x = x.squeeze().detach().cpu().numpy()
        plot_images = list(chain.from_iterable(zip(x, x_hat)))
    else:
        samples = vae.sample(n_samples)
        samples = samples.reshape(n_samples, img_size, img_size)
        plot_images = samples.detach().cpu().numpy()

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
