import numpy as np
import torch

from assignment3 import datasets


def visualize_reconstruction(encoder, decoder, device, samples, n_samples: int, file_path: str, img_size: int):
    plot_rows = 3
    # Sample from the VAE
    x, z = decoder.sample(n_samples, device)
    # Encode some samples
    input_x = samples.reshape(n_samples, img_size, img_size).unsqueeze(1)
    mu_z, var_z = encoder(input_x)
    z_encoded = mu_z + torch.sqrt(var_z) * torch.randn(n_samples, encoder.n_latent, device=device)
    # Decode the samples
    x_decoded = decoder(z_encoded)
    # Save images
    plot_img = np.stack((samples.detach().cpu().numpy(),
                         x_decoded.detach().cpu().numpy(),
                         x.detach().cpu().numpy()))
    plot_img = np.reshape(plot_img, (n_samples * plot_rows, img_size, img_size))
    datasets.save_image_stack(plot_img, plot_rows, n_samples, file_path, margin=3)
