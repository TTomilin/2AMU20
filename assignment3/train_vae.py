import os
from math import sqrt

import numpy as np
import torch
from tqdm import tqdm

import datasets
from assignment3.model import ConvEncoder, ConvDecoder
from assignment3.plotting import plot_interpolation, plot_latent_space, plot_ELBO
from assignment3.util import visualize_reconstruction

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_path = 'img_vae'
    model_path = 'models'
    datasets.mkdir_p(img_path)
    datasets.mkdir_p(model_path)

    # Training parameters
    n_latent = 50
    batch_size = 50
    num_epochs = 100
    learning_rate = 1e-4
    # Use a fixed variance for the decoder for more robust training
    var_x = 0.05

    # Inference parameters
    n_samples = 10
    plot_interval = 5

    # Download and load the data if needed (MNIST or fashion-MNIST)
    train_x, train_labels, test_x, test_labels = datasets.load_mnist()
    # train_x, train_labels, test_x, test_labels = datasets.load_fashion_mnist()

    # Normalize the data to be between 0 and 1
    train_x = datasets.normalize_min_max(train_x, 0., 1.)
    test_x = datasets.normalize_min_max(test_x, 0., 1.)

    # Split off the validation set
    valid_x = train_x[-10000:, :]
    train_x = train_x[:-10000, :]
    valid_labels = train_labels[-10000:]
    train_labels = train_labels[:-10000]

    # Generate torch tensors from the data
    train_x = torch.tensor(train_x).to(device)
    test_x = torch.tensor(test_x).to(device)
    train_N, train_D = train_x.shape

    # Determine the number of pixels on one side of the image
    img_size = int(sqrt(train_D))

    # Create and print the encoder and decoder networks
    encoder = ConvEncoder(n_latent).to(device)
    decoder = ConvDecoder(train_D, n_latent, var=var_x).to(device)
    print(encoder)
    print(decoder)

    # Use Adam as the optimizer for both the encoder and decoder
    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(encoder.parameters()), lr=learning_rate)

    ELBO_history = []
    for epoch in tqdm(range(num_epochs)):
        # make batches of training indices
        shuffled_idx = torch.randperm(train_x.shape[0])
        idx_batches = shuffled_idx.split(batch_size)
        sum_neg_ELBO = 0.0
        for batch_count, idx in enumerate(idx_batches):
            optimizer.zero_grad()
            batch_x = train_x[idx, :]
            input_x = batch_x.reshape(batch_size, img_size, img_size).unsqueeze(1)
            # batch_mu_z: batch_size, num_latent
            # batch_var_z: batch_size, num_latent
            batch_mu_z, batch_var_z = encoder(input_x)
            # sample z, using the "reparametrization trick"
            batch_z = batch_mu_z + torch.sqrt(batch_var_z) * torch.randn(batch_var_z.shape, device=device)
            # mu_x: batch_size, D
            mu_x = decoder(batch_z)
            # squared distances between mu_x and batch_x
            d2 = (mu_x - batch_x)**2
            # Gaussian likelihood: 1/sqrt(2*pi*var) exp(-0.5 * (mu-x)**2 / var)
            # Thus, log-likelihood = -0.5 * ( log(2*pi*var) + (mu-x)**2 / var )
            log_p = -0.5 * torch.sum(np.log(decoder.var * 2 * np.pi) + d2 / decoder.var)
            KL = -0.5 * torch.sum(1 + torch.log(batch_var_z) - batch_mu_z**2 - batch_var_z)
            # we want to maximize the ELBO, hence minimize the negative ELBO
            negative_ELBO = -log_p + KL
            negative_ELBO.backward()
            optimizer.step()
            sum_neg_ELBO += negative_ELBO
        mean_neg_ELBO = sum_neg_ELBO / train_x.shape[0]
        print(f'Epoch {epoch}. Mean negative ELBO = {mean_neg_ELBO}')
        ELBO_history.append(mean_neg_ELBO.cpu().detach().numpy())

        if epoch % plot_interval == 0:
            with torch.no_grad():
                samples = train_x[0:n_samples, :]
                file_name = os.path.join(img_path, 'samples_{}.png'.format(epoch))
                visualize_reconstruction(encoder, decoder, device, samples, n_samples, file_name, img_size)

                # Store the model weights
                torch.save(encoder.state_dict(), os.path.join(model_path, f'encoder_{epoch}.pt'))
                torch.save(decoder.state_dict(), os.path.join(model_path, f'decoder_{epoch}.pt'))

                test_input = test_x.reshape(test_x.shape[0], img_size, img_size).unsqueeze(1)
                plot_ELBO(ELBO_history, img_path)
                plot_latent_space(encoder, test_input, test_labels)
                plot_interpolation(encoder, decoder, test_input, test_labels)

        if mean_neg_ELBO < -300:
            print('Training criterion reached. Stopping training.')
            break