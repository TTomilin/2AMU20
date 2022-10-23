import argparse
import os
from math import sqrt

import numpy as np
import torch
from torch.nn.functional import gumbel_softmax, binary_cross_entropy
from tqdm import tqdm

import datasets
from assignment3.model import ConvEncoder, ConvDecoder, CategoricalEncoder, CategoricalDecoder, BernoulliEncoder, \
    BernoulliDecoder
from assignment3.plotting import plot_interpolation, plot_latent_space, plot_loss, plot_reconstruction
from assignment3.utils import categorical_kl, binomial_kl

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, default='mnist', help='Dataset to use', choices=['mnist', 'mnist-fashion']
    )
    parser.add_argument(
        '--distribution', type=str, default='gaussian', help='Dataset to use',
        choices=['gaussian', 'beta', 'categorical', 'bernoulli']
    )

    args = parser.parse_args()
    dist_type = args.distribution

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_path = f'img_vae/{dist_type}'
    model_path = f'models/{dist_type}'
    datasets.mkdir_p(img_path)
    datasets.mkdir_p(model_path)

    # Training parameters
    n_latent = 50
    n_distributions = 50
    batch_size = 50
    num_epochs = 100
    learning_rate = 1e-4
    stop_criterion = 50
    # Use a fixed variance for the decoder for more robust training
    var_x = 0.05

    # Inference parameters
    n_samples = 10
    plot_interval = 5

    # Download and load the data if needed (MNIST or fashion-MNIST)
    train_x, train_labels, test_x, test_labels = datasets.load_mnist() if args.dataset == 'mnist' else datasets.load_fashion_mnist()

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

    # Determine the number of classes
    n_classes = len(set(train_labels))

    # Create and print the encoder and decoder networks
    if dist_type == 'categorical':
        encoder = CategoricalEncoder(n_classes, n_distributions).to(device)
        decoder = CategoricalDecoder(train_D, n_classes, n_distributions).to(device)
    elif dist_type == 'bernoulli':
        encoder = BernoulliEncoder(n_latent).to(device)
        decoder = BernoulliDecoder(n_latent).to(device)
    else:
        encoder = ConvEncoder(n_latent).to(device)
        decoder = ConvDecoder(train_D, n_latent, var=var_x).to(device)
    print(encoder)
    print(decoder)

    print(f'Number of encoder parameters: {sum(p.numel() for p in encoder.parameters())}')
    print(f'Number of decoder parameters: {sum(p.numel() for p in decoder.parameters())}')

    # Use Adam as the optimizer for both the encoder and decoder
    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(encoder.parameters()), lr=learning_rate)

    loss_history = []
    for epoch in tqdm(range(num_epochs)):
        # make batches of training indices
        shuffled_idx = torch.randperm(train_x.shape[0])
        idx_batches = shuffled_idx.split(batch_size)
        sum_loss = 0.0
        for batch_count, idx in enumerate(idx_batches):
            optimizer.zero_grad()
            batch_x = train_x[idx, :]
            input_x = batch_x.reshape(batch_size, img_size, img_size).unsqueeze(1)

            if dist_type == 'categorical':
                z = encoder(input_x)
                z_given_x = gumbel_softmax(z)
                x_decoded = decoder(z_given_x)
                KL = torch.mean(torch.sum(categorical_kl(z, device), dim=1))
                reconstruction = x_decoded + sqrt(decoder.var) * torch.randn(batch_x.shape[1], device=device)
            elif dist_type == 'bernoulli':
                input_x_binary = torch.bernoulli(input_x)
                z = encoder(input_x_binary)
                z_given_x = gumbel_softmax(z)
                x_decoded = decoder(z_given_x)
            else:
                batch_mu_z, batch_var_z = encoder(input_x)
                # Sample z using the reparameterization trick
                z = batch_mu_z + torch.sqrt(batch_var_z) * torch.randn(batch_var_z.shape, device=device)
                x_decoded = decoder(z)
                KL = -0.5 * torch.sum(1 + torch.log(batch_var_z) - batch_mu_z**2 - batch_var_z)
                reconstruction = x_decoded + sqrt(decoder.var) * torch.randn(batch_x.shape[1], device=device)

            if dist_type == 'bernoulli':
                entropy = binary_cross_entropy(x_decoded, batch_x)
                KL = torch.mean(torch.sum(binomial_kl(z, device), dim=1))

                # loss = entropy + KL
                loss = entropy

            else:
                # Squared distances between the original input and the reconstruction
                d2 = (reconstruction - batch_x)**2

                # Gaussian likelihood: 1/sqrt(2*pi*var) exp(-0.5 * (mu-x)**2 / var)
                # Thus, log-likelihood = -0.5 * ( log(2*pi*var) + (mu-x)**2 / var )
                log_p = -0.5 * torch.sum(np.log(decoder.var * 2 * np.pi) + d2 / decoder.var)
                # We want to maximize the ELBO, hence minimize the negative ELBO
                loss = KL - log_p

            loss.backward()
            optimizer.step()
            sum_loss += loss
        mean_loss = sum_loss / train_x.shape[0]
        print(f'Epoch {epoch}. Loss = {mean_loss}')
        loss_history.append(mean_loss.cpu().detach().numpy())

        if epoch % plot_interval == 0:
            with torch.no_grad():
                # Store the model weights
                torch.save(encoder.state_dict(), os.path.join(model_path, f'encoder_{epoch}.pt'))
                torch.save(decoder.state_dict(), os.path.join(model_path, f'decoder_{epoch}.pt'))

                # Plot and save the ELBO curve and data reconstruction
                samples = train_x[0:n_samples, :]
                file_name = os.path.join(img_path, f'samples_{epoch}.png')
                plot_reconstruction(dist_type, encoder, decoder, device, samples, n_samples, file_name, img_size)
                plot_loss(loss_history, img_path)

        if mean_loss < stop_criterion:
            print('Training criterion reached. Stopping training.')
            # Store the model weights
            torch.save(encoder.state_dict(), os.path.join(model_path, f'encoder.pt'))
            torch.save(decoder.state_dict(), os.path.join(model_path, f'decoder.pt'))
            break

    test_input = test_x.reshape(test_x.shape[0], img_size, img_size).unsqueeze(1)
    plot_latent_space(dist_type, encoder, test_input[:1000], test_labels[:1000])
    plot_latent_space(dist_type, encoder, test_input[:1000], test_labels[:1000], use_pca=True)
    plot_interpolation(dist_type, encoder, decoder, test_input, test_labels)

    samples = test_x[0:n_samples, :]
    file_name = os.path.join(img_path, 'samples_test.png')
    plot_reconstruction(dist_type, encoder, decoder, device, samples, n_samples, file_name, img_size)
