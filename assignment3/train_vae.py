import argparse
import os
from math import sqrt

import torch
from tqdm import tqdm

import datasets
from assignment3.model import GaussianEncoder, BernoulliEncoder
from assignment3.plotting import plot_interpolation, plot_loss, plot_reconstruction, plot_latent_space
from assignment3.vae import GaussianVAE, BernoulliVAE, BetaVAE, CategoricalVAE

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
    n_bins = 50
    batch_size = 50
    num_epochs = 100
    learning_rate = 1e-4
    stop_criterion = -500
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

    vae = None

    # Create the encoder network
    if dist_type == 'bernoulli':
        encoder = BernoulliEncoder(n_latent)
    else:
        encoder = GaussianEncoder(n_latent)

    # Create the decoder network
    if dist_type == 'gaussian':
        vae = GaussianVAE(n_latent, train_D, device)
    elif dist_type == 'categorical':
        vae = CategoricalVAE(n_latent, train_D, device, n_bins)
    elif dist_type == 'bernoulli':
        vae = BernoulliVAE(n_latent, train_D, device)
    elif dist_type == 'beta':
        vae = BetaVAE(n_latent, train_D, device)
    else:
        raise ValueError(f'Unknown distribution type {dist_type}')

    print(vae.encoder)
    print(vae.decoder)
    print(f'Number of encoder parameters: {sum(p.numel() for p in vae.encoder.parameters())}')
    print(f'Number of decoder parameters: {sum(p.numel() for p in vae.decoder.parameters())}')

    # Use Adam as the optimizer for both the encoder and decoder
    optimizer = torch.optim.Adam(list(vae.decoder.parameters()) + list(vae.encoder.parameters()), lr=learning_rate)

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

            loss = vae.loss(input_x)
            loss.backward()
            optimizer.step()
            sum_loss += loss
        mean_loss = sum_loss / train_x.shape[0]
        print(f'Epoch {epoch}. Loss = {mean_loss}')
        loss_history.append(mean_loss.cpu().detach().numpy())

        if epoch % plot_interval == 0:
            with torch.no_grad():
                # Store the model weights
                torch.save(vae.encoder.state_dict(), os.path.join(model_path, f'encoder_{epoch}.pt'))
                torch.save(vae.decoder.state_dict(), os.path.join(model_path, f'decoder_{epoch}.pt'))

                # Plot and save the ELBO curve and data reconstruction
                samples = test_x[0:n_samples, :]
                file_name = os.path.join(img_path, f'samples_{epoch}.png')
                plot_reconstruction(dist_type, vae, samples, n_samples, file_name, img_size)
                plot_loss(loss_history, img_path)

        if mean_loss < stop_criterion:
            print('Training criterion reached. Stopping training.')
            # Store the model weights
            torch.save(encoder.state_dict(), os.path.join(model_path, f'encoder.pt'))
            torch.save(decoder.state_dict(), os.path.join(model_path, f'decoder.pt'))
            break

        test_input = test_x.reshape(test_x.shape[0], img_size, img_size).unsqueeze(1)
        plot_latent_space(dist_type, vae.encoder, test_input[:1000], test_labels[:1000])
        plot_latent_space(dist_type, vae.encoder, test_input[:1000], test_labels[:1000], use_pca=True)
        plot_interpolation(dist_type, vae, test_input, test_labels)

    samples = test_x[0:n_samples, :]
    file_name = os.path.join(img_path, 'samples_test.png')
    plot_reconstruction(dist_type, encoder, decoder, device, samples, n_samples, file_name, img_size)
