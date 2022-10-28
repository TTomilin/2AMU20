import argparse
import os
from math import sqrt

import torch
from tqdm import tqdm

import datasets
from assignment3.plotting import plot_reconstruction, plot_latent_space, plot_interpolation, plot_loss
from assignment3.utils import EarlyStopping
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
    n_latent = 16
    n_bins = 51
    batch_size = 50
    num_epochs = 100
    learning_rate = 1e-4
    # Use a fixed variance for the decoder for more robust training
    var_x = 0.05

    # Inference parameters
    n_samples = 32
    plot_interval = 5

    # Download and load the data if needed (MNIST or fashion-MNIST)
    train_x, train_labels, test_x, test_labels = datasets.load_mnist() if args.dataset == 'mnist' else datasets.load_fashion_mnist()

    # Normalize the data to be between 0 and 1
    train_x = datasets.normalize_min_max(train_x, 0., 1.)
    test_x = datasets.normalize_min_max(test_x, 0., 1.)

    # Split off the validation set
    val_x = train_x[-10000:, :]
    train_x = train_x[:-10000, :]
    val_labels = train_labels[-10000:]
    train_labels = train_labels[:-10000]

    # Generate torch tensors from the data
    train_x = torch.tensor(train_x).to(device)
    test_x = torch.tensor(test_x).to(device)
    val_x = torch.tensor(val_x).to(device)
    train_N, train_D = train_x.shape

    # Determine the number of pixels on one side of the image
    img_size = int(sqrt(train_D))

    # Determine the number of classes
    n_classes = len(set(train_labels))

    # Create the decoder network
    if dist_type == 'gaussian':
        vae = GaussianVAE(n_latent, train_D)
        stop_threshold = 2
    elif dist_type == 'categorical':
        vae = CategoricalVAE(n_latent, train_D, n_bins)
        stop_threshold = 10
    elif dist_type == 'bernoulli':
        vae = BernoulliVAE(n_latent, train_D)
        stop_threshold = 0.1
    elif dist_type == 'beta':
        vae = BetaVAE(n_latent, train_D)
        stop_threshold = 10
    else:
        raise ValueError(f'Unknown distribution type {dist_type}')

    print(vae.encoder)
    print(vae.decoder)
    print(f'Number of encoder parameters: {sum(p.numel() for p in vae.encoder.parameters())}')
    print(f'Number of decoder parameters: {sum(p.numel() for p in vae.decoder.parameters())}')

    # Use Adam as the optimizer for both the encoder and decoder
    optimizer = torch.optim.Adam(list(vae.decoder.parameters()) + list(vae.encoder.parameters()), lr=learning_rate)

    # Use early stopping to stop training when the validation loss stops improving
    stop_early = EarlyStopping(stop_threshold)

    train_loss_history = []
    val_loss_history = []
    pbar = tqdm(range(num_epochs), position=0, leave=False)
    for epoch in pbar:
        vae.train()
        # make batches of training indices
        shuffled_idx = torch.randperm(train_x.shape[0])
        idx_batches = shuffled_idx.split(batch_size)
        train_loss, val_loss = 0, 0
        for batch_count, idx in enumerate(idx_batches):
            optimizer.zero_grad()
            batch_x = train_x[idx, :]
            input_x = batch_x.reshape(batch_size, img_size, img_size).unsqueeze(1)

            loss = vae.elbo(input_x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss_history.append(train_loss / train_x.shape[0])

        if epoch % plot_interval == 0:
            with torch.no_grad():
                # Store the model weights
                torch.save(vae.encoder.state_dict(), os.path.join(model_path, f'encoder_{epoch}.pt'))
                torch.save(vae.decoder.state_dict(), os.path.join(model_path, f'decoder_{epoch}.pt'))

        with torch.no_grad():
            vae.eval()
            shuffled_idx = torch.randperm(val_x.shape[0])
            idx_batches = shuffled_idx.split(batch_size)
            for batch_count, idx in enumerate(idx_batches):
                batch_x = train_x[idx, :]
                input_x = batch_x.reshape(batch_size, img_size, img_size).unsqueeze(1)

                loss = vae.elbo(input_x)
                val_loss += loss.item()

            val_loss_history.append(val_loss / val_x.shape[0])

        pbar.set_postfix({'Train Loss': round(train_loss, 4), 'Validation Loss': round(val_loss, 4)})
        print()

        if stop_early(train_loss_history[-1]):
            print(f'Training criterion reached at epoch {epoch}. Stopping training...')
            # Store the model weights
            torch.save(vae.encoder.state_dict(), os.path.join(model_path, 'encoder.pt'))
            torch.save(vae.decoder.state_dict(), os.path.join(model_path, 'decoder.pt'))
            break

        # Plot and save the ELBO curve and data reconstruction
        # plot_loss(train_loss_history, val_loss_history, img_path)
        # test_input = test_x.reshape(test_x.shape[0], img_size, img_size).unsqueeze(1)
        test_samples = test_x[0:n_samples, :]
        # plot_latent_space(vae.encoder, test_input[:1000], test_labels[:1000])
        # plot_latent_space(vae.encoder, test_input[:1000], test_labels[:1000], use_pca=True)
        # plot_interpolation(vae, test_input, test_labels, img_size)
        plot_reconstruction(vae, n_samples, img_size, test_samples, reconstruct=True)
        plot_reconstruction(vae, n_samples, img_size, test_samples, reconstruct=False)
