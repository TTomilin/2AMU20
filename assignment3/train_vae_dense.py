import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import datasets
from assignment3.plotting import plot_latent_space, plot_interpolation

device = 'cuda' if torch.cuda.is_available() else 'cpu'
datasets.mkdir_p('img_vae_mlp')

# training params
# MNIST has 784 pixels
num_latent = 50
num_neurons = [1000, 1000]
batch_size = 50
num_epochs = 100
# we are using a fixed variance for the decoder
# the variance can be also made an output of the decoder, see lecture slides,
# but training becomes trickier and somewhat "brittle"
var_x = 0.05

# these function download (if needed) and load MNIST or fashion-MNIST
# Note: sometimes the websites providing the data are down...
#
train_x, train_labels, test_x, test_labels = datasets.load_mnist()
# train_x, train_labels, test_x, test_labels = datasets.load_fashion_mnist()

# normalize data
train_x = datasets.normalize_min_max(train_x, 0., 1.)
test_x = datasets.normalize_min_max(test_x, 0., 1.)

# split off a validation set (not used here, but good practice)
valid_x = train_x[-10000:, :]
train_x = train_x[:-10000, :]
valid_labels = train_labels[-10000:]
train_labels = train_labels[:-10000]

# generate torch tensors
train_x = torch.tensor(train_x).to(device)
test_x = torch.tensor(test_x).to(device)

train_N, train_D = train_x.shape


class MLPDecoder(torch.nn.Module):
    def __init__(self, num_var, num_latent, num_neurons, var=0.05):
        super(MLPDecoder, self).__init__()

        self.num_var = num_var
        self.num_latent = num_latent
        self.num_neurons = num_neurons
        self.var = var

        # generate hidden layers
        # E.g., if num_neurons = [500, 1000, 500], then three hidden layers are generated,
        # with 500, 1000 and 500 neurons, respectively.
        layers = []
        num_units = [num_latent] + num_neurons
        for n_prev, n_next in zip(num_units[0:-1], num_units[1:]):
            layers.append(torch.nn.Linear(n_prev, n_next))
            layers.append(torch.nn.ReLU())
            # layers.append(torch.nn.BatchNorm1d(num_features=n_next))
        self.layers = torch.nn.ModuleList(layers)

        # generate output layers, mu
        self.mu = torch.nn.Linear(num_neurons[-1], num_var)

    def forward(self, z):
        res = z
        for layer in self.layers:
            res = layer(res)
        mu = self.mu(res)
        return mu

    def sample(self, N, convert_to_numpy=False, suppress_noise=True):
        with torch.no_grad():
            z = torch.randn(N, self.num_latent, device=device)
            mu = self.forward(z)
            x = mu
            # the conditional VAE distribution is isotropic Gaussian, hence we just add noise when sampling it
            # for images, one might want to suppress this
            if not suppress_noise:
                x += np.sqrt(self.var) * torch.randn(N, self.num_var, device=device)

        if convert_to_numpy:
            z = z.cpu().numpy()
            x = x.cpu().numpy()
        return x, z


class MLPEncoder(torch.nn.Module):
    def __init__(self, num_var, num_latent, num_neurons):
        super(MLPEncoder, self).__init__()

        self.num_var = num_var
        self.num_latent = num_latent
        self.num_neurons = num_neurons

        layers = []
        num_units = [num_var] + num_neurons
        for n_prev, n_next in zip(num_units[0:-1], num_units[1:]):
            layers.append(torch.nn.Linear(n_prev, n_next))
            layers.append(torch.nn.ReLU())
        self.layers = torch.nn.ModuleList(layers)

        # generate output layers, mu and var
        self.mu = torch.nn.Linear(num_neurons[-1], num_latent)
        self.var = torch.nn.Linear(num_neurons[-1], num_latent)
        self.var_act = torch.nn.Softplus()

    def forward(self, x):
        res = x
        for layer in self.layers:
            res = layer(res)
        mu = self.mu(res)
        var = self.var_act(self.var(res))
        return mu, var


decoder = MLPDecoder(train_x.shape[1], num_latent, num_neurons, var=var_x).to(device)
encoder = MLPEncoder(train_x.shape[1], num_latent, num_neurons).to(device)

print(decoder)
print(encoder)

# Adam optimizer, optimizes both decoder and encoder
optimizer = torch.optim.Adam(list(decoder.parameters()) + list(encoder.parameters()), lr=0.0001)

ELBO_history = []
for epoch in tqdm(range(num_epochs)):
    # make batches of training indices
    shuffled_idx = torch.randperm(train_x.shape[0])
    idx_batches = shuffled_idx.split(batch_size)

    sum_neg_ELBO = 0.0
    for batch_count, idx in enumerate(idx_batches):
        optimizer.zero_grad()

        batch_x = train_x[idx, :]

        # batch_mu_z: batch_size, num_latent
        # batch_var_z: batch_size, num_latent
        batch_mu_z, batch_var_z = encoder(batch_x)

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
    print('epoch {}   mean negative ELBO = {}'.format(epoch, mean_neg_ELBO))
    ELBO_history.append(mean_neg_ELBO.cpu().detach().numpy())

    if epoch % 5 == 0:
        with torch.no_grad():
            # sample from the VAE
            x, z = decoder.sample(5)

            # encode some samples
            mu_z, var_z = encoder(train_x[0:5, :])
            z_encoded = mu_z + torch.sqrt(var_z) * torch.randn(5, num_latent, device=device)

            # decode the samples
            x_decoded = decoder(z_encoded)

            # save images
            plot_img = np.stack((train_x[0:5, :].detach().cpu().numpy(),
                                 x_decoded.detach().cpu().numpy(),
                                 x.detach().cpu().numpy()))
            plot_img = np.reshape(plot_img, (15, 28, 28))
            file_name = os.path.join('img_vae_mlp', 'samples_{}.png'.format(epoch))
            datasets.save_image_stack(plot_img, 3, 5, file_name, margin=3)

        plt.figure(1)
        plt.clf()
        plt.plot(ELBO_history)
        plt.savefig(os.path.join('img_vae_mlp', 'elbo.png'))
        test_input = test_x
        plot_latent_space(encoder, test_input, test_labels)
        plot_interpolation(encoder, decoder, test_input, test_labels)
