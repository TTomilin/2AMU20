from typing import List

import torch
import torch.distributions as dist


def categorical_kl(phi: torch.Tensor, device: str) -> torch.Tensor:
    """
    Computes the KL divergence between categorical distributions.
    :param phi: A tensor of shape (batch_size, n_classes, n_distributions) containing the logits of the categorical
    :param device: cpu/cuda
    :return:
    """
    # phi is logits of shape [batch_size, n_classes, n_distributions] where batch_size is batch, n_classes is number of categorical distributions, n_distributions is number of classes
    batch_size, n_distributions, n_classes = phi.shape
    phi = phi.view(batch_size * n_classes, n_distributions)
    q = dist.Categorical(logits=phi)
    p = dist.Categorical(probs=torch.full((batch_size * n_classes, n_distributions), 1.0 / n_distributions,
                                          device=device))  # uniform bunch of n_distributions-class categorical distributions
    kl = dist.kl.kl_divergence(q, p)  # kl is of shape [batch_size*n_classes]
    return kl.view(batch_size, n_classes)


def binomial_kl(z: torch.Tensor, device: str) -> torch.Tensor:
    """
    Computes the KL divergence between binomial distributions.
    :param z: A tensor of shape (batch_size, n_latent) containing the logits of the binomial
    :param device: cpu/cuda
    :return:
    """

    p = torch.distributions.Bernoulli(torch.full(z.shape, 0.5, device=device))
    q = torch.distributions.Bernoulli(logits=z)

    new_z = q.sample()

    log_qzx = q.log_prob(new_z)
    log_pz = p.log_prob(new_z)

    KL = (log_qzx - log_pz)
    KL = torch.mean(torch.sum(KL, dim=1))

    return KL


def kl_loss(mu_z, var_z):
    return torch.mean(0.5 * torch.sum(mu_z**2 + var_z**2 - torch.log(var_z**2) - 1, dim=1))


def log_beta_pdf(batch_x, alpha, beta, eps=1e-8):
    """
    Calculate the log pdf of a beta distribution
    :param batch_x: batch of samples from the beta distribution
    :param alpha: alpha parameter of the beta distribution
    :param beta: beta parameter of the beta distribution
    :param eps: epsilon to avoid numerical issues when taking logarithms
    :return: log pdf of the beta distribution
    """
    return torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta) + (alpha - 1) * torch.log(
        batch_x + eps) + (beta - 1) * torch.log(1 - batch_x + eps)


class EarlyStopping:
    def __init__(self, threshold: float, patience=3):
        """
        :param threshold: The loss decrease threshold to consider for early stopping
        :param patience: The number of epochs to wait before stopping
        """
        self.threshold = threshold
        self.patience = patience
        self.last_loss = float("inf")
        self.counter = 0

    def __call__(self, loss: float) -> bool:
        """
        :param loss: The current loss
        :return: True if the training should stop, False otherwise
        """
        if loss >= self.last_loss or self.last_loss - loss < self.threshold:
            self.counter += 1
        else:
            self.counter = 0
        self.last_loss = loss
        return self.counter >= self.patience
