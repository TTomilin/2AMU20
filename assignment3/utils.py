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
    p = dist.Categorical(probs=torch.full((batch_size * n_classes, n_distributions), 1.0 / n_distributions, device=device))  # uniform bunch of n_distributions-class categorical distributions
    kl = dist.kl.kl_divergence(q, p)  # kl is of shape [batch_size*n_classes]
    return kl.view(batch_size, n_classes)
