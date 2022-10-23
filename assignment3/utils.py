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
