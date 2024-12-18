import torch

def sample_gaussian_mixture(n_samples, means, covariances, weights, device="cpu"):
    """
    Sample from a Gaussian Mixture Model using PyTorch.
    """
    means = torch.tensor(means, device=device)
    covariances = torch.tensor(covariances, device=device)
    weights = torch.tensor(weights, device=device)
    
    components = torch.multinomial(weights, n_samples, replacement=True)
    samples = torch.cat([
        torch.distributions.MultivariateNormal(means[k], covariances[k]).sample((torch.sum(components == k),))
        for k in range(len(weights))
    ])
    labels = torch.cat([torch.full((torch.sum(components == k),), k, device=device) for k in range(len(weights))])
    return samples, labels