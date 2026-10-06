"""Tensor-specific evaluation helpers."""


def pairwise_mean_sq_diff_broadcast(values):
    """Return pairwise mean squared differences for a two-dimensional tensor."""
    return (values.unsqueeze(1) - values.unsqueeze(0)).pow(2).mean(dim=2)
