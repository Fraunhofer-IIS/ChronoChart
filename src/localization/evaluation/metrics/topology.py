"""Topology-preservation and chart-stress metrics."""

import random

import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances

from src.localization.evaluation.metrics.localization import cep_r95


def continuity(ground_truth: np.ndarray, chart: np.ndarray, n_neighbors: int) -> float:
    """Compute continuity by swapping inputs to trustworthiness."""
    return float(trustworthiness(chart, ground_truth, n_neighbors=n_neighbors))


def kruskal_stress(original: np.ndarray, embedded: np.ndarray, *, metric: str = "euclidean") -> float:
    """Compute scale-invariant Kruskal stress."""
    original_distances = pairwise_distances(original, metric=metric)
    embedded_distances = pairwise_distances(embedded, metric=metric)
    denominator = np.sum(embedded_distances**2)
    scale = np.sum(original_distances * embedded_distances) / denominator
    return float(
        np.sqrt(np.sum((original_distances - scale * embedded_distances) ** 2) / np.sum(original_distances**2))
    )


def calculate_metrics(
    channel_chart_positions: np.ndarray,
    groundtruth_positions: np.ndarray,
    *,
    sample_fraction: float = 0.1,
) -> dict[str, float]:
    """Return continuity, trustworthiness, stress, CEP, and R95 metrics."""
    chart = np.asarray(channel_chart_positions)
    ground_truth = np.asarray(groundtruth_positions)
    sample_count = max(3, min(len(ground_truth), int(len(ground_truth) * sample_fraction)))
    subset = np.asarray(random.sample(range(len(ground_truth)), sample_count))
    ground_truth_subset = ground_truth[subset]
    chart_subset = chart[subset]
    neighbors = max(1, min(sample_count - 1, int(0.05 * sample_count)))
    cep, r95 = cep_r95(chart, ground_truth)
    return {
        "CT": continuity(ground_truth_subset, chart_subset, neighbors),
        "TW": float(trustworthiness(ground_truth_subset, chart_subset, n_neighbors=neighbors)),
        "KS": kruskal_stress(ground_truth_subset, chart_subset),
        "cep": cep,
        "r95": r95,
    }
