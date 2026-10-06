"""Localization-error plots."""

import numpy as np
import matplotlib.pyplot as plt

from src.localization.evaluation.alignment import affine_transform_channel_chart
from src.localization.evaluation.metrics.localization import localization_error_metrics
from src.localization.evaluation.visualization.embeddings import plot_colorized


def plot_error_vectors(prediction, ground_truth, *, stride: int = 15):
    """Plot prediction-to-reference error vectors and return error metrics."""
    prediction = np.asarray(prediction)
    ground_truth = np.asarray(ground_truth)
    metrics = localization_error_metrics(prediction, ground_truth)
    figure = plot_colorized(
        prediction,
        ground_truth,
        title=f"DRMS = {metrics['DRMS']:.3f}m | Euclidean = {metrics['Euclidean']:.3f}m",
        show=False,
        alpha=0.3,
    )
    axis = figure.axes[0]
    error_vectors = ground_truth - prediction
    axis.quiver(
        prediction[::stride, 0],
        prediction[::stride, 1],
        error_vectors[::stride, 0],
        error_vectors[::stride, 1],
        color="black",
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    return figure, {"DRMS": round(metrics["DRMS"], 3), "Euclidean": round(metrics["Euclidean"], 3)}


def plot_error_cdf(prediction, ground_truth, *, bins: int = 200, x_limit=(0, 2)):
    """Plot the empirical CDF of Euclidean localization errors."""
    errors = np.linalg.norm(np.asarray(ground_truth) - np.asarray(prediction), axis=1)
    counts, edges = np.histogram(errors, bins=bins)
    cdf = np.cumsum(counts / counts.sum())
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(np.concatenate(([0], edges[1:])), np.concatenate(([0], cdf)))
    axis.set(xlim=x_limit, xlabel="Absolute Localization Error [m]", ylabel="CDF")
    axis.grid()
    return figure


def plot_all(embeddings, ground_truth):
    """Align an embedding and create the standard evaluation plot set."""
    aligned = affine_transform_channel_chart(embeddings, ground_truth)
    return {
        "transformed": plot_colorized(aligned, ground_truth, title="Transformed"),
        "reference": plot_colorized(ground_truth, ground_truth, title="Reference"),
        "cdf": plot_error_cdf(aligned, ground_truth),
        "vectors": plot_error_vectors(aligned, ground_truth)[0],
    }
