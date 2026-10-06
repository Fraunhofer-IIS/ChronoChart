"""Plots for embeddings and reference coordinates."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.neighbors import KNeighborsClassifier


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _normalize(values: np.ndarray) -> np.ndarray:
    span = np.max(values) - np.min(values)
    return np.zeros_like(values) if span == 0 else (values - np.min(values)) / span


def plot_colorized(positions, groundtruth_positions, title=None, show=True, alpha=1.0):
    """Scatter positions using colors derived from ground-truth coordinates."""
    positions = _as_numpy(positions)
    ground_truth = _as_numpy(groundtruth_positions)
    center = 0.5 * (ground_truth.min(axis=0) + ground_truth.max(axis=0))
    colors = np.zeros((len(ground_truth), 3))
    colors[:, 0] = 1 - 0.9 * _normalize(ground_truth[:, 0])
    colors[:, 1] = 0.8 * _normalize(np.linalg.norm(ground_truth - center, axis=1) ** 2)
    colors[:, 2] = 0.9 * _normalize(ground_truth[:, 1])
    figure, axis = plt.subplots(figsize=(6, 6))
    if title is not None:
        axis.set_title(title, fontsize=16)
    axis.scatter(positions[:, 0], positions[:, 1], c=colors, alpha=alpha, s=10, linewidths=0)
    axis.set_xlabel("x coordinate")
    axis.set_ylabel("y coordinate")
    if show:
        plt.show()
    return figure


def plot_voronoi_with_knn(embeddings, centroid_indices, reference=None):
    """Plot a Voronoi partition induced by selected embedding centroids."""
    embeddings = np.asarray(embeddings)
    centroids = embeddings[centroid_indices]
    classifier = KNeighborsClassifier(n_neighbors=1).fit(centroids, np.arange(len(centroids)))
    labels = classifier.predict(embeddings)
    points = embeddings if reference is None else np.asarray(reference)
    plotted_centroids = centroids if reference is None else points[centroid_indices]
    figure, axis = plt.subplots(figsize=(8, 6))
    voronoi_plot_2d(Voronoi(plotted_centroids), ax=axis, show_vertices=False)
    axis.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=20)
    axis.scatter(plotted_centroids[:, 0], plotted_centroids[:, 1], c="black", s=100, marker="x")
    return figure


def plot_2d_scatter_time_color(
    x,
    y,
    time,
    title="2D Scatter Plot with Time as Color",
    xlabel="X-coordinate",
    ylabel="Y-coordinate",
    time_label="Time",
    cmap="viridis",
    marker="o",
    alpha=0.8,
    figsize=(10, 8),
    show=True,
):
    """Plot coordinates colored by time."""
    figure, axis = plt.subplots(figsize=figsize)
    scatter = axis.scatter(x, y, c=time, cmap=cmap, marker=marker, alpha=alpha)
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    figure.colorbar(scatter, ax=axis, shrink=0.75, aspect=10).set_label(time_label)
    if show:
        plt.show()
    return figure
