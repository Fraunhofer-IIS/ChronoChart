"""Public numerical evaluation API."""

from src.localization.evaluation.alignment import (
    affine_transform_channel_chart,
    affine_transform_channel_chart_torch,
)
from src.localization.evaluation.metrics.localization import (
    cep_r95,
    localization_error_metrics,
)
from src.localization.evaluation.metrics.tensor import pairwise_mean_sq_diff_broadcast


_LAZY_EXPORTS = {
    "calculate_metrics": ("src.localization.evaluation.metrics.topology", "calculate_metrics"),
    "continuity": ("src.localization.evaluation.metrics.topology", "continuity"),
    "kruskal_stress": ("src.localization.evaluation.metrics.topology", "kruskal_stress"),
    "plot_2d_scatter_time_color": (
        "src.localization.evaluation.visualization.embeddings",
        "plot_2d_scatter_time_color",
    ),
    "plot_colorized": ("src.localization.evaluation.visualization.embeddings", "plot_colorized"),
    "plot_voronoi_with_knn": (
        "src.localization.evaluation.visualization.embeddings",
        "plot_voronoi_with_knn",
    ),
    "plot_all": ("src.localization.evaluation.visualization.errors", "plot_all"),
    "plot_error_cdf": ("src.localization.evaluation.visualization.errors", "plot_error_cdf"),
    "plot_error_vectors": (
        "src.localization.evaluation.visualization.errors",
        "plot_error_vectors",
    ),
    # Compatibility for callers using the historical misspelling.
    "plot_error_vacotrs": (
        "localization.evaluation.visualization.errors",
        "plot_error_vectors",
    ),
}


def __getattr__(name: str):
    """Load plotting and scikit-learn-backed metrics only when requested."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module_name, attribute = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value

__all__ = [
    "affine_transform_channel_chart",
    "affine_transform_channel_chart_torch",
    "calculate_metrics",
    "cep_r95",
    "continuity",
    "kruskal_stress",
    "localization_error_metrics",
    "pairwise_mean_sq_diff_broadcast",
    "plot_2d_scatter_time_color",
    "plot_all",
    "plot_colorized",
    "plot_error_cdf",
    "plot_error_vectors",
    "plot_error_vacotrs",
    "plot_voronoi_with_knn",
]
