"""Physical localization-error metrics."""

import numpy as np


def localization_error_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    """Return mean, DRMS, median/CEP, and R95 Euclidean errors."""
    errors = np.linalg.norm(np.asarray(prediction) - np.asarray(ground_truth), axis=1)
    return {
        "Euclidean": float(np.mean(errors)),
        "DRMS": float(np.sqrt(np.mean(errors**2))),
        "cep": float(np.percentile(errors, 50)),
        "r95": float(np.percentile(errors, 95)),
    }


def cep_r95(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Return median/CEP and 95th-percentile Euclidean error."""
    metrics = localization_error_metrics(pred, gt)
    return metrics["cep"], metrics["r95"]
