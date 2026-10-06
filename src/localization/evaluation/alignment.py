"""Affine alignment between learned embeddings and physical coordinates."""

import numpy as np


def affine_transform_channel_chart(
    channel_chart_positions: np.ndarray,
    ground_truth_positions: np.ndarray,
) -> np.ndarray:
    """Fit and apply an affine map from chart coordinates to ground truth."""
    chart = np.asarray(channel_chart_positions)
    ground_truth = np.asarray(ground_truth_positions)
    padded_chart = np.column_stack((chart, np.ones(len(chart))))
    padded_ground_truth = np.column_stack((ground_truth, np.ones(len(ground_truth))))
    transform = np.linalg.lstsq(padded_chart, padded_ground_truth, rcond=None)[0]
    return (padded_chart @ transform)[:, :-1]


def affine_transform_channel_chart_torch(channel_chart_pos, groundtruth_pos, values):
    """Fit an affine map with Torch and apply it to additional values."""
    import torch

    chart = channel_chart_pos.float()
    ground_truth = groundtruth_pos.float()
    values = values.float()

    def pad(tensor):
        ones = torch.ones((len(tensor), 1), dtype=tensor.dtype, device=tensor.device)
        return torch.cat((tensor, ones), dim=1)

    transform = torch.linalg.lstsq(pad(chart), pad(ground_truth)).solution
    return (pad(values) @ transform)[:, :-1]
