from dataclasses import dataclass
import torch
@dataclass
class ModelResult:
    decoded: torch.Tensor = None
    sigma: torch.Tensor = None
    mask: torch.Tensor = None
    x_flattened: torch.Tensor = None
    time_prediction : torch.Tensor = None
    embeddings_highdim: torch.Tensor = None
    embeddings_2d: torch.Tensor = None
    embeddings_128: torch.Tensor = None
    embeddings_32: torch.Tensor = None