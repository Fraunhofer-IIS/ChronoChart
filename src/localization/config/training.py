
from dataclasses import dataclass,field
from pathlib import Path
from src.localization.config.paths import PathConfig

@dataclass
class TrainingConfig:
    """Configuration for triplet learning training"""
    model_name: str = "dichasus" #"5g"
    path_config: PathConfig =  field(default_factory=PathConfig)
    K: int = 5
    temporal_window_seconds: float | None = None
    
    deterministic: bool = False
    # Optimizer settings
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    
    # Loss coefficients
    coeff_triplet_loss: float = 1.0
    coeff_mse_loss: float = 1.0
    coeff_dissimilarity_loss: float = 1.0
    coeff_time_loss: float = 0.0
    
    # Training settings
    train_batch_size: int = 128
    val_batch_size: int = 128
    max_epochs: int = 200
    
    
    # Model settings
    mask_ratio_train: float = 0.9
    mask_ratio_val: float = 0.0
    
    
    loss_scheduler_start: float = 0.00001
    loss_scheduler_steps: int = 100
    
    # Data settings
    num_workers: int = 8
    prefetch_factor: int = 1
    
    # Validation settings
    val_check_interval: int = 5  # Check every N epochs
    val_samples: int = 4096
    
    aggregator_name:str = "NashMTL"

