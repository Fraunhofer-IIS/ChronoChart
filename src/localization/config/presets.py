
from src.localization.config.training import TrainingConfig
from src.localization.config.paths import PathConfig
path_config = PathConfig()
config_dichasus = TrainingConfig(
    model_name = "dichasus",
    K = 40,
)

config_5g = TrainingConfig(
    model_name = "5G",
    K=4,
    max_epochs=300,
)

config_passive = TrainingConfig(
    model_name = "passive",
    K=50,
    learning_rate=0.0001,
    train_batch_size=512,
    val_batch_size=512,
    max_epochs=150,
    loss_scheduler_start=0.0001,
    loss_scheduler_steps=100,
)

config_custom = TrainingConfig(
    model_name = "custom",
    custom_model = "dichasus", # 5G / dichasus
    K=4,
    max_epochs=300,
    custom_path = "path/to/data"
)