from src.localization.config.presets import config_dichasus
from src.localization.config.training import TrainingConfig
from src.localization.training.trainer import run

def run_pcgrad(config:TrainingConfig, K=15):
    
    #config_dichasus , config_5g , config_jonas , config_max , config_passive
    #config = config_dichasus
    
    for i in range(0 , K):
        run(config)


if __name__ == "__main__":
    
    
    config_dichasus.mask_ratio_train = 0.01
    config_dichasus.max_epochs = 400
    config_dichasus.train_batch_size = 256 
    config_dichasus.learning_rate = 0.001
    config_dichasus.val_batch_size = 256
    #config_dichasus.loss_scheduler_start = 0.000001
    config_dichasus.loss_scheduler_steps= 200
    config_dichasus.deterministic = False
    config_dichasus.coeff_time_loss = 0.0
    config_dichasus.coeff_mse_loss = 1.0
    #config_dichasus.coeff_triplet_loss = 0.0
    #config_dichasus.loss_scheduler_start = 0.0
    config_dichasus.coeff_dissimilarity_loss = 1.0
    
    config_dichasus.aggregator_name = "AlignedMTL"
    #config_dichasus.aggregator_name = "Mean"
    
    #config_dichasus.train_batch_size = 64
    #config_dichasus.val_batch_size = 64
    config_dichasus.val_check_interval = 5
    run_pcgrad(config_dichasus)
