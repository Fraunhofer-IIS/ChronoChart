
from src.localization.config.presets import config_dichasus
from src.localization.config.training import TrainingConfig
from src.localization.training.trainer import run
from copy import deepcopy

def run_upgrad(config:TrainingConfig, K=10):    
    config_list = [  ]
    
    coeff_mse_loss_list = [0.0,0.5,1.0,1.5]
    coeff_dissimilarity_loss_list = [0.0,0.5,1.0,1.5]
    
    for mse in coeff_mse_loss_list:
        for diss in coeff_dissimilarity_loss_list:
                for i in range(K):
                    newcfg = deepcopy(config)
                    newcfg.coeff_mse_loss = mse
                    newcfg.coeff_dissimilarity_loss = diss
                    #newcfg.aggregator_name = "UPGrad"
                    config_list.append(newcfg)
            

    for i, cfg in enumerate(config_list): 
        run(cfg)
        
        
        print("-"*10,f"run {i} done","-"*10)
        
        
if __name__ == "__main__":
    config_dichasus.mask_ratio_train = 0.9
    
    config_dichasus.train_batch_size = 256
    config_dichasus.val_batch_size = 256
    config_dichasus.max_epochs = 200
    config_dichasus.deterministic = True
    config_dichasus.learning_rate = 0.001
    config_dichasus.val_check_interval=10
    config_dichasus.loss_scheduler_start = 0.000001
    config_dichasus.loss_scheduler_steps = 400
    #config_dichasus.aggregator_name = "AlignedMTL"
    config_dichasus.aggregator_name = "UPGrad"
    
    run_upgrad(config_dichasus,1)
