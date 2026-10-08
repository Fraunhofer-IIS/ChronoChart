#%%
from abc import abstractmethod
from src.localization.models.cir import CIRModel
from src.localization.models.csi import CSIModel
from src.localization.data.triplet_dataset_with_adp import TripletDatasetWithADP
from src.localization.data.triplet_dataset import TripletDataset
import numpy as np
import pandas as pd
from src.localization.config.paths import PathConfig
from src.localization.config.training import TrainingConfig
import torch
import h5py
import pickle

class BaseLoader():
    def __init__(self,config:TrainingConfig):
        self.config: TrainingConfig = config
        self.path_config: PathConfig = config.path_config
        pass
    
    @abstractmethod
    def load_model(self):
        pass
    def load_dataset(self):
        pass

class CustomLoader(BaseLoader):
    def __init__(self,config:TrainingConfig):
        super().__init__(config)
        
    def load_model(self):
        if self.config.custom_model == "5G":
            return CIRModel()
        elif self.config.custom_model == "dichasus":
            return CSIModel()
        else:
            raise Exception("Unknown model " + self.config.custom_model)
    
    def load_dataset(self):
        x = pd.read_pickle(self.config.custom_path)
        
        cir = np.array(x[0])
        reference = torch.from_numpy(x[1]).float()
        timestamp = torch.from_numpy(x[2]).float()

        print("cir shape",cir.shape)
        print("reference shape",reference.shape)
        print("timestamp shape",timestamp.shape)
        dataset = TripletDataset(
            csi_time_domain=cir, timestamps=timestamp, reference=reference, K=self.config.K,
            temporal_window_seconds=self.config.temporal_window_seconds,
        )

        return dataset

class DichasusLoader(BaseLoader):
    
    def __init__(self,config:TrainingConfig):
        super().__init__(config)
        
    def load_model(self):
        return CSIModel()
    
    def load_dataset(self):
        return TripletDatasetWithADP(self.config.K, self.config.temporal_window_seconds)


class FiveGLoader(BaseLoader):
    def __init__(self,config:TrainingConfig):
        super().__init__(config)
        
    def load_model(self):
        return CIRModel()
    
    def load_dataset(self):
        train = True
        if train:
            x = pd.read_pickle(self.path_config.fiveg_train)
        else:
            x = pd.read_pickle(self.path_config.fiveg_test)
        
        cir = np.array(x[0])
        reference = torch.from_numpy(x[1]).float()
        timestamp = torch.from_numpy(x[2]).float()

        print("cir shape",cir.shape)
        print("reference shape",reference.shape)
        print("timestamp shape",timestamp.shape)
        dataset = TripletDataset(
            csi_time_domain=cir, timestamps=timestamp, reference=reference, K=self.config.K,
            temporal_window_seconds=self.config.temporal_window_seconds,
        )

        return dataset
    
    
class PassiveLoader(BaseLoader):
    def __init__(self,config:TrainingConfig):
        super().__init__(config)
        
    def load_model(self):
        return CIRModel(stride=1)
    def load_arrays(self, split="train"):
        """Load raw passive arrays while preserving timestamps in seconds."""
        paths = {
            "train": self.path_config.passive_train,
            "test": self.path_config.passive_p2_test,
            "test1": self.path_config.passive_p1_test,
            "test2": self.path_config.passive_p2_test,
        }
        if split not in paths:
            raise ValueError(f"unknown passive split {split!r}; choose from {tuple(paths)}")
        with h5py.File(paths[split], "r") as f:
            data = f["fp"][:]   # list/array of tuples
            print(data.shape)
            print(data.dtype)

        df = pd.DataFrame({
            "cir": list(data["CIR"]),
            "timestamp": data["timestamp"],
            "reference": list(data["reference"])  # keeping your spelling
        })

        ref_df = pd.DataFrame(
            df["reference"].to_list(),
            columns=pd.MultiIndex.from_product([["reference"], [0, 1]]),
            index=df.index
        )

        df = pd.concat([df, ref_df], axis=1)
        df["x"] = df[("reference", 0)]
        df["y"] = df[("reference", 1)]
        df = df.drop(columns=[("reference", 0),("reference", 1)])



        cir = df["cir"].to_numpy()
        reference = df["reference"].to_numpy()
        timestamp = df["timestamp"].to_numpy()

        cir = np.stack(cir).astype(np.float32)  
        reference = np.stack(reference)
        timestamp = np.stack(timestamp)

        return cir, reference, timestamp

    def load_dataset(self):
        cir, reference, timestamp = self.load_arrays("train")
        reference = torch.from_numpy(reference)
        timestamp = torch.from_numpy(timestamp)


        print("cir shape",cir.shape)
        print("reference shape",reference.shape)
        print("timestamp shape",timestamp.shape)
        dataset = TripletDataset(
            csi_time_domain=cir, timestamps=timestamp, reference=reference, K=self.config.K,
            temporal_window_seconds=self.config.temporal_window_seconds,
        )

        return dataset