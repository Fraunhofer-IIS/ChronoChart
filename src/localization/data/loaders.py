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

        """cir = np.array(x[0])
        reference = np.array(x[1])
        timestamp = np.array(x[2])"""
        
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
    
    
class JonasLoader(BaseLoader):
    def __init__(self,config:TrainingConfig):
        super().__init__(config)
        
    def load_model(self):
        return CIRModel()
    def load_dataset(self):

    


        file_path = self.path_config.jonas_list[0]

        import h5py
        import pandas as pd

        with h5py.File(file_path, "r") as f:
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
        timestamps = df["timestamp"].to_numpy()

        cir = np.stack(cir)#.astype(np.float32)  
        reference = np.stack(reference)#.astype(np.float32)  
        timestamps = np.stack(timestamps)#.astype(np.float32)  
        
        return TripletDataset(cir, timestamps, reference, self.config.K, self.config.temporal_window_seconds)
    
    
class MaxLoader(BaseLoader):
    def __init__(self,config:TrainingConfig):
        super().__init__(config)
        
    def load_model(self):
        return CIRModel()
    def load_dataset(self):


        data = torch.load(self.path_config.max_pdr)

        data = data["fp"]


        

        cir = data["CIR"]
        reference = data["reference"]
        timestamp = data["timestamp"]

        cir = np.stack(cir)
        reference = np.stack(reference)
        timestamp = np.stack(timestamp)


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


class UncertaintyLoader(BaseLoader):
    """Load the pickled Industrial uncertainty-localization dataset."""

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        # Keep scenario selection in the training config, while retaining the
        # environment-variable default supplied by PathConfig.
        self.path_config.uncertainty_scenario = config.uncertainty_scenario

    def load_model(self):
        return CIRModel()

    def _load_arrays(self, path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Uncertainty dataset split does not exist: {path}. "
                "Set UNCERTAINTY_DATA_ROOT or choose a valid scenario."
            )
        with path.open("rb") as file:
            data = pickle.load(file)
        if not isinstance(data, (list, tuple)) or len(data) != 3:
            raise ValueError(f"Expected [cir, reference, timestamps] in {path}")

        cir, reference, timestamps = (np.asarray(value) for value in data)
        if cir.ndim != 4 or reference.ndim != 2 or timestamps.ndim != 1:
            raise ValueError(
                f"Unexpected uncertainty split shapes in {path}: "
                f"{cir.shape}, {reference.shape}, {timestamps.shape}"
            )
        if not (len(cir) == len(reference) == len(timestamps)):
            raise ValueError(f"Uncertainty split arrays have different lengths in {path}")
        return (
            cir.astype(np.float32, copy=False),
            reference.astype(np.float32, copy=False),
            timestamps,
        )

    def _make_dataset(self, path):
        cir, reference, timestamps = self._load_arrays(path)
        return TripletDataset(
            cir,
            torch.from_numpy(timestamps).float(),
            torch.from_numpy(reference).float(),
            K=self.config.K,
            temporal_window_seconds=self.config.temporal_window_seconds,
        )

    def load_dataset(self):
        return self._make_dataset(self.path_config.uncertainty_train)

    def load_validation_dataset(self):
        return self._make_dataset(self.path_config.uncertainty_validation)
    


if __name__=="__main__":
    pass
    #%%
    from src.localization.config.paths import PathConfig
    pc = PathConfig()
    #loader = [DichasusLoader(),FiveGLoader(),JonasLoader(),MaxLoader(),PassiveLoader()]
    
    dataset = DichasusLoader(TrainingConfig()).load_dataset()
    import cProfile
    cProfile.run("dataset[0]")
    #%%
    for l in loader:
        l:BaseLoader
        print(l.__class__)
        dataset = l.load_dataset()
        print(len(dataset))
    
    
    
    #%%    
    def _diff(x1,x2):
        diff = x1 - x2               # shape: (N, D)
        sq_diff = diff ** 2          # square element-wise
        sum_sq = torch.mean(sq_diff, dim=1, keepdim=True)   # sum over D

        return sum_sq
    
    loader = DichasusLoader()
    dataset = loader.load_dataset()
    gt = dataset.reference
    t = dataset.timestamps
    gt = dataset._normalize(gt)
    #%%
    import matplotlib.pyplot as plt
    from src.localization.evaluation.metrics import plot_colorized
    plot_colorized(gt,gt)    
    #%%
    c = _diff(gt[:-1],gt[1:])
    ct =_diff( t[:-1] , t[1:] )
    #%%
    import numpy as np

    import numpy as np

    t_all = []
    r_all = []

    for d in dataset:
        t = d.anchor_samples.times - d.positive_samples.times   # (N,)
        r = d.anchor_samples.reference - d.positive_samples.reference  # (N, D)

        t_all.append(t)
        
        # reduce r to (N,)
        r = (r**2).mean()
        r_all.append(np.atleast_1d(r))

    # concatenate

    
    t_all = np.concatenate(t_all)   # (N_total,)
    r_all = np.concatenate(r_all)   # (N_total,)

    # compute
    dt2 = t_all**2                  # (N,)
    numerator = np.sum(r_all * dt2)
    denominator = np.sum(dt2**2)

    k = np.sqrt(numerator / denominator)

    print("Optimal k:", k)
    
    #%%
    
    #%%
    min = gt.min(dim=0, keepdim=True)[0]
    max = gt.max(dim=0, keepdim=True)[0]
    gt = (gt - min) / (max - min)
    
    gt
    #%%
    fig = plot_colorized(gt,gt)    
    import matplotlib.pyplot as plt
    ax = fig.axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Remove axis, ticks, frame
    ax.set_axis_off()

    # Remove all margins
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save without border or padding
    fig.savefig(
        #pc.map_5g,
        pc.map_dichasus,
        
        dpi=300,
        bbox_inches="tight",
        pad_inches=0
    )
    #%%
        
    """loader = FiveGLoader()
    dataset = loader.load_dataset()
    known = [0,1000,8000]
    known_csi =  dataset.csi[known]
    known_reference = dataset.reference[known]"""
    pass
    
