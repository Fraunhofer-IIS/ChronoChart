import torch
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass,asdict
from torch.utils.data.dataloader import default_collate
from src.localization.data.smoothing import smooth_signal

import torch
@dataclass
class SRT:
     samples: torch.Tensor
     reference: torch.Tensor
     times: torch.Tensor
     index: torch.Tensor


@dataclass
class TripletData:
    anchor_samples: SRT
    positive_samples: SRT
    negative_samples: SRT
    p1: SRT
    p2: SRT
    
from abc import ABC, abstractmethod

class BaseTripletDataset(Dataset):
    # This dataset class forms triplets based on time
    def __init__(self, csi_time_domain, timestamps, reference, K, temporal_window_seconds=None):
        self.timestamp_dissimilarity_matrix = None
        self.adp_dissimilarity_matrix = None
        self.dissimilarity_matrix_geodesic = None
        
        self.csi = csi_time_domain  
        self.timestamps = timestamps
        self.reference = reference
        

        
        

        self.positive_indexes = []
        self.negtive_indexes = []
        self.neighbor_negtive_indexes = []
        
        self.T = 0
        self.K = K
        self.temporal_window_seconds = temporal_window_seconds
        

        

        
    def _preprocess(self):
        if True:
            if isinstance(self.timestamps, np.ndarray):
                arg = np.argsort(self.timestamps, kind="stable")
            else:
                arg = torch.argsort(self.timestamps, descending=False, stable=True)
            self.csi = self.csi[arg]
            self.timestamps = self.timestamps[arg]
            self.reference = self.reference[arg]
        
        timestamp_span = float(self.timestamps.max() - self.timestamps.min())
        if self.temporal_window_seconds is None:
            self.T = 1 / len(self.timestamps) * self.K
        else:
            if self.temporal_window_seconds <= 0 or timestamp_span <= 0:
                raise ValueError("temporal_window_seconds and timestamp span must be positive")
            self.T = self.temporal_window_seconds / timestamp_span
        temporal_window_seconds = self.T * timestamp_span
        print(
            f"T before normalization (K={self.K}): "
            f"{temporal_window_seconds:.6f} seconds"
        )
        self.timestamps = self._normalize(self.timestamps)
        
        #self.timestamps = smooth_signal(self.timestamps,1000)
        
        self.timestamps = self.timestamps[...,None]
        self.timestamps =torch.from_numpy(self.timestamps).to(torch.float32) if isinstance(self.timestamps,np.ndarray) else self.timestamps
        self.reference = torch.from_numpy(self.reference).to(torch.float32) if isinstance(self.reference, np.ndarray) else self.reference
        
        
        self._make_triplets()

        
        print("T",self.T)
        print("len",len(self.csi))
        print("pos",len(self.positive_indexes))
        self._print_median()
        pass
        

    def _normalize(self,x):
        return (x - x.min()) / (x.max() - x.min())
    def _print_median(self,):
        m = [len(y)for y in self.positive_indexes]
        m = np.array(m)
        m = np.median(m)
        print("median",m)
        return m
    
    def _len__(self):
        return len(self.positive_indexes) #len(self.keep_idxs) # Or however many pairs you want

    def _getitem__(self, i):
        p = self.positive_indexes[i]
        N = len(self.csi)

        a_i = i
        if len(p) == 0:
            p_i = i
            n_i = np.random.choice(np.r_[0:i , i+1:N])

        else:
            p_i = np.random.choice(p)
            n_i = np.random.choice(np.r_[0:p[0] , p[-1]+1:N])


        samples = {
        "anchor_samples": a_i,
        "positive_samples": p_i,
        "negative_samples": n_i,
        "p1": p[0] if len(p) > 0 else i,
        "p2": p[1] if len(p) > 1 else i,
        }

        results = {}

        for name, index in samples.items():
            _csi = torch.from_numpy(self.csi[index]) 
            _time_stamp = self.timestamps[index]
            _reference = self.reference[index]
            results[name] = SRT(samples =_csi, reference = _reference, times =_time_stamp ,index=index)


        return TripletData(**results)

    def _find_mean(self,i):
        from scipy.spatial.distance import cdist

        
        p = self.positive_indexes[i]
        L = len(self.csi)
        if len(p)==0:
            neg = np.r_[0:i , i+1:L-1]
            p = i
            ref_pos = self.reference[p][None,...]
        else:
            #print(p)
            neg = np.r_[0:p[0] , p[-1]+1:L-1]
            ref_pos = self.reference[p]


        ref_anchor = self.reference[i][None,...]
        ref_neg = self.reference[neg]
        anchor_neg = cdist(ref_anchor,ref_neg,metric="euclidean")
        anchor_pos = cdist(ref_anchor,ref_pos,metric="euclidean")
        return anchor_neg,anchor_pos
        
    def cdist(self,e):
        self.d = torch.cdist(e,e)
    def min(self,i,r,p):
        
        c = self.d[i]
        pos_dist = c[p]            # distance to positive
        neg_dists = c[r]           # distances to candidate negatives
        mask = (neg_dists > pos_dist) & (neg_dists < pos_dist + self.margin)
        indexes = torch.nonzero(mask, as_tuple=False).squeeze()
        m = torch.mean(c[indexes])
        self.m.append(m)
        return indexes
        
    def random_min(self,i,r,p):
        
        c = torch.cdist(self.e[i],self.e)
        pos_dist = c[p]            # distance to positive
        neg_dists = c[r]           # distances to candidate negatives
        mask = (neg_dists > pos_dist) & (neg_dists < pos_dist + self.margin)
        indexes = torch.nonzero(mask, as_tuple=False).squeeze()
        return indexes


	
    def _make_triplets(self):
        self.positive_indexes = []
        self.negtive_indexes = []
        self.neighbor_negtive_indexes = []
        N = len(self.csi)
        i_min, i_max = 0, 1
        half_T = self.T / 2

        for i in range(N - 1):

            while i_min >= 0 and torch.abs(self.timestamps[i_min] - self.timestamps[i]) < half_T:
                i_min -= 1
            i_min += 1  

       
            while i_max < N and torch.abs(self.timestamps[i_max] - self.timestamps[i]) < half_T:
                i_max += 1
            i_max -= 1  

            self.positive_indexes.append(np.r_[i_min:i, i+1:i_max+1])

            i_min = i
            i_max = i + 2
	
    def collate_fn(self,batch):
        batch=[asdict(b) for b in batch]
        return default_collate(batch)
    
    
class TripletDataset(BaseTripletDataset):
    def __init__(self, csi_time_domain, timestamps, reference, K=5, temporal_window_seconds=None):
        super().__init__(csi_time_domain, timestamps, reference, K, temporal_window_seconds)
        self._preprocess()
        
    
    
    def __getitem__(self, index):
        return self._getitem__(index)
    
    def __len__(self):
        return self._len__()
