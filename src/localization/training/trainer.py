"""
Triplet Learning Module for CSI-based Localization
Clean refactored version with configuration class
"""
import os
if "ROOT_DIR" not in os.environ:
    os.environ["ROOT_DIR"] = "/var/tmp/localization"


from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Any
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler
import torch.nn.utils as utils

from src.localization.evaluation.metrics import (
    affine_transform_channel_chart,
    calculate_metrics,
    plot_colorized,
    plot_error_cdf,
    plot_error_vectors,
    affine_transform_channel_chart_torch,
    plot_2d_scatter_time_color,
)
from pytorch_metric_learning.distances import LpDistance

from omegaconf import OmegaConf
import torch.nn.functional as F

import ot
from torchjd.aggregation import (
    AlignedMTL,
    CAGrad,
    ConFIG,
    Flattening,
    GradDrop,
    IMTLG,
    Mean,
    MGDA,
    NashMTL,
    PCGrad,
    UPGrad,
    UPGradWeighting,
)
from torchjd.autogram import Engine
from torchjd.autojac import backward, mtl_backward
from src.localization.models.base import ModelResult

from src.localization.data.loaders import (
    DichasusLoader,
    FiveGLoader,
    PassiveLoader,
    CustomLoader,
)
from src.localization.config.training import TrainingConfig
from copy import deepcopy
from src.localization.config.paths import PathConfig


class MarginScheduler:
    """Linear scheduler for margin values"""

    def __init__(self, start: float, end: float, total_steps: int):
        self.start = start
        self.end = end
        self.total_steps = total_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1

    def get_margin(self) -> float:
        ratio = min(self.step_num / self.total_steps, 1.0)
        return self.start + ratio * (self.end - self.start)


class SoftTripletLoss(nn.Module):
    """Soft triplet loss using log-sum-exp"""

    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:

        d_pos = (anchor - positive).pow(2).sum(1)
        d_neg = (anchor - negative).pow(2).sum(1)


        t = 1.0

        loss = -torch.log(
            torch.exp(-d_pos / t)
            / (torch.exp(-d_pos / t) + torch.exp(-d_neg / t) + 1e-8)
            + 1e-8
        )
        norm = torch.norm(anchor, dim=1)
        L2 = norm.pow(2) * 1e-6# L2 regularization on anchor embeddings
        L3 =  -torch.log(norm + 1e-8).mean() * 1e-6 # Encourage non-zero norm to avoid collapse

        loss = loss + L2 
        
        #return losses[:, None]
        return loss


class TripletLearningModule(LightningModule):
    """Main training module for triplet learning"""

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.save_hyperparameters()
        self.max = torch.tensor(1.0, requires_grad=False)
        self.min = torch.tensor(0.0, requires_grad=False)

        self.config = config
        self.path_config = PathConfig()

        loader_dict = {
            "dichasus": DichasusLoader(self.config),
            "5G": FiveGLoader(self.config),
            "passive": PassiveLoader(self.config),
            "custom": CustomLoader(self.config)
        }

        # Model
        print(config.model_name)

        self.loader: BaseLoader = loader_dict[self.config.model_name]
        self.dataset = self.loader.load_dataset()
        self.model = self.loader.load_model()

        self.transform = None

        # Loss functions
        self.triplet_loss = SoftTripletLoss()
        self.huber = nn.HuberLoss(reduction="none", delta=1.0)

        # Schedulers
        self.margin_scheduler = None
        self.loss_coeff_scheduler = None

        self.loss_coeff_scheduler = MarginScheduler(
            config.loss_scheduler_start,
            config.coeff_triplet_loss,
            config.loss_scheduler_steps,
        )
        self.ds_scheduler = MarginScheduler(self.dataset.T,self.dataset.T//4,10)
        self.train_dataset = None
        self.val_dataset = None
        self.indices = None

        self.reset_validation_buffers()
        self.save_hyperparameters(OmegaConf.structured(config))

        self.Y_i = None
        self.automatic_optimization = False
        self.pref_vector = []
        for coeff in [
            self.config.coeff_triplet_loss,
            self.config.coeff_dissimilarity_loss,
            self.config.coeff_mse_loss,
        ]:
            if coeff > 0.0:
                self.pref_vector.append(coeff)

        # self.pref_vector.append(1.0)  # Add weight for potential additional loss
        self.pref_vector = torch.tensor(self.pref_vector, device=torch.device("cuda"))
        self.pref_vector = self.pref_vector / self.pref_vector.sum()
        aggregator_dict = {
            "PCGrad": PCGrad(),
            "NashMTL": NashMTL(len(self.pref_vector), optim_niter=20),
            "CAGrad": CAGrad(0.1),
            "IMTLG": IMTLG(),
            "GradDrop": GradDrop(),
            "MGDA": MGDA(),
            "AlignedMTL": AlignedMTL(self.pref_vector),
            "ConFIG": ConFIG(self.pref_vector),
            "UPGrad": UPGrad(self.pref_vector),
            "Mean": Mean(),
            
            
            
        }

        self.aggregator = aggregator_dict[self.config.aggregator_name]

        self.reduction = False

        self.alpha = torch.nn.Parameter(
            torch.tensor(6.930, device=torch.device("cuda")), requires_grad=True
        )
        self.alpha = torch.nn.Parameter(
            torch.tensor(3.65, device=torch.device("cuda")), requires_grad=True
        )

        self.beta = torch.nn.Parameter(
            torch.tensor(0.0, device=torch.device("cuda")), requires_grad=True
        )

    def l2_regularization(self, result: ModelResult) -> torch.Tensor:
        return torch.norm(result["anchor"].embeddings_2d, dim=1).pow(2)

    def reset_validation_buffers(self):
        """Reset validation epoch buffers"""
        self.val_embeddings_2d = []
        self.val_time_prediction = []
        self.ground_truth = []
        self.time_truth = []
        self.e = []

    def forward(self, x: torch.Tensor, t=None, index=None) -> ModelResult:

        return self.model(x, t, index)

    def compute_triplet_loss(
        self, batch: dict
    ) -> Tuple[torch.Tensor, dict[ModelResult]]:
        """Compute triplet loss for anchor, positive, negative samples"""

        self.model.mask_ratio = 0.0
        result = {}
        result["anchor"] = self.forward(
            batch["anchor_samples"]["samples"],
            batch["anchor_samples"]["times"],
            batch["anchor_samples"]["index"],
        )
        result["positive"] = self.forward(
            batch["positive_samples"]["samples"],
            batch["positive_samples"]["times"],
            batch["positive_samples"]["index"],
        )
        result["negative"] = self.forward(
            batch["negative_samples"]["samples"],
            batch["negative_samples"]["times"],
            batch["negative_samples"]["index"],
        )

        # Compute loss

        """loss_3 = self.triplet_loss(result["anchor"].embeddings_32, 
                                 result["positive"].embeddings_32, 
                                 result["negative"].embeddings_32)"""

        loss = self.triplet_loss(
            result["anchor"].embeddings_2d,
            result["positive"].embeddings_2d,
            result["negative"].embeddings_2d,
        )
        loss = loss.unsqueeze(dim=1)


        return loss, result

    def compute_reconstruction_loss(
        self, batch: dict, prev_result: dict[ModelResult]
    ) -> torch.Tensor:
        """Compute MSE reconstruction loss"""

        # Concatenate samples

        self.model.mask_ratio = self.config.mask_ratio_train
        result = {}
        result2 = {}
        result["anchor"] = self.forward(
            batch["anchor_samples"]["samples"],
            batch["anchor_samples"]["times"],
            batch["anchor_samples"]["index"],
        )
        """result2["anchor"] = self.forward(
            batch["anchor_samples"]["samples"],
            batch["anchor_samples"]["times"],
            batch["anchor_samples"]["index"],
        )"""

        self.model.mask_ratio = 0.0

        n_2d = result["anchor"].embeddings_2d
        prev_anchor = prev_result["anchor"].embeddings_2d
        
        n_h = result["anchor"].embeddings_highdim
        prev_anchor_h = prev_result["anchor"].embeddings_highdim





        def covariance_loss(z):
            # z: (N, D)
            z = z - z.mean(dim=0, keepdim=True)  # center

            N = z.size(0)
            cov = (z.T @ z) / N  # (D, D)

            off_diag = cov - torch.diag(torch.diag(cov))
            return (off_diag**2).sum()

        def variance_loss(z, dim=0):
            std = z.std(dim=dim)
            return torch.mean(torch.relu(1 - std))


        def D(p, z):  # negative cosine similarity
            z = z.detach()  # stop gradient
            p = F.normalize(p, dim=1)  # l2-normalize
            z = F.normalize(z, dim=1)  # l2-normalize

            return torch.pairwise_distance(p, z).mean()


        self.log("abs", prev_anchor.abs().sum(1).mean())



        loss = (n_2d - prev_anchor).pow(2)
        
        



        return loss, result

    def compute_dissimilarity_loss(
        self, batch: dict, result: dict[ModelResult]
    ) -> torch.Tensor:
        """Compute dissimilarity loss between embeddings and time"""

        # Calculate batch sizes
        n_anchors = len(batch["anchor_samples"]["index"])

        indices = torch.cat(
            (
                batch["anchor_samples"]["index"],
                batch["positive_samples"]["index"],
                batch["negative_samples"]["index"],
            ),
            dim=0,
        ).cpu()

        indices = batch["anchor_samples"]["index"].cpu()

        geo_dist = None
        gdm = None  # self.dataset.dissimilarity_matrix_geodesic
        tdm = self.dataset.timestamp_dissimilarity_matrix
        if gdm is not None:
            # print("tdm",gdm.shape)
            from src.localization.evaluation.metrics import pairwise_mean_sq_diff_broadcast

            embedding_dist = torch.cdist(
                result["anchor"].embeddings_2d, result["anchor"].embeddings_2d
            )  # pairwise_mean_sq_diff_broadcast(result.embeddings_2d)

            geo_dist = gdm[indices][:, indices].cuda()
            time_dist = tdm[indices][:, indices].cuda()

            anchor_pos = torch.arange(n_anchors, device=embedding_dist.device)
            w = torch.zeros_like(embedding_dist, device=embedding_dist.device)
            w[anchor_pos, anchor_pos] = 1.0 / (time_dist[anchor_pos, anchor_pos] + 1.0)

            w2 = 1 / (geo_dist.abs() + 1.0)
            w2 = w2 * 0.0 + 1.0

            loss = torch.sum(w2 * torch.square(embedding_dist - geo_dist)) / torch.sum(
                w2
            )

        else:

           

            time_dist = torch.pairwise_distance(
                batch["anchor_samples"]["times"],
                batch["positive_samples"]["times"],
                p=1,
            )
            embedding_dist = torch.pairwise_distance(
                result["anchor"].embeddings_2d, result["positive"].embeddings_2d, p=2
            )

            k = torch.exp(self.alpha) ** 2 + self.beta
            w = 1 - 2 * time_dist / self.dataset.T
            
            mse = self.huber(embedding_dist, time_dist * k)
            loss = mse * w/w.mean()
            loss = torch.unsqueeze(loss, dim=1)


            self.log("alpha", self.alpha.detach().cpu(), prog_bar=False)
            self.log("mean_time_dist", time_dist.mean().detach().cpu(), prog_bar=False)
            self.log(
                "mean_embedding_dist",
                embedding_dist.mean().detach().cpu(),
                prog_bar=False,
            )
            self.log(
                "correlation",
                torch.corrcoef(
                    torch.stack([embedding_dist.flatten(), time_dist.flatten()])
                )[0, 1]
                .detach()
                .cpu(),
                prog_bar=False,
            )
            self.log(
                "division",
                (embedding_dist.mean() / time_dist.mean()).detach().cpu(),
                prog_bar=False,
            )
            self.log(
                "divison with exp",
                (embedding_dist.mean() / (time_dist.mean() * k)).detach().cpu(),
                prog_bar=False,
            )

            self.log("w_mean", w.mean().detach().cpu(), prog_bar=False)
            self.log("w_min", w.min().detach().cpu(), prog_bar=False)

            pass

        return loss

    def compute_time_prediction_loss(
        self, batch: dict, result: ModelResult
    ) -> torch.Tensor:
        """Compute time prediction loss"""

        times = batch["anchor_samples"]["times"]

        time_prediction = result["anchor"].time_prediction

        if isinstance(time_prediction, tuple):
            log_pi, mu, sigma = time_prediction
            loss = self.model.mdn.loss(log_pi, mu, sigma, times).mean()
        else:
            # loss = self.huber(time_prediction, times).mean()
            loss = (time_prediction - times).pow(2)

        return loss

    def compute_mixup_loss(self, batch: dict, result: ModelResult) -> torch.Tensor:
        def f(x0,x1,z0,z1):
            t = torch.rand(x0.shape[0], 1, device=x0.device)[:, 0]
            view_shape_x = [x0.shape[0]] + [1] * (x0.dim() - 1)
            t = t.view(*view_shape_x)
            
            x_t = (1 - t) * x0 + t * x1
            self.model.mask_ratio = 0.0
            z_t = self.forward(x_t)
            view_shape_z = [z0.shape[0]] + [1] * (z0.dim() - 1)
            t = t.view(*view_shape_z)
            z_t_target = (1 - t) * z0 + t * z1

            return z_t, z_t_target
        
        
        x0 = batch["anchor_samples"]["samples"]
        x1 = batch["positive_samples"]["samples"]
        x2 = batch["negative_samples"]["samples"]
        z0 = result["anchor"].embeddings_2d.detach()
        z1 = result["positive"].embeddings_2d.detach()
        z2 = result["negative"].embeddings_2d.detach()
        
        zh0 = result["anchor"].embeddings_highdim.detach()
        zh1 = result["positive"].embeddings_highdim.detach()
        zh2 = result["negative"].embeddings_highdim.detach()
        
        z_t , z_t_target = f(x0,x1,z0,z1)
        loss = torch.square(z_t.embeddings_2d - z_t_target).mean(dim=1,keepdim=True)
        

        
        if not True:
            z_t , z_t_target = f(x0,x1,zh0,zh1)
            z_t = F.normalize(z_t.embeddings_highdim, dim=1)
            z_t_target = F.normalize(z_t_target, dim=1) 
            loss = torch.square(z_t - z_t_target).mean(dim=1,keepdim=True)
            
        
        

        return loss

        pass

    
    def compute_diversity(self, batch: dict, result: ModelResult) -> torch.Tensor:
        """Compute diversity loss to encourage variety in embeddings"""

        emb = result["anchor"].embeddings_2d
        emb = emb - emb.mean(dim=0)

        loss_diversity = -emb.std(dim=0).mean()
     

        return loss_diversity

    def _training_step(self, batch: dict, batch_idx: int) -> dict:
        """Training step"""
        # Update schedulers
        if self.loss_coeff_scheduler:
            coeff_scheduler = self.loss_coeff_scheduler.get_margin()

            self.loss_coeff_scheduler.step()

        loss_dict = {}

        # 1. Triplet Loss
        triplet_loss, result = self.compute_triplet_loss(batch)
        diversity = self.compute_diversity(batch, result) * (1-coeff_scheduler)

        if self.config.coeff_triplet_loss > 0:
 
            loss_dict["triplet_loss"] = triplet_loss * coeff_scheduler

        # 2. Dissimilarity Loss
        if self.config.coeff_dissimilarity_loss > 0:
            dissim_loss = self.compute_dissimilarity_loss(batch, result)
            loss_dict["dissimilarity_loss"] = dissim_loss

        # 3. Reconstruction Loss
        if self.config.coeff_mse_loss > 0:
            recon_loss, recon_result = self.compute_reconstruction_loss(batch, result)
            loss_dict["mse_loss"] = recon_loss *coeff_scheduler

            
        return loss_dict

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:

        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        loss_dict = self._training_step(batch, batch_idx)
        for key, value in loss_dict.items():
            #mean over feature dimension [B, D] -> [B]
            value = value.mean(dim=1)
            loss_dict[key] = value
            #print(key, value.shape)
            
        
        #add over zero dimension to make it (num of losses ,batch_size) = (L, B)
        losses = torch.stack(list(loss_dict.values()),dim=0)
        # mean over batch since it is not fit into vram
        losses = losses.mean(dim=1)

        backward(losses, aggregator=self.aggregator)


        total_norm = utils.clip_grad_norm_(self.model.parameters(), max_norm=0.9)
        self.log("grad_norm", total_norm.max(), prog_bar=False)
        
        #if self.global_step % 10 == 0:
        optimizer.step()
        optimizer.zero_grad()
        
        scheduler.step()
        self.log("learning_rate", optimizer.param_groups[0]["lr"], prog_bar=False)
        
        for key, value in zip(loss_dict.keys(), losses):
            self.log(f"loss/{key}", value.detach().cpu(), prog_bar=False)
            
        return None

    def validation_step(self, batch: dict, batch_idx: int):
        """Validation step"""

        anchor = batch["anchor_samples"]["samples"]
        times = batch["anchor_samples"]["times"]
        indices = batch["anchor_samples"]["index"]
        gt = batch["anchor_samples"]["reference"]

        # Forward pass
        with torch.no_grad():
            result = self.forward(anchor, times, indices)

            self.val_embeddings_2d.append(result.embeddings_2d.cpu())
            self.ground_truth.append(gt.cpu())
            self.time_truth.append(times.cpu())

        # Time prediction if enabled
        if self.config.coeff_time_loss > 0 and result.time_prediction is not None:
            if isinstance(result.time_prediction, tuple):
                log, mu, sigma = result.time_prediction
                tp = self.model.mdn.sample(log, mu, sigma)
            else:
                tp = result.time_prediction
            self.val_time_prediction.append(tp.cpu())
            

    def on_validation_epoch_start(self):
        """Reset validation buffers at epoch start"""
        self.map_embeddings = torch.zeros([len(self.dataset), 2])

        self.reset_validation_buffers()

    def on_validation_epoch_end(self):
        """Process validation results at epoch end"""

        self.val_embeddings_2d = torch.cat(self.val_embeddings_2d, dim=0)

        self._compute_max_embeddings(self.val_embeddings_2d)

        self.val_embeddings_2d = self.val_embeddings_2d.numpy()
        self.ground_truth = torch.cat(self.ground_truth, dim=0).numpy()
        time_truth = torch.cat(self.time_truth, dim=0).numpy()

        # Time prediction plots
        if len(self.val_time_prediction) > 0:
            time_pred = torch.cat(self.val_time_prediction, dim=0).numpy()

            fig = plot_2d_scatter_time_color(
                self.ground_truth[:, 0], self.ground_truth[:, 1], time_pred, show=False
            )
            self.logger.experiment.add_figure(
                "Time/prediction", fig, global_step=self.global_step
            )
            fig.clear()

            fig = plot_2d_scatter_time_color(
                self.ground_truth[:, 0], self.ground_truth[:, 1], time_truth, show=False
            )
            self.logger.experiment.add_figure(
                "Time/true", fig, global_step=self.global_step
            )
            fig.clear()

        # Plot and log metrics
        self.plot_validation_results(self.val_embeddings_2d, self.ground_truth, "")

        if self.Y_i != None:
            fig = plot_colorized(self.Y_i.cpu(), self.Y_i.cpu(), "drawsamples")
            self.logger.experiment.add_figure(
                "drawsamples", fig, global_step=self.global_step
            )
            fig.clear()

        plt.close("all")
        
        if not True:    
            self.dataset.T =  self.ds_scheduler.get_margin()
            self.dataset._make_triplets()
            self.log("T", self.dataset.T, prog_bar=False)
            self.ds_scheduler.step()

    def plot_validation_results(
        self, embeddings: np.ndarray, ground_truth: np.ndarray, section: str
    ):
        """Plot validation results"""

        # Original embeddings
        fig = plot_colorized(embeddings, ground_truth, title="embeddings", show=False)
        self.logger.experiment.add_figure(
            f"{section}Plot/Embeddings", fig, global_step=self.global_step
        )
        fig.clear()

        # Transformed embeddings
        embeddings_transformed = affine_transform_channel_chart(
            embeddings, ground_truth
        )

        fig = plot_colorized(
            embeddings_transformed, ground_truth, title="Transformed", show=False
        )
        self.logger.experiment.add_figure(
            f"{section}Plot/Transformed", fig, global_step=self.global_step
        )
        fig.clear()

        # Reference
        fig = plot_colorized(ground_truth, ground_truth, title="Reference", show=False)
        self.logger.experiment.add_figure(
            f"{section}Plot/Reference", fig, global_step=self.global_step
        )
        fig.clear()

        # Error plots
        fig = plot_error_cdf(embeddings_transformed, ground_truth)
        self.logger.experiment.add_figure(
            f"{section}Error/Error_CDF", fig, global_step=self.global_step
        )
        fig.clear()

        fig, errors = plot_error_vectors(embeddings_transformed, ground_truth)
        self.logger.experiment.add_figure(
            f"{section}Error/Error_Vectors", fig, global_step=self.global_step
        )
        fig.clear()

        for k, v in errors.items():
            self.log(f"{section}Error/{k}", v, prog_bar=True)
            print(f"{section}Error/{k}", v)

        self.log_validation_metrics(embeddings_transformed, ground_truth, section)

    def log_validation_metrics(
        self, embeddings: np.ndarray, ground_truth: np.ndarray, section: str
    ):
        """Log validation metrics"""
        metrics = calculate_metrics(embeddings, ground_truth)
        for k, v in metrics.items():
            self.log(f"{section}Error/{k}", v, prog_bar=False)

    def setup(self, stage: str):
        """Setup datasets"""
        if stage == "fit":
            self.train_dataset = self.dataset
            self.setup_epoch_indices()

    def setup_epoch_indices(self):

        pass

    def on_train_epoch_start(self):
        """Regenerate indices at epoch start"""
        

        # self.setup_epoch_indices()

        pass

    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader"""

        """            _sampler=RandomSampler(
                self.train_dataset, 
                num_samples=self.config.val_samples
            )
            _shuffle = False"""

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self.dataset.collate_fn,
            #drop_last=True,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.config.prefetch_factor,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self.dataset.collate_fn,
        )

    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        # from torch.optim.lr_scheduler import OneCycleLR,ReduceLROnPlateau

        """for n,v in self.model.named_parameters():
            print(n)"""

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.learning_rate, weight_decay=0.0,fused=True
        )
        # even creating this scheduler changes the learning rate
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters= self.config.max_epochs * len(self.train_dataloader()), 
            

            
           
        )
        return [optimizer], [scheduler]

    def _draw_samples(self, n=None):

        self.Y_i = draw_samples(
            self.config.path_map, self.config.map_matching_size, torch.FloatTensor
        )
        self.C2 = ot.dist(self.Y_i, p=self.config.p, metric=self.config.metric)

    def _clone_samples(self):
        self.C2 = self.C2.clone().cuda()
        self.Y_i = self.Y_i.clone().cuda()

    def _compute_max_embeddings(self, e):

        self.min = e.min(dim=0, keepdim=True)[0].detach().cuda()
        self.max = e.max(dim=0, keepdim=True)[0].detach().cuda()
        return self.max, self.min

    def _normilze_embeddings(self, e):

        return (e - self.min) / (self.max - self.min)

    def _diff(self, x1, x2):
        diff = x1 - x2  # shape: (N, D)
        diff = diff**2  # square element-wise
        diff = torch.mean(diff, dim=1, keepdim=True)  # sum over D

        return diff


def run(config: TrainingConfig, ckpt_path=None):
    # from pytorch_lightning.callbacks import ModelCheckpoint
    from lightning.pytorch.callbacks import LearningRateMonitor
    torch.backends.cudnn.benchmark = True

    from pytorch_lightning import seed_everything
    pc = PathConfig()

    if config.deterministic:
        seed_everything(42, workers=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="Error/Euclidean",
        mode="min",
        save_top_k=1,
        filename="epoch_{epoch:02d}",
        save_last=True,
        save_on_train_epoch_end=True,
        every_n_epochs=config.val_check_interval,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    rich_progress_bar = RichProgressBar(leave=True)
    """Run training"""
    trainer = Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu",
        check_val_every_n_epoch=config.val_check_interval,
        log_every_n_steps=5,
        callbacks=[lr_monitor],
        num_sanity_val_steps=0,
        deterministic=config.deterministic,
        default_root_dir = pc.root_dir,
        enable_progress_bar=True,


    )
    # Create model
    module = TripletLearningModule(config)
    # Setup
    module.setup("fit")
    module.setup("validate")
    #not supported
    #module.model = torch.compile(module.model)
    
    if ckpt_path is not None:
        pass
        print("validating")
        x = torch.rand([8, 1, 14, 90])
        # x = torch.rand([8, 6, 49])

        t = torch.rand([8, 1])
        module.forward(x, t)

        trainer.validate(module, module.val_dataloader(), ckpt_path=ckpt_path)
    else:
        trainer.fit(module, ckpt_path=ckpt_path)
        
    torch.cuda.empty_cache()   
    


def run_predict(config: TrainingConfig, ckpt_path):
    from pytorch_lightning import seed_everything
    import torch
    pc = PathConfig()

    if config.deterministic:
        seed_everything(42, workers=True)

    trainer = Trainer(
        accelerator="gpu",
        log_every_n_steps=5,
        num_sanity_val_steps=0,
        deterministic=config.deterministic,
        default_root_dir = pc.root_dir,

    )
    module = TripletLearningModule(config)
    module.setup("fit")
    module.setup("validate")

    print("validating")
    dl = module.val_dataloader()
    batch = next(iter(dl))
    x = batch["anchor_samples"]["samples"]
    
    #t = torch.rand([8, 1])
    module.forward(x)

    trainer.validate(module, module.val_dataloader(), ckpt_path=ckpt_path)
    return module.val_embeddings_2d, module.ground_truth


def main():
    from src.localization.config.presets import config_5g, config_dichasus


    # config_dichasus , config_5g , config_jonas , config_max , config_passive
    from src.localization.config.training import TrainingConfig
    from pathlib import Path

    for k in [4]:
        config_5g.max_epochs = 100
        config_5g.K = k
        run(config_5g)


if __name__ == "__main__":
    main()
