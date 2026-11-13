"""Self-contained MIMO U-Net for HybridCT.

Replaces the external ``mimo-unet-forked`` dependency
(``mimo.models.mimo_unet``, ``mimo.models.utils``, ``mimo.models.ensemble``,
``mimo.losses``). The network is built from the *primary-study* U-Net in
``emulator/Model.py``:

* identical conv block (``UNet._block``: 3x3 conv, no bias, zero padding,
  BatchNorm, ReLU, twice),
* identical down-sampling (``MaxPool2d(2)``),
* identical up-sampling (``ConvTranspose2d(kernel_size=2, stride=2)``), i.e.
  the bilinear up-sampling of the external library is gone,
* identical 4-level depth and skip connections, 1x1 output conv and Xavier init.

MIMO topology (Havasi et al. 2021; Baumann et al. 2024 MIMO-UNet):

    per-subnetwork  : encoder1_i (C_in -> F), pool, encoder2_i (F -> 2F)
    shared core     : cat_i(enc2_i) -> encoder3 .. bottleneck .. decoder2
                      (width multiplier m on F. Default m = 1 keeps the
                      primary U-Net width: ~34 M parameters for S = 5, F = 64,
                      vs 31 M for the primary U-Net. m = S mimics the old
                      MIMO-UNet core scaling but gives ~768 M parameters.)
    per-subnetwork  : upconv1_i, cat(enc1_i), decoder1_i, 1x1 conv -> 2*C_out
                      (location and log-scale)

Shapes: input [B, S, C_in, H, W] -> (p1, p2) each [B, S, C_out, H, W].
H and W must be divisible by 16 (four pooling levels), as for the primary U-Net.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # package-style import (pip install -e . from the repo root)
    from emulator.Model import UNet as PrimaryUNet
except ImportError:  # flat import when run from emulator/mimo/pyct
    from Model import UNet as PrimaryUNet  # identical copy of emulator/Model.py

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl


# --------------------------------------------------------------------------- #
# Losses (same semantics as mimo.losses, so old configs / numbers are comparable)
# --------------------------------------------------------------------------- #
class UncertaintyLoss(nn.Module, ABC):
    """Heteroscedastic NLL. ``p2`` is the log of the distribution parameter."""

    name: str = ""
    eps_min: float = 1e-5
    eps_max: float = 1e3

    @abstractmethod
    def forward(self, mu, log_param, y, *, mask=None, reduce_mean=True): ...

    @abstractmethod
    def scale(self, log_param) -> torch.Tensor:
        """Native distribution parameter (Laplace b, Gaussian sigma)."""

    @abstractmethod
    def std(self, mu, log_param) -> torch.Tensor: ...

    def mode(self, mu, log_param):
        return mu

    @abstractmethod
    def calculate_dist_param(self, std, *, log=False): ...

    @classmethod
    def from_name(cls, name: str) -> "UncertaintyLoss":
        name = name.lower()
        if name in ("laplace_nll", "laplace"):
            return LaplaceNLL()
        if name in ("gaussian_nll", "gaussian", "normal"):
            return GaussianNLL()
        raise ValueError(f"Unknown loss '{name}' (use 'laplace_nll' or 'gaussian_nll').")


class LaplaceNLL(UncertaintyLoss):
    """NLL of Laplace(mu, b) with p2 = log b. std = sqrt(2) b."""

    name = "laplace_nll"

    def forward(self, mu, log_scale, y, *, mask=None, reduce_mean=True):
        scale = torch.exp(log_scale).clone()
        with torch.no_grad():  # value clamp, identity gradient (as in mimo.losses)
            scale.clamp_(min=self.eps_min, max=self.eps_max)
        loss = torch.log(scale) + (mu - y).abs() / scale
        if mask is not None:
            loss = loss * mask
        return loss.mean() if reduce_mean else loss

    def scale(self, log_scale):
        # same value clamp as in forward(): the identity-gradient clamp lets log_scale drift
        # outside the bounds in training, so unclamped exp() can overflow at inference.
        return torch.exp(log_scale).clamp(min=self.eps_min, max=self.eps_max)

    def std(self, mu, log_scale):
        return self.scale(log_scale) * math.sqrt(2.0)

    def calculate_dist_param(self, std, *, log=False):
        param = (std / math.sqrt(2.0)).clamp(min=self.eps_min, max=self.eps_max)
        return torch.log(param) if log else param


class GaussianNLL(UncertaintyLoss):
    """NLL of N(mu, sigma^2) with p2 = log sigma^2 (constant terms dropped)."""

    name = "gaussian_nll"

    def forward(self, mu, log_var, y, *, mask=None, reduce_mean=True):
        var = torch.exp(log_var).clone()
        with torch.no_grad():
            var.clamp_(min=self.eps_min, max=self.eps_max)
        loss = 0.5 * (torch.log(var) + (mu - y) ** 2 / var)
        if mask is not None:
            loss = loss * mask
        return loss.mean() if reduce_mean else loss

    def scale(self, log_var):
        return torch.exp(log_var).clamp(min=self.eps_min, max=self.eps_max).sqrt()

    def std(self, mu, log_var):
        return self.scale(log_var)

    def calculate_dist_param(self, std, *, log=False):
        param = (std ** 2).clamp(min=self.eps_min, max=self.eps_max)
        return torch.log(param) if log else param


# --------------------------------------------------------------------------- #
# Loss buffer (subnetwork synchronisation) and MIMO batch utilities
# --------------------------------------------------------------------------- #
class LossBuffer:
    """Running buffer of per-subnetwork losses -> softmax(mean / T) * S weights."""

    def __init__(self, subnetworks: int, temperature: float, buffer_size: int) -> None:
        self.index = 0
        self.temperature = temperature
        self.buffer_size = int(buffer_size)
        self.subnetworks = subnetworks
        self.buffer = torch.zeros(max(self.buffer_size, 1), subnetworks)

    def add(self, loss: torch.Tensor) -> None:
        if self.buffer_size > 0:
            self.buffer[self.index] = loss.detach().float().cpu()
            self.index = (self.index + 1) % self.buffer_size

    def get_weights(self) -> torch.Tensor:
        if self.buffer_size == 0 or self.temperature <= 0:
            return torch.ones(self.subnetworks)
        mean = self.buffer.mean(dim=0)
        return F.softmax(mean / self.temperature, dim=-1) * self.subnetworks


def repeat_subnetworks(x: torch.Tensor, num_subnetworks: int) -> torch.Tensor:
    """[B, C, H, W] -> [B, S, C, H, W] (inference: every member sees the same input)."""
    return x[:, None].repeat(1, num_subnetworks, 1, 1, 1)


def flatten_subnetwork_dimension(x: torch.Tensor) -> torch.Tensor:
    B, S, C, H, W = x.shape
    return x.reshape(B * S, C, H, W)


def apply_input_transform(image, label, mask, num_subnetworks: int,
                          input_repetition_probability: float = 0.0,
                          batch_repetitions: int = 1):
    """MIMO training transform (batch repetition + partial input repetition).

    A fraction ``1 - input_repetition_probability`` of the (repeated) batch is
    shuffled independently per subnetwork; the rest is shared by all members.
    Returns tensors of shape [B * batch_repetitions, S, C, H, W].
    """
    B = image.shape[0]
    main = torch.randperm(B, device=image.device).repeat(batch_repetitions)
    n_shuffle = int(main.shape[0] * (1.0 - input_repetition_probability))
    idx = [torch.cat((main[:n_shuffle][torch.randperm(n_shuffle, device=image.device)],
                      main[n_shuffle:]), dim=0) for _ in range(num_subnetworks)]

    def _stack(t):
        return None if t is None else torch.stack([t.index_select(0, i) for i in idx], dim=1)

    return _stack(image), _stack(label), _stack(mask)


def compute_uncertainties(criterion: UncertaintyLoss, y_preds: torch.Tensor, log_params: torch.Tensor):
    """Mean prediction and variance decomposition over the S members.

    Returns (mean, aleatoric_variance, epistemic_variance), each [B, C, H, W].
    aleatoric = mean_s std_s^2 (= mean_s 2 b_s^2 for Laplace, i.e. a true variance);
    epistemic = unbiased variance of member locations (denominator S - 1).
    """
    S = y_preds.shape[1]
    mean = criterion.mode(y_preds, log_params).mean(dim=1)
    aleatoric = torch.square(criterion.std(y_preds, log_params)).mean(dim=1)
    if S > 1:
        epistemic = torch.square(y_preds - y_preds.mean(dim=1, keepdim=True)).sum(dim=1) / (S - 1)
    else:
        epistemic = torch.zeros_like(aleatoric)
    return mean, aleatoric, epistemic


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
def _block(cin: int, cout: int, dropout: float = 0.0) -> nn.Module:
    """Primary-study conv block, with optional spatial dropout appended."""
    blk = PrimaryUNet._block(cin, cout)
    return nn.Sequential(blk, nn.Dropout2d(dropout)) if dropout > 0 else blk


def _upconv(cin: int, cout: int) -> nn.ConvTranspose2d:
    """Primary-study up-sampling: learned 2x2 transposed conv, stride 2."""
    return nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2)


class MimoUNet(nn.Module):
    """MIMO U-Net assembled from the primary HybridCT U-Net components."""

    def __init__(self, in_channels: int, out_channels: int, num_subnetworks: int,
                 filter_base_count: int = 64, core_width_multiplier: Optional[int] = None,
                 center_dropout_rate: float = 0.0, final_dropout_rate: float = 0.0,
                 encoder_dropout_rate: float = 0.0, core_dropout_rate: float = 0.0,
                 decoder_dropout_rate: float = 0.0) -> None:
        super().__init__()
        S, f = num_subnetworks, filter_base_count
        m = 1 if core_width_multiplier is None else int(core_width_multiplier)
        fc = f * m  # shared-core base width
        self.num_subnetworks = S
        self.in_channels = in_channels
        self.out_channels = out_channels

        # per-subnetwork encoder (levels 1-2)
        self.encoder1 = nn.ModuleList([_block(in_channels, f, encoder_dropout_rate) for _ in range(S)])
        self.encoder2 = nn.ModuleList([_block(f, 2 * f, encoder_dropout_rate) for _ in range(S)])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # shared core (levels 3-5 and decoder levels 4-2)
        self.encoder3 = _block(2 * f * S, 4 * fc, core_dropout_rate)
        self.encoder4 = _block(4 * fc, 8 * fc, core_dropout_rate)
        self.bottleneck = _block(8 * fc, 16 * fc, core_dropout_rate)
        self.center_dropout = nn.Dropout(center_dropout_rate)
        self.upconv4 = _upconv(16 * fc, 8 * fc)
        self.decoder4 = _block(16 * fc, 8 * fc, core_dropout_rate)
        self.upconv3 = _upconv(8 * fc, 4 * fc)
        self.decoder3 = _block(8 * fc, 4 * fc, core_dropout_rate)
        self.upconv2 = _upconv(4 * fc, 2 * fc)
        self.decoder2 = _block(2 * fc + 2 * f * S, 2 * fc, core_dropout_rate)

        # per-subnetwork decoder (level 1) and heads
        self.upconv1 = nn.ModuleList([_upconv(2 * fc, f) for _ in range(S)])
        self.decoder1 = nn.ModuleList([_block(2 * f, f, decoder_dropout_rate) for _ in range(S)])
        self.final_dropout = nn.ModuleList([nn.Dropout(final_dropout_rate) for _ in range(S)])
        self.conv = nn.ModuleList([nn.Conv2d(f, out_channels, kernel_size=1) for _ in range(S)])

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(mod: nn.Module) -> None:
        # same initialisation as emulator/fit_model.py (Xavier on Conv2d / Linear)
        if type(mod) in (nn.Conv2d, nn.Linear):
            nn.init.xavier_uniform_(mod.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C, H, W = x.shape
        if S != self.num_subnetworks:
            raise ValueError(f"subnetwork dim {S} != num_subnetworks {self.num_subnetworks}")
        if H % 16 or W % 16:
            raise ValueError(f"tile size ({H}x{W}) must be divisible by 16")
        enc1 = [self.encoder1[i](x[:, i]) for i in range(S)]
        enc2 = [self.encoder2[i](self.pool(enc1[i])) for i in range(S)]
        x2 = torch.cat(enc2, dim=1)

        enc3 = self.encoder3(self.pool(x2))
        enc4 = self.encoder4(self.pool(enc3))
        bott = self.center_dropout(self.bottleneck(self.pool(enc4)))
        d4 = self.decoder4(torch.cat((self.upconv4(bott), enc4), dim=1))
        d3 = self.decoder3(torch.cat((self.upconv3(d4), enc3), dim=1))
        d2 = self.decoder2(torch.cat((self.upconv2(d3), x2), dim=1))

        outs = []
        for i in range(S):
            d1 = self.decoder1[i](torch.cat((self.upconv1[i](d2), enc1[i]), dim=1))
            outs.append(self.conv[i](self.final_dropout[i](d1)))
        return torch.stack(outs, dim=1)  # [B, S, out_channels, H, W]


# --------------------------------------------------------------------------- #
# Lightning module (drop-in for mimo.models.mimo_unet.MimoUnetModel)
# --------------------------------------------------------------------------- #
def _regression_metrics(y_hat: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
    y_hat, y = y_hat.detach().flatten().float(), y.detach().flatten().float()
    err = y_hat - y
    mse = torch.mean(err ** 2)
    ss_tot = torch.sum((y - y.mean()) ** 2).clamp_min(1e-12)
    return {"mae": err.abs().mean(), "mse": mse, "rmse": torch.sqrt(mse),
            "r2": 1.0 - torch.sum(err ** 2) / ss_tot}


class MimoUnetModel(pl.LightningModule):
    """MIMO U-Net with the same constructor arguments as the old external model.

    New optional argument ``core_width_multiplier`` (default None -> 1, the
    primary U-Net width; see module docstring).
    """

    def __init__(self, in_channels: int, out_channels: int, num_subnetworks: int,
                 filter_base_count: int, center_dropout_rate: float = 0.0,
                 final_dropout_rate: float = 0.0, encoder_dropout_rate: float = 0.0,
                 core_dropout_rate: float = 0.0, decoder_dropout_rate: float = 0.0,
                 loss: str = "laplace_nll", weight_decay: float = 0.0,
                 learning_rate: float = 1e-4, seed: int = 1, loss_buffer_size: int = 10,
                 loss_buffer_temperature: float = 0.3, input_repetition_probability: float = 0.0,
                 batch_repetitions: int = 1, scheduler_step_size: int = 20,
                 scheduler_gamma: float = 0.5, core_width_multiplier: Optional[int] = None,
                 cloud_threshold_scaled: Optional[Sequence[float]] = None):
        super().__init__()
        if out_channels % 2:
            raise ValueError("out_channels must be 2 * C_out (location and log-scale)")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subnetworks = num_subnetworks
        self.loss_fn = UncertaintyLoss.from_name(loss)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.input_repetition_probability = input_repetition_probability
        self.batch_repetitions = batch_repetitions
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.model = MimoUNet(in_channels, out_channels, num_subnetworks, filter_base_count,
                              core_width_multiplier, center_dropout_rate, final_dropout_rate,
                              encoder_dropout_rate, core_dropout_rate, decoder_dropout_rate)
        self.loss_buffer = LossBuffer(num_subnetworks, loss_buffer_temperature, loss_buffer_size)
        # cloudy-pixel threshold (scaled target units, per output channel) for selection metrics
        thr = cloud_threshold_scaled if cloud_threshold_scaled is not None else [float("-inf")]
        self.register_buffer("_cloud_thr", torch.tensor(list(thr), dtype=torch.float32).view(1, -1, 1, 1),
                             persistent=False)
        self._val_acc = None
        self.save_hyperparameters()
        self.save_hyperparameters({"architecture": "hybridct_mimo_unet_convtranspose_v1",
                                   "trainable_params": count_trainable_parameters(self.model)})

    # ---- core API -------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [B, S, C_in, H, W] -> (loc, log_param), each [B, S, C_out, H, W]."""
        out = self.model(x)
        half = self.out_channels // 2
        return out[:, :, :half], out[:, :, half:]

    @torch.no_grad()
    def predict_members(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [B, C_in, H, W] -> per-member (loc, native scale), [B, S, C_out, H, W]."""
        loc, logp = self(repeat_subnetworks(x, self.num_subnetworks))
        return loc, self.loss_fn.scale(logp)

    def _bs(self) -> Optional[int]:
        dm = getattr(self.trainer, "datamodule", None) if self._trainer is not None else None
        return getattr(dm, "batch_size", None)

    # ---- training -------------------------------------------------------- #
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        image, label, mask = batch["image"], batch["label"], batch.get("mask")
        image_t, label_t, mask_t = apply_input_transform(
            image, label, mask, self.num_subnetworks,
            self.input_repetition_probability, self.batch_repetitions)
        p1, p2 = self(image_t)
        per_member = self.loss_fn(p1, p2, label_t, mask=mask_t, reduce_mean=False).mean(dim=(0, 2, 3, 4))
        weights = self.loss_buffer.get_weights().to(per_member.device)
        self.loss_buffer.add(per_member)
        loss = (per_member * weights).mean()
        bs = self._bs()
        self.log("train_loss", per_member.mean(), batch_size=bs)
        for i in range(self.num_subnetworks):
            self.log(f"train_loss_{i}", per_member[i], batch_size=bs)
            self.log(f"train_weight_{i}", weights[i], batch_size=bs)
        for k, v in _regression_metrics(p1, label_t).items():
            self.log(f"metric_train/{k}", v, on_step=True, on_epoch=True, batch_size=bs)
        return loss

    # exact-density constants (dropped in the training losses; equal for all members)
    def _nll_const(self) -> float:
        return math.log(2.0) if self.loss_fn.name.startswith("laplace") else 0.5 * math.log(2 * math.pi)

    def mixture_nll(self, p1: torch.Tensor, p2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Exact per-pixel NLL of the equal-weight S-member mixture, [B, C, H, W] (scaled units)."""
        nll_s = self.loss_fn(p1, p2, repeat_subnetworks(y, p1.shape[1]), reduce_mean=False) + self._nll_const()
        return -(torch.logsumexp(-nll_s, dim=1) - math.log(p1.shape[1]))

    def on_validation_epoch_start(self) -> None:
        self._val_acc = {k: [torch.zeros((), dtype=torch.float64, device=self.device),
                             torch.zeros((), dtype=torch.float64, device=self.device)]
                         for k in ("nll_mix", "nll_mix_cloudy", "se_cloudy")}

    def _acc(self, key: str, values: torch.Tensor, sel: Optional[torch.Tensor] = None) -> None:
        v = values.double()
        if sel is not None:
            v = v[sel]
        self._val_acc[key][0] += v.sum()
        self._val_acc[key][1] += v.numel()

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        image, label, mask = batch["image"], batch["label"], batch.get("mask")
        image_r = repeat_subnetworks(image, self.num_subnetworks)
        label_r = repeat_subnetworks(label, self.num_subnetworks)
        p1, p2 = self(image_r)
        per_member = self.loss_fn(p1, p2, label_r, reduce_mean=False).mean(dim=(0, 2, 3, 4))
        mean, ale, epi = compute_uncertainties(self.loss_fn, p1, p2)
        comb_logp = self.loss_fn.calculate_dist_param(torch.sqrt(ale + epi), log=True)
        val_comb = self.loss_fn(mean, comb_logp, label, mask=mask, reduce_mean=True)
        bs = self._bs()
        self.log("val_loss", per_member.mean(), batch_size=bs, sync_dist=True)
        self.log("val_loss_combined", val_comb, batch_size=bs, sync_dist=True)
        for i in range(self.num_subnetworks):
            self.log(f"val_loss_{i}", per_member[i], batch_size=bs)
        for k, v in _regression_metrics(mean, label).items():
            self.log(f"metric_val/{k}", v, on_epoch=True, batch_size=bs)
        self.log("metric_val/aleatoric_std_mean", ale.sqrt().mean(), batch_size=bs)
        self.log("metric_val/epistemic_std_mean", epi.sqrt().mean(), batch_size=bs)
        # pixel-exact epoch accumulators (selection metrics; independent of batch sizes)
        if self._val_acc is None:
            self.on_validation_epoch_start()
        nll_mix = self.mixture_nll(p1.float(), p2.float(), label.float())
        cloudy = label >= self._cloud_thr.to(label.dtype)
        self._acc("nll_mix", nll_mix)
        self._acc("nll_mix_cloudy", nll_mix, cloudy)
        self._acc("se_cloudy", (mean - label) ** 2, cloudy)
        return per_member.mean()

    def on_validation_epoch_end(self) -> None:
        if not self._val_acc:
            return
        acc = self._val_acc
        if self.trainer is not None and self.trainer.world_size > 1:
            for k in acc:
                acc[k] = [self.all_gather(v).sum() for v in acc[k]]
        def ratio(k):
            s, n = acc[k]
            return (s / n).float() if n > 0 else torch.tensor(float("nan"), device=self.device)
        self.log("val_nll_mixture", ratio("nll_mix"))
        self.log("val_nll_mixture_cloudy", ratio("nll_mix_cloudy"))
        self.log("metric_val/mse_cloudy", ratio("se_cloudy"))
        self.log("metric_val/cloudy_fraction", (acc["nll_mix_cloudy"][1] / acc["nll_mix"][1].clamp_min(1)).float())
        self._val_acc = None

    def configure_optimizers(self) -> Dict[str, Any]:
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=self.scheduler_step_size,
                                                gamma=self.scheduler_gamma)
        return {"optimizer": opt, "lr_scheduler": sched, "monitor": "val_loss"}

    @classmethod
    def from_config(cls, config: Dict[str, Any], in_channels: int, out_channels: int) -> "MimoUnetModel":
        """Build from a HybridCT YAML config (``mimo_*`` keys)."""
        g = config.get
        return cls(
            in_channels=in_channels, out_channels=2 * out_channels,
            num_subnetworks=config["mimo_num_subnetworks"],
            filter_base_count=config["mimo_filter_base_count"],
            center_dropout_rate=g("mimo_center_dropout_rate", 0.0),
            final_dropout_rate=g("mimo_final_dropout_rate", 0.0),
            encoder_dropout_rate=g("mimo_encoder_dropout_rate", 0.0),
            core_dropout_rate=g("mimo_core_dropout_rate", 0.0),
            decoder_dropout_rate=g("mimo_decoder_dropout_rate", 0.0),
            loss=g("mimo_loss", "laplace_nll"), weight_decay=g("weight_decay", 0.0),
            learning_rate=g("learning_rate", 1e-4), seed=g("rng_seed", 1),
            loss_buffer_size=g("mimo_loss_buffer_size", 10),
            loss_buffer_temperature=g("mimo_loss_buffer_temperature", 0.3),
            input_repetition_probability=g("mimo_input_repetition_probability", 0.0),
            batch_repetitions=g("mimo_batch_repetitions", 1),
            scheduler_step_size=g("scheduler_step_size", 20),
            scheduler_gamma=g("scheduler_gamma", 0.5),
            core_width_multiplier=g("mimo_core_width_multiplier", None),
            cloud_threshold_scaled=g("_cloud_threshold_scaled", None))


def load_mimo_checkpoint(path: str, map_location="cpu") -> MimoUnetModel:
    """Load a checkpoint written by this module, with a clear error for old ones."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    if hp.get("architecture") != "hybridct_mimo_unet_convtranspose_v1":
        raise RuntimeError(
            f"{path} was not written by emulator/mimo/pyct/mimo_unet.py (old bilinear "
            "MIMO-UNet checkpoints are not weight-compatible). Retrain with train_cloud.py.")
    model = MimoUnetModel.load_from_checkpoint(path, map_location=map_location)
    model.eval()
    return model


class EnsembleModule(nn.Module):
    """Deep ensemble of MIMO checkpoints; members are concatenated on dim 1."""

    def __init__(self, checkpoint_paths: Sequence[str], map_location="cpu") -> None:
        super().__init__()
        self.models = nn.ModuleList([load_mimo_checkpoint(p, map_location) for p in checkpoint_paths])
        self.loss_fn = self.models[0].loss_fn
        self.num_subnetworks = sum(m.num_subnetworks for m in self.models)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [B, C, H, W] -> (loc, log_param) over all members, [B, S_total, C_out, H, W]."""
        outs = [m(repeat_subnetworks(x, m.num_subnetworks)) for m in self.models]
        return torch.cat([o[0] for o in outs], dim=1), torch.cat([o[1] for o in outs], dim=1)

    @torch.no_grad()
    def predict_members(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        loc, logp = self(x)
        return loc, self.loss_fn.scale(logp)
