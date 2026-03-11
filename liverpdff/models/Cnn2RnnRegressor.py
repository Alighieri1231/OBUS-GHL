"""
Cnn2RnnRegressor.py

PDFF regression model with modular CNN and temporal aggregation blocks.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torchmetrics.functional import concordance_corrcoef as ccc_func
from torchmetrics.functional import pearson_corrcoef

try:
    from lightning import LightningModule
except ImportError:
    from pytorch_lightning import LightningModule


def stratified_metrics_by_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: list[float],
    name: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_true >= lo) & (y_true < hi)
        if hi == bins[-1]:
            mask = (y_true >= lo) & (y_true <= hi)
        yt = y_true[mask]
        yp = y_pred[mask]
        if yt.size == 0:
            rows.append(
                {
                    "split": name,
                    "bin_lo": float(lo),
                    "bin_hi": float(hi),
                    "n": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "bias": np.nan,
                }
            )
            continue
        diff = yp - yt
        rows.append(
            {
                "split": name,
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "n": int(yt.size),
                "mae": float(np.mean(np.abs(diff))),
                "rmse": float(np.sqrt(np.mean(diff**2))),
                "bias": float(np.mean(diff)),
            }
        )
    return pd.DataFrame(rows)


class Cnn2RnnRegressor(LightningModule):
    def __init__(
        self,
        cnn: nn.Module | None = None,
        rnn: nn.Module | None = None,
        regressor: nn.Module | None = None,
        loss: nn.Module = nn.L1Loss(reduction="mean"),
        lr: float | None = 5e-5,
        clamp_min: float = 0.0,
        clamp_max: float = 100.0,
        target_transform: str = "none",
        report_intermediates: bool = False,
        report_dir: str | None = None,
        save_epoch_reports: bool = True,
        report_every_n_epochs: int = 25,
        pdff_bins: list[float] | None = None,
    ) -> None:
        super().__init__()
        if cnn is None:
            from ghlobus.models.TvCnn import TvCnn

            cnn = TvCnn()
        if rnn is None:
            from ghlobus.models.BasicAdditiveAttention import BasicAdditiveAttention

            rnn = BasicAdditiveAttention()
        if regressor is None:
            regressor = nn.Linear(in_features=1000, out_features=1)
        self.cnn = cnn
        self.rnn = rnn
        self.regressor = regressor
        self.loss = loss
        self.lr = lr
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.target_transform = target_transform
        self.report_intermediates = report_intermediates
        self.report_dir = report_dir
        self.save_epoch_reports = save_epoch_reports
        self.report_every_n_epochs = max(1, int(report_every_n_epochs))
        self.pdff_bins = pdff_bins or [0.0, 5.75, 15.50, 21.35, float(clamp_max)]

        self.save_hyperparameters(
            ignore=["cnn", "rnn", "regressor", "loss", "report_intermediates"]
        )

        self._train_outputs: dict[str, list[Any]] = {}
        self._val_outputs: dict[str, list[Any]] = {}

    def forward(self, x: Tensor):
        frame_features = self.cnn(x)
        context, attention = self.rnn(frame_features)
        y_hat = self.regressor(context).squeeze(-1)
        y_hat = self._prediction_to_label_space(y_hat)
        if self.report_intermediates:
            return y_hat, frame_features, context, attention
        return y_hat

    def _label_to_training_space(self, y: Tensor) -> Tensor:
        y = torch.clamp(y, min=self.clamp_min, max=self.clamp_max)
        if self.target_transform == "log1p":
            return torch.log1p(y)
        if self.target_transform == "none":
            return y
        raise ValueError(f"Unknown target_transform: {self.target_transform}")

    def _prediction_to_label_space(self, y_hat: Tensor) -> Tensor:
        if self.target_transform == "log1p":
            y_hat = torch.expm1(y_hat)
        elif self.target_transform != "none":
            raise ValueError(f"Unknown target_transform: {self.target_transform}")
        return torch.clamp(y_hat, min=self.clamp_min, max=self.clamp_max)

    def _unpack_batch(self, batch) -> tuple[Tensor, Tensor, list[str], list[int]]:
        x, y = batch[:2]
        if y.ndim == 0:
            y = y.unsqueeze(0)
        if y.ndim > 1:
            y = y.squeeze(-1)

        patient_ids: list[str] = []
        sweep_ids: list[int] = []
        if len(batch) >= 4:
            patient_ids = [str(pid) for pid in batch[2]]
            sweep_raw = batch[3]
            if torch.is_tensor(sweep_raw):
                sweep_ids = [int(s) for s in sweep_raw.detach().cpu().view(-1).tolist()]
            else:
                sweep_ids = [int(s) for s in sweep_raw]
        return x, y, patient_ids, sweep_ids

    @staticmethod
    def _safe_r2(y_hat: Tensor, y: Tensor) -> float:
        if y.numel() < 2:
            return 0.0
        y_mean = torch.mean(y)
        ss_tot = torch.sum((y - y_mean) ** 2)
        ss_res = torch.sum((y - y_hat) ** 2)
        if float(ss_tot) <= 1.0e-8:
            return 0.0
        return float((1.0 - (ss_res / ss_tot)).item())

    @staticmethod
    def _safe_pearson(y_hat: Tensor, y: Tensor) -> float:
        if y.numel() < 2:
            return 0.0
        value = pearson_corrcoef(y_hat, y)
        if torch.isnan(value):
            return 0.0
        return float(value.item())

    @staticmethod
    def _safe_ccc(y_hat: Tensor, y: Tensor) -> float:
        if y.numel() < 2:
            return 0.0
        value = ccc_func(y_hat, y)
        if torch.isnan(value):
            return 0.0
        return float(value.item())

    @staticmethod
    def _bin_label(lo: float, hi: float) -> str:
        def _fmt(v: float) -> str:
            return str(v).replace(".", "p").replace("-", "m")

        return f"{_fmt(lo)}_{_fmt(hi)}"

    def _default_report_dir(self) -> str:
        if self.report_dir:
            return self.report_dir
        root = Path(self.trainer.default_root_dir if self.trainer is not None else ".")
        return str(root / "reports")

    @staticmethod
    def _empty_epoch_outputs() -> dict[str, list[Any]]:
        return {
            "y_hat": [],
            "y": [],
            "batch_mae": [],
            "patient_id": [],
            "sweep_id": [],
        }

    def _gather_epoch_outputs(self, outputs: dict[str, list[Any]]) -> dict[str, list[Any]]:
        payload = {
            "y_hat": [float(v) for v in outputs["y_hat"]],
            "y": [float(v) for v in outputs["y"]],
            "batch_mae": [float(v) for v in outputs["batch_mae"]],
            "patient_id": [str(v) for v in outputs["patient_id"]],
            "sweep_id": [int(v) for v in outputs["sweep_id"]],
        }
        if dist.is_available() and dist.is_initialized():
            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, payload)
            merged = self._empty_epoch_outputs()
            for part in gathered:
                for key in merged:
                    merged[key].extend(part[key])
            return merged
        return payload

    def _compute_epoch_metrics(
        self,
        outputs: dict[str, list[Any]],
        stage: str,
        include_patient_metrics: bool,
    ) -> tuple[dict[str, float], dict[str, pd.DataFrame]]:
        metrics: dict[str, float] = {}
        artifacts: dict[str, pd.DataFrame] = {}
        if len(outputs["y"]) == 0:
            return metrics, artifacts

        y = torch.tensor(outputs["y"], dtype=torch.float32)
        y_hat = torch.tensor(outputs["y_hat"], dtype=torch.float32)
        diff = y_hat - y

        mae_emilio = float(torch.mean(torch.abs(diff)).item())
        mse = float(torch.mean(diff**2).item())
        rmse = float(torch.sqrt(torch.mean(diff**2)).item())
        mae_ghl = float(np.mean(outputs["batch_mae"])) if outputs["batch_mae"] else mae_emilio

        metrics[f"{stage}_mae"] = mae_emilio
        metrics[f"{stage}_mae_emilio"] = mae_emilio
        metrics[f"{stage}_mae_ghl"] = mae_ghl
        metrics[f"{stage}_mse"] = mse
        metrics[f"{stage}_rmse"] = rmse
        metrics[f"{stage}_r2"] = self._safe_r2(y_hat, y)
        metrics[f"{stage}_pearson_r"] = self._safe_pearson(y_hat, y)
        metrics[f"{stage}_ccc"] = self._safe_ccc(y_hat, y)

        if not include_patient_metrics or len(outputs["patient_id"]) == 0:
            return metrics, artifacts

        df_sweep = pd.DataFrame(
            {
                "patient_id": outputs["patient_id"],
                "sweep_idx": outputs["sweep_id"],
                "pdff_true": y.numpy(),
                "pdff_pred": y_hat.numpy(),
            }
        )
        df_sweep["error"] = df_sweep["pdff_pred"] - df_sweep["pdff_true"]
        df_sweep["abs_error"] = np.abs(df_sweep["error"])

        df_patient = (
            df_sweep.groupby("patient_id", as_index=False)
            .agg(
                pdff_true=("pdff_true", "mean"),
                pdff_pred=("pdff_pred", "mean"),
                n_sweeps=("pdff_pred", "size"),
            )
        )
        df_patient["error"] = df_patient["pdff_pred"] - df_patient["pdff_true"]
        df_patient["abs_error"] = np.abs(df_patient["error"])

        patient_y = torch.tensor(df_patient["pdff_true"].to_numpy(), dtype=torch.float32)
        patient_y_hat = torch.tensor(df_patient["pdff_pred"].to_numpy(), dtype=torch.float32)

        metrics[f"{stage}_patient_mae"] = float(df_patient["abs_error"].mean()) if len(df_patient) else 0.0
        metrics[f"{stage}_patient_rmse"] = (
            float(np.sqrt(np.mean(df_patient["error"].to_numpy() ** 2))) if len(df_patient) else 0.0
        )
        metrics[f"{stage}_patient_pearson_r"] = self._safe_pearson(patient_y_hat, patient_y)
        metrics[f"{stage}_patient_ccc"] = self._safe_ccc(patient_y_hat, patient_y)

        bins = list(self.pdff_bins)
        if bins[-1] <= bins[-2]:
            bins[-1] = bins[-2] + 1.0e-3
        sweep_bins = stratified_metrics_by_bins(
            df_sweep["pdff_true"].to_numpy(),
            df_sweep["pdff_pred"].to_numpy(),
            bins=bins,
            name=f"{stage}_sweep_raw",
        )
        patient_bins = stratified_metrics_by_bins(
            df_patient["pdff_true"].to_numpy(),
            df_patient["pdff_pred"].to_numpy(),
            bins=bins,
            name=f"{stage}_patient_raw",
        )

        if self.global_rank == 0:
            print(f"[{stage.upper()}] Sweep-level stratified metrics by bins:")
            print(sweep_bins)
            print(f"[{stage.upper()}] Patient-level stratified metrics by bins:")
            print(patient_bins)

        artifacts["df_sweep"] = df_sweep
        artifacts["df_patient"] = df_patient
        artifacts["sweep_bins"] = sweep_bins
        artifacts["patient_bins"] = patient_bins

        if self.global_rank == 0 and self.save_epoch_reports:
            report_dir = Path(self._default_report_dir())
            report_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = report_dir / f"{stage}_epoch{self.current_epoch:03d}_{timestamp}"
            df_sweep.to_csv(f"{base}_sweep_predictions.csv", index=False)
            sweep_bins.to_csv(f"{base}_sweep_metrics_by_bins.csv", index=False)
            df_patient.to_csv(f"{base}_patient_predictions.csv", index=False)
            patient_bins.to_csv(f"{base}_patient_metrics_by_bins.csv", index=False)
            df_global = pd.DataFrame(
                [
                    {
                        "level": "sweep",
                        "n_samples": int(len(df_sweep)),
                        "mae_emilio": mae_emilio,
                        "mae_ghl": mae_ghl,
                        "rmse": rmse,
                        "mse": mse,
                        "r2": metrics[f"{stage}_r2"],
                        "pearson_r": metrics[f"{stage}_pearson_r"],
                        "ccc": metrics[f"{stage}_ccc"],
                    },
                    {
                        "level": "patient",
                        "n_samples": int(len(df_patient)),
                        "mae": metrics[f"{stage}_patient_mae"],
                        "rmse": metrics[f"{stage}_patient_rmse"],
                        "pearson_r": metrics[f"{stage}_patient_pearson_r"],
                        "ccc": metrics[f"{stage}_patient_ccc"],
                        "mean_sweeps_per_patient": float(df_patient["n_sweeps"].mean()) if len(df_patient) else np.nan,
                    },
                ]
            )
            df_global.to_csv(f"{base}_global_metrics.csv", index=False)

            if stage == "val" and self._should_save_epoch_figures():
                self._save_patient_plots(df_patient=df_patient, base=base)

        return metrics, artifacts

    def _should_save_epoch_figures(self) -> bool:
        epoch_one_based = int(self.current_epoch) + 1
        return epoch_one_based % self.report_every_n_epochs == 0

    @staticmethod
    def _save_patient_plots(df_patient: pd.DataFrame, base: Path) -> None:
        if len(df_patient) == 0:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        true_vals = df_patient["pdff_true"].to_numpy(dtype=float)
        pred_vals = df_patient["pdff_pred"].to_numpy(dtype=float)

        parity_path = f"{base}_patient_parity_raw.png"
        bland_path = f"{base}_patient_bland_altman_raw.png"

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(true_vals, pred_vals, alpha=0.8)
        lo = float(min(np.min(true_vals), np.min(pred_vals)))
        hi = float(max(np.max(true_vals), np.max(pred_vals)))
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, color="black")
        ax.set_xlabel("True PDFF")
        ax.set_ylabel("Predicted PDFF")
        ax.set_title("Patient-Level Parity Plot (Raw)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(parity_path, dpi=200)
        plt.close(fig)

        mean_vals = 0.5 * (pred_vals + true_vals)
        diff_vals = pred_vals - true_vals
        bias = float(np.mean(diff_vals))
        sd = float(np.std(diff_vals, ddof=1)) if len(diff_vals) > 1 else 0.0
        upper = bias + 1.96 * sd
        lower = bias - 1.96 * sd

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(mean_vals, diff_vals, alpha=0.8)
        ax.axhline(bias, linestyle="-", linewidth=1.5, color="black", label="Bias")
        ax.axhline(upper, linestyle="--", linewidth=1.2, color="tab:red", label="+1.96 SD")
        ax.axhline(lower, linestyle="--", linewidth=1.2, color="tab:red", label="-1.96 SD")
        ax.set_xlabel("Mean PDFF")
        ax.set_ylabel("Prediction - Truth")
        ax.set_title("Patient-Level Bland-Altman (Raw)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(bland_path, dpi=200)
        plt.close(fig)

    def _log_epoch_metrics(self, metrics: dict[str, float], prog_bar_keys: set[str]) -> None:
        for name, value in metrics.items():
            self.log(
                name,
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=name in prog_bar_keys,
                sync_dist=False,
                rank_zero_only=False,
            )

    def on_train_epoch_start(self) -> None:
        self._train_outputs = self._empty_epoch_outputs()

    def training_step(self, batch, batch_idx):
        x, y, _, _ = self._unpack_batch(batch)
        y_hat = self.forward(x)
        if isinstance(y_hat, tuple):
            y_hat = y_hat[0]
        if y_hat.ndim == 0:
            y_hat = y_hat.unsqueeze(0)

        y_train = self._label_to_training_space(y)
        y_hat_train = self._label_to_training_space(y_hat)
        loss = self.loss(y_hat_train, y_train)

        batch_mae = F.l1_loss(y_hat, y)
        self._train_outputs["y_hat"].extend(y_hat.detach().cpu().view(-1).tolist())
        self._train_outputs["y"].extend(y.detach().cpu().view(-1).tolist())
        self._train_outputs["batch_mae"].append(float(batch_mae.detach().cpu().item()))

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=int(y.shape[0]),
        )
        return loss

    def on_train_epoch_end(self) -> None:
        outputs = self._gather_epoch_outputs(self._train_outputs)
        metrics, _ = self._compute_epoch_metrics(outputs, stage="train", include_patient_metrics=False)
        self._log_epoch_metrics(metrics, prog_bar_keys={"train_mae", "train_mae_emilio"})

    def on_validation_epoch_start(self) -> None:
        self._val_outputs = self._empty_epoch_outputs()

    def validation_step(self, batch, batch_idx):
        x, y, patient_ids, sweep_ids = self._unpack_batch(batch)
        y_hat = self.forward(x)
        if isinstance(y_hat, tuple):
            y_hat = y_hat[0]
        if y_hat.ndim == 0:
            y_hat = y_hat.unsqueeze(0)

        y_train = self._label_to_training_space(y)
        y_hat_train = self._label_to_training_space(y_hat)
        loss = self.loss(y_hat_train, y_train)
        batch_mae = F.l1_loss(y_hat, y)

        self._val_outputs["y_hat"].extend(y_hat.detach().cpu().view(-1).tolist())
        self._val_outputs["y"].extend(y.detach().cpu().view(-1).tolist())
        self._val_outputs["batch_mae"].append(float(batch_mae.detach().cpu().item()))
        self._val_outputs["patient_id"].extend(patient_ids)
        self._val_outputs["sweep_id"].extend(sweep_ids)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=int(y.shape[0]),
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        outputs = self._gather_epoch_outputs(self._val_outputs)
        metrics, _ = self._compute_epoch_metrics(outputs, stage="val", include_patient_metrics=True)
        self._log_epoch_metrics(
            metrics,
            prog_bar_keys={
                "val_loss",
                "val_mae",
                "val_mae_emilio",
                "val_patient_mae",
                "val_rmse",
                "val_patient_rmse",
                "val_ccc",
                "val_patient_ccc",
            },
        )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def test_step(self, batch, batch_idx):
        x, y, patient_ids, sweep_ids = self._unpack_batch(batch)
        y_hat = self.forward(x)
        if isinstance(y_hat, tuple):
            y_hat = y_hat[0]
        if y_hat.ndim == 0:
            y_hat = y_hat.unsqueeze(0)

        y_train = self._label_to_training_space(y)
        y_hat_train = self._label_to_training_space(y_hat)
        loss = self.loss(y_hat_train, y_train)
        batch_mae = F.l1_loss(y_hat, y)

        if not hasattr(self, "_test_outputs"):
            self._test_outputs = self._empty_epoch_outputs()
        self._test_outputs["y_hat"].extend(y_hat.detach().cpu().view(-1).tolist())
        self._test_outputs["y"].extend(y.detach().cpu().view(-1).tolist())
        self._test_outputs["batch_mae"].append(float(batch_mae.detach().cpu().item()))
        self._test_outputs["patient_id"].extend(patient_ids)
        self._test_outputs["sweep_id"].extend(sweep_ids)

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=int(y.shape[0]),
        )
        return loss

    def on_test_epoch_start(self) -> None:
        self._test_outputs = self._empty_epoch_outputs()

    def on_test_epoch_end(self) -> None:
        outputs = self._gather_epoch_outputs(self._test_outputs)
        metrics, _ = self._compute_epoch_metrics(outputs, stage="test", include_patient_metrics=True)
        self._log_epoch_metrics(metrics, prog_bar_keys={"test_mae", "test_patient_mae"})
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def predict_step(self, batch, batch_idx):
        x = batch[0]
        return self.forward(x)

    def configure_optimizers(self):
        if self.lr is None:
            raise ValueError(
                "No optimizer was provided via LightningCLI and model.lr is None."
            )
        return optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1.0e-8,
            weight_decay=0.0,
        )
