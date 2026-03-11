"""
VideoDataModuleTraining.py

LightningDataModule for PDFF sweep regression experiments.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from lightning import LightningDataModule
except ImportError:
    from pytorch_lightning import LightningDataModule

from liverpdff.data.VideoDatasetTraining import VideoDatasetTraining
from liverpdff.utilities.pdff_utils import (
    compute_pdff_sampling_weights,
    discover_patient_ids,
    discover_sweeps,
    find_patient_dir,
    load_pdff_from_excel,
    natural_sort,
    stratified_split_by_pdff,
)


class VideoDataModuleTraining(LightningDataModule):
    def __init__(
        self,
        root_dir: str = "/mnt/castaneda_lab/GitHub/liver_ai_25/research_tasks/output_segmentations_total1",
        excel_path: str = "/mnt/castaneda_lab/GitHub/liver_ai_25/Clinical Liver Study enrollments.xlsx",
        batch_size: int = 8,
        num_workers: int = 4,
        frames: int = 64,
        channels: int = 3,
        image_dims: tuple[int, int] = (256, 256),
        frame_glob: str = "*_rgb.png",
        val_split: float = 0.2,
        seed: int = 42,
        transforms: Sequence[Callable] = (),
        augmentations: Sequence[Callable] = (),
        use_stratified_sampler: bool = True,
        split_mode: str = "random_stratified",
        split_csv_dir: Optional[str] = None,
        split_tag: str = "default",
        save_train_ids: bool = True,
        include_sweeps: Optional[List[int]] = None,
        exclude_sweeps: Optional[List[int]] = None,
        return_metadata: bool = True,
        path_col: str = "sweep_dir",
        label_cols: str | List[str] = "pdff",
        frames_or_channel_first: str = "frames",
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.excel_path = excel_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frames = frames
        self.channels = channels
        self.image_dims = image_dims
        self.frame_glob = frame_glob
        self.val_split = val_split
        self.seed = seed
        self.transforms = list(transforms) if transforms else []
        self.augmentations = list(augmentations) if augmentations else []
        self.use_stratified_sampler = use_stratified_sampler
        self.split_mode = split_mode
        self.split_csv_dir = split_csv_dir
        self.split_tag = split_tag
        self.save_train_ids = save_train_ids
        self.include_sweeps = set(include_sweeps) if include_sweeps else None
        self.exclude_sweeps = set(exclude_sweeps) if exclude_sweeps else None
        self.return_metadata = return_metadata
        self.path_col = path_col
        self.label_cols = label_cols
        self.frames_or_channel_first = frames_or_channel_first

        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.train_ds = None
        self.val_ds = None

    def _split_paths(self) -> tuple[Optional[str], Optional[str]]:
        if self.split_csv_dir is None:
            return None, None
        os.makedirs(self.split_csv_dir, exist_ok=True)
        suffix = "" if not str(self.split_tag).strip() else f"_{self.split_tag}"
        val_csv = os.path.join(self.split_csv_dir, f"val_ids{suffix}.csv")
        train_csv = os.path.join(self.split_csv_dir, f"train_ids{suffix}.csv")
        return val_csv, train_csv

    @staticmethod
    def _read_ids_csv(path: str) -> List[str]:
        df = pd.read_csv(path)
        col = "patient_id" if "patient_id" in df.columns else df.columns[0]
        return df[col].astype(str).tolist()

    @staticmethod
    def _write_ids_csv(path: str, ids: List[str]) -> None:
        pd.DataFrame({"patient_id": natural_sort(ids)}).to_csv(path, index=False)

    def _discover_samples(self, patient_ids: List[str], pdff_map: dict[str, float]) -> pd.DataFrame:
        rows = []
        for patient_id in patient_ids:
            patient_dir = find_patient_dir(self.root_dir, patient_id)
            if patient_dir is None:
                continue
            sweeps = discover_sweeps(
                patient_dir,
                include_sweeps=self.include_sweeps,
                exclude_sweeps=self.exclude_sweeps,
            )
            for sweep_dir, sweep_id in sweeps:
                frame_count = len(list(Path(sweep_dir).glob(self.frame_glob)))
                if frame_count == 0:
                    continue
                rows.append(
                    {
                        "patient_id": str(patient_id),
                        "sweep_id": int(sweep_id),
                        "sweep_dir": str(sweep_dir),
                        "pdff": float(pdff_map[patient_id]),
                        "num_frames": int(frame_count),
                    }
                )
        return pd.DataFrame(rows)

    def _split_patient_ids(self, patient_ids: List[str], pdff_map: dict[str, float]) -> tuple[List[str], List[str]]:
        val_csv, train_csv = self._split_paths()
        patient_set = set(patient_ids)

        def intersect(ids: List[str]) -> List[str]:
            return natural_sort(list(set(map(str, ids)) & patient_set))

        if self.split_mode == "random":
            rng = np.random.default_rng(self.seed)
            shuffled = patient_ids.copy()
            rng.shuffle(shuffled)
            n_val = max(1, int(len(shuffled) * self.val_split))
            val_ids = natural_sort(shuffled[:n_val])
            train_ids = natural_sort(shuffled[n_val:])
        elif self.split_mode == "random_stratified":
            train_ids, val_ids = stratified_split_by_pdff(
                patients=patient_ids,
                pdff_map=pdff_map,
                val_split=self.val_split,
                seed=self.seed,
            )
        elif self.split_mode == "fixed_val":
            if val_csv is None or not os.path.isfile(val_csv):
                raise RuntimeError(f"Missing validation split file: {val_csv}")
            val_ids = intersect(self._read_ids_csv(val_csv))
            train_ids = [pid for pid in patient_ids if pid not in set(val_ids)]
        elif self.split_mode == "fixed_both":
            if val_csv is None or not os.path.isfile(val_csv):
                raise RuntimeError(f"Missing validation split file: {val_csv}")
            if train_csv is None or not os.path.isfile(train_csv):
                raise RuntimeError(f"Missing training split file: {train_csv}")
            val_ids = intersect(self._read_ids_csv(val_csv))
            train_ids = intersect(self._read_ids_csv(train_csv))
        else:
            raise ValueError(f"Unknown split_mode: {self.split_mode}")

        if val_csv is not None and self.split_mode in {"random", "random_stratified"}:
            self._write_ids_csv(val_csv, val_ids)
            if self.save_train_ids and train_csv is not None:
                self._write_ids_csv(train_csv, train_ids)

        return train_ids, val_ids

    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in (None, "fit", "validate"):
            raise ValueError("VideoDataModuleTraining only supports fit/validate stages.")

        pdff_map = load_pdff_from_excel(self.excel_path)
        patient_ids = discover_patient_ids(self.root_dir)
        patient_ids = [patient_id for patient_id in patient_ids if patient_id in pdff_map]
        if len(patient_ids) == 0:
            raise RuntimeError("No overlapping patients found between sweep root and Excel labels.")

        train_ids, val_ids = self._split_patient_ids(patient_ids, pdff_map)
        self.train_df = self._discover_samples(train_ids, pdff_map)
        self.val_df = self._discover_samples(val_ids, pdff_map)

        self.train_ds = VideoDatasetTraining(
            df=self.train_df,
            transforms=[*self.transforms, *self.augmentations],
            channels=self.channels,
            path_col=self.path_col,
            label_cols=self.label_cols,
            frames_or_channel_first=self.frames_or_channel_first,
            image_dims=self.image_dims,
            frame_glob=self.frame_glob,
            frames=self.frames,
            selection_mode="random",
            return_metadata=self.return_metadata,
        )
        self.val_ds = VideoDatasetTraining(
            df=self.val_df,
            transforms=self.transforms,
            channels=self.channels,
            path_col=self.path_col,
            label_cols=self.label_cols,
            frames_or_channel_first=self.frames_or_channel_first,
            image_dims=self.image_dims,
            frame_glob=self.frame_glob,
            frames=self.frames,
            selection_mode="uniform",
            return_metadata=self.return_metadata,
        )

    def train_dataloader(self) -> DataLoader:
        sampler = None
        shuffle = True
        trainer = getattr(self, "trainer", None)
        world_size = int(getattr(trainer, "world_size", 1) or 1)
        if self.use_stratified_sampler and world_size == 1 and self.train_df is not None:
            weights = compute_pdff_sampling_weights(self.train_df["pdff"])
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(weights),
                replacement=True,
            )
            shuffle = False

        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
