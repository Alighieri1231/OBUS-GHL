"""
VideoDatasetBase.py

Base dataset for PDFF video regression from frame directories.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, List, Literal, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class VideoDatasetBase(Dataset, ABC):
    def __init__(
        self,
        df: pd.DataFrame,
        transforms: Sequence[Callable] = (),
        channels: int = 3,
        path_col: str = "sweep_dir",
        label_cols: Union[str, List[str]] = "pdff",
        frames_or_channel_first: Literal["frames", "channel"] = "frames",
        image_dims: Tuple[int, int] = (256, 256),
        frame_glob: str = "*_rgb.png",
        subsample: Callable | None = None,
        return_metadata: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.transforms = list(transforms) if transforms else []
        self.channels = channels
        self.path_col = path_col
        self.label_cols = [label_cols] if isinstance(label_cols, str) else label_cols
        self.frames_or_channel_first = frames_or_channel_first
        self.image_dims = tuple(image_dims)
        self.frame_glob = frame_glob
        self.subsample = subsample
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        frames = self.load_sample(Path(row[self.path_col]))
        frames = self._common_frames_processing(frames)
        labels = self._common_labels_processing(row)
        metadata = {
            "patient_id": str(row.get("patient_id", "")),
            "sweep_id": int(row.get("sweep_id", -1)),
            "sweep_dir": str(row[self.path_col]),
        }
        return self._get_batch(frames, labels, metadata)

    def load_sample(self, sweep_dir: Path) -> torch.Tensor:
        frame_paths = sorted(sweep_dir.glob(self.frame_glob))
        if len(frame_paths) == 0:
            raise FileNotFoundError(
                f"No frames matching '{self.frame_glob}' found in {sweep_dir}"
            )

        if self.subsample is not None:
            frame_paths = self.subsample(frame_paths)

        frames = [self._load_frame(path) for path in frame_paths]
        return torch.stack(frames, dim=0)

    def _load_frame(self, frame_path: Path) -> torch.Tensor:
        image = Image.open(frame_path)
        image = image.convert("RGB" if self.channels == 3 else "L")
        image = image.resize((self.image_dims[1], self.image_dims[0]), Image.Resampling.BILINEAR)
        frame = np.asarray(image, dtype=np.float32) / 255.0
        if frame.ndim == 2:
            frame = frame[None, :, :]
        else:
            frame = np.transpose(frame, (2, 0, 1))
        frame = torch.from_numpy(frame)
        if self.channels == 3 and frame.shape[0] == 1:
            frame = frame.repeat(3, 1, 1)
        elif self.channels == 1 and frame.shape[0] == 3:
            frame = frame[:1]
        return frame

    def _common_frames_processing(self, frames: torch.Tensor) -> torch.Tensor:
        frames = self._process_frames(frames)
        for transform in self.transforms:
            frames = transform(frames)
        if self.frames_or_channel_first == "channel":
            frames = torch.movedim(frames, 1, 0)
        return frames

    def _common_labels_processing(self, row: pd.Series) -> torch.Tensor:
        labels = row[self.label_cols].to_numpy(dtype=np.float32)
        return torch.tensor(labels, dtype=torch.float32).squeeze()

    @abstractmethod
    def _process_frames(self, frames: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _get_batch(
        self, frames: torch.Tensor, labels: torch.Tensor, metadata: dict[str, Any]
    ):
        raise NotImplementedError
