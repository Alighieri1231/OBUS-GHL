"""
VideoDatasetTraining.py

Training dataset for PDFF regression from liver sweep frame directories.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from liverpdff.data.VideoDatasetBase import VideoDatasetBase


class VideoDatasetTraining(VideoDatasetBase):
    def __init__(self, frames: int = 64, selection_mode: str = "random", **kwargs) -> None:
        super().__init__(**kwargs)
        self.frames = int(frames)
        self.selection_mode = selection_mode

    def _select_frame_paths(self, frame_paths: Sequence[Path]) -> list[Path]:
        num_available = len(frame_paths)
        if num_available == 0:
            return []
        if num_available <= self.frames:
            return list(frame_paths)

        if self.selection_mode == "random":
            indices = np.sort(
                np.random.choice(num_available, size=self.frames, replace=False)
            )
        elif self.selection_mode == "uniform":
            indices = np.linspace(0, num_available - 1, self.frames, dtype=int)
        else:
            raise ValueError(f"Unknown selection_mode: {self.selection_mode}")
        return [frame_paths[i] for i in indices.tolist()]

    def load_sample(self, sweep_dir: Path) -> torch.Tensor:
        frame_paths = sorted(sweep_dir.glob(self.frame_glob))
        if len(frame_paths) == 0:
            raise FileNotFoundError(
                f"No frames matching '{self.frame_glob}' found in {sweep_dir}"
            )

        frame_paths = self._select_frame_paths(frame_paths)
        frames = [self._load_frame(path) for path in frame_paths]
        video = torch.stack(frames, dim=0)

        if video.shape[0] < self.frames:
            pad = self.frames - video.shape[0]
            video = torch.cat([video, video[-1:].repeat(pad, 1, 1, 1)], dim=0)
        return video

    def _process_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return frames

    def _get_batch(self, frames: torch.Tensor, labels: torch.Tensor, metadata: dict):
        if self.return_metadata:
            return frames, labels, metadata["patient_id"], metadata["sweep_id"]
        return frames, labels

