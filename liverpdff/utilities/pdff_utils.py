"""
pdff_utils.py

Utilities for PDFF label loading, patient-level stratified splitting,
and balanced sampling weights.

Author: Auto-generated following ghlobus conventions.
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Excel loading
# ---------------------------------------------------------------------------
def load_pdff_from_excel(excel_path: str) -> Dict[str, float]:
    """
    Read the clinical Excel and return {patient_id_str: pdff_float}.

    Searches for 'Subject ID' and 'PDFF' columns by fuzzy name matching,
    consistent with the end-to-end training script.
    """
    df = pd.read_excel(excel_path)

    id_col, pdff_col = None, None
    for c in df.columns:
        lc = str(c).strip().lower()
        if id_col is None and (
            "subject id" in lc
            or lc == "subjectid"
            or ("subject" in lc and "id" in lc)
        ):
            id_col = c
        if pdff_col is None and (
            "proton density fat fraction" in lc
            or lc == "pdff"
            or "fat fraction" in lc
        ):
            pdff_col = c

    if id_col is None or pdff_col is None:
        raise RuntimeError(
            f"Could not find Subject ID / PDFF columns in {excel_path}. "
            f"Columns: {list(df.columns)}"
        )

    out: Dict[str, float] = {}
    for _, row in df.iterrows():
        sid = str(row[id_col]).strip()
        if sid == "" or sid.lower() == "nan":
            continue
        try:
            pdff = float(row[pdff_col])
            if np.isfinite(pdff):
                out[sid] = float(np.clip(pdff, 0.0, 100.0))
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Patient directory discovery
# ---------------------------------------------------------------------------
_ALLOWED_SWEEPS = set(range(1, 14))


def natural_sort(values: List[str]) -> List[str]:
    """Sort strings containing numeric ids in human order."""
    def key(value: str):
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]
    return sorted(values, key=key)


def parse_patient_id(dirname: str) -> Optional[str]:
    """Return the numeric patient id from a directory name like 'patient_42'."""
    m = re.fullmatch(r"patient_(\d+)", dirname)
    if m:
        return m.group(1)
    if re.fullmatch(r"\d+", dirname):
        return dirname
    return None


def discover_patient_ids(root_dir: str) -> List[str]:
    """Return sorted patient ids discovered under the sweep root directory."""
    patient_ids = []
    for path in Path(root_dir).iterdir():
        if not path.is_dir():
            continue
        patient_id = parse_patient_id(path.name)
        if patient_id is not None:
            patient_ids.append(patient_id)
    return natural_sort(list(set(patient_ids)))


def find_patient_dir(root_dir: str, patient_id: str) -> Optional[str]:
    """Return the matching patient directory for a patient id."""
    candidates = [
        Path(root_dir) / str(patient_id),
        Path(root_dir) / f"patient_{patient_id}",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return None


def discover_sweeps(
    patient_dir: str,
    include_sweeps: Optional[set] = None,
    exclude_sweeps: Optional[set] = None,
) -> List[Tuple[str, int]]:
    """
    Discover sweep sub-directories under a patient directory.

    Returns list of (sweep_dir_path, sweep_idx) sorted by sweep_idx.
    Each sweep dir is e.g. ``patient_dir/1-6`` and contains:
      - ``*_rgb.png`` and ``*_mask.png`` frames
      - ``tinyusfm_features_base.pt``, ``tinyusfm_features_liver.pt``, etc.
    """
    sweeps = []
    for entry in os.listdir(patient_dir):
        sweep_path = os.path.join(patient_dir, entry)
        if not os.path.isdir(sweep_path):
            continue
        # Parse sweep index from e.g. "1-6"
        m = re.fullmatch(r"\d+-(\d+)", entry)
        if m is None:
            continue
        sid = int(m.group(1))
        if sid not in _ALLOWED_SWEEPS:
            continue
        if include_sweeps is not None and sid not in include_sweeps:
            continue
        if exclude_sweeps is not None and sid in exclude_sweeps:
            continue
        sweeps.append((sweep_path, sid))
    sweeps.sort(key=lambda x: x[1])
    return sweeps


# ---------------------------------------------------------------------------
# Stratified patient-level split by PDFF bins
# ---------------------------------------------------------------------------
def stratified_split_by_pdff(
    patients: List[str],
    pdff_map: Dict[str, float],
    val_split: float = 0.2,
    seed: int = 42,
    pdff_max: float = 40.0,
) -> Tuple[List[str], List[str]]:
    """
    Split patients into train/val sets, stratified by PDFF severity bins.

    Bins follow steatosis thresholds: [0, 5.75, 15.50, 21.35, pdff_max].
    """
    rng = np.random.default_rng(seed)

    edges = sorted(list(set([0.0, 5.75, 15.50, 21.35, float(pdff_max)])))
    if len(edges) < 3:
        edges = [0.0, float(pdff_max) * 0.5, float(pdff_max)]
    if edges[-1] <= edges[-2]:
        edges[-1] = edges[-2] + 1e-3
    bins = np.array(edges[:-1] + [edges[-1] + 1e-3], dtype=float)

    y = np.array([float(pdff_map[p]) for p in patients], dtype=float)
    y = np.clip(y, bins[0], bins[-1] - 1e-6)
    bin_idx = np.digitize(y, bins, right=False) - 1

    bin_to_ids: Dict[int, List[str]] = {}
    for pid, b in zip(patients, bin_idx):
        bin_to_ids.setdefault(int(b), []).append(pid)

    val_ids, train_ids = [], []
    for _b, ids in bin_to_ids.items():
        ids = ids.copy()
        rng.shuffle(ids)
        n = len(ids)
        n_val_bin = int(np.round(val_split * n))
        if n >= 2:
            n_val_bin = max(1, min(n - 1, n_val_bin))
        else:
            n_val_bin = 0
        val_ids.extend(ids[:n_val_bin])
        train_ids.extend(ids[n_val_bin:])

    if len(val_ids) == 0:
        shuffled = patients.copy()
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_val = max(1, int(np.floor(val_split * n_total)))
        val_ids = shuffled[:n_val]
        train_ids = shuffled[n_val:]

    return natural_sort(train_ids), natural_sort(val_ids)


# ---------------------------------------------------------------------------
# Sampling weights for WeightedRandomSampler (PDFF-bin balanced)
# ---------------------------------------------------------------------------
def compute_pdff_sampling_weights(pdff_values: pd.Series) -> list:
    """
    Compute per-sample weights so that PDFF severity bins are equally
    represented during training (via WeightedRandomSampler).
    """
    bins = pd.cut(
        x=pdff_values.astype(float),
        bins=[0.0, 5.75, 15.50, 21.35, 100.0],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    )
    vcs = bins.value_counts(dropna=False)
    n = vcs.sum()
    num_bins = len(vcs.index)
    weights = (1.0 / num_bins) / (vcs / n)
    weights_by_index = bins.apply(lambda x: weights[x]).tolist()
    return weights_by_index
