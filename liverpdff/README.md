# liverpdff

PDFF regression experiments for liver ultrasound sweeps, following the same modular LightningCLI style as `ghlobus`.

## Data assumptions

The current setup is built around:

- frames root: `/mnt/castaneda_lab/GitHub/liver_ai_25/research_tasks/output_segmentations_total1`
- labels file: `/mnt/castaneda_lab/GitHub/liver_ai_25/Clinical Liver Study enrollments.xlsx`

Expected sweep layout:

```text
output_segmentations_total1/
  patient_1/
    1-1/
      1-1_frame00155_rgb.png
      1-1_frame00155_mask.png
      ...
    1-2/
    ...
  patient_2/
  ...
```

The datamodule reads `*_rgb.png` frames and ignores masks by default.

## Environment

### Create the tested Blackwell environment

```bash
eval "$(conda shell.bash hook)"
conda env create -f environments/lightning_blackwell.yml
conda activate lightning_blackwell
```

### Validate the environment

```bash
PYTHONNOUSERSITE=1 python - <<'PY'
import torch, torchvision, lightning
print(torch.__version__)
print(torchvision.__version__)
print(lightning.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_arch_list())
PY
```

The tested stack uses CUDA 13.0 wheels and works on the RTX PRO 6000 Blackwell GPUs in this workstation.

## Main entrypoint

```bash
PYTHONNOUSERSITE=1 python -m liverpdff.training.train fit --config liverpdff/training/configs/pdff_additive_attention.yaml
```

## Running modes

### 1. Full GPU training

```bash
PYTHONNOUSERSITE=1 python -m liverpdff.training.train fit \
  --config liverpdff/training/configs/pdff_additive_attention.yaml \
  --trainer.accelerator gpu \
  --trainer.devices 1
```

### 2. Quick smoke test

```bash
PYTHONNOUSERSITE=1 python -m liverpdff.training.train fit \
  --config liverpdff/training/configs/pdff_additive_attention.yaml \
  --trainer.fast_dev_run true \
  --data.init_args.batch_size 2 \
  --data.init_args.num_workers 0 \
  --data.init_args.frames 8 \
  --data.init_args.image_dims '[128,128]'
```

### 3. CPU debug mode

```bash
PYTHONNOUSERSITE=1 python -m liverpdff.training.train fit \
  --config liverpdff/training/configs/pdff_additive_attention.yaml \
  --trainer.accelerator cpu \
  --trainer.devices 1 \
  --trainer.fast_dev_run true
```

### 4. Resume from checkpoint

```bash
PYTHONNOUSERSITE=1 python -m liverpdff.training.train fit \
  --config liverpdff/training/configs/pdff_additive_attention.yaml \
  --ckpt_path /path/to/checkpoint.ckpt
```

### 5. Override from CLI

Example: change batch size, frames, and model LR without editing YAML:

```bash
PYTHONNOUSERSITE=1 python -m liverpdff.training.train fit \
  --config liverpdff/training/configs/pdff_additive_attention.yaml \
  --data.init_args.batch_size 4 \
  --data.init_args.frames 32 \
  --model.init_args.lr 1e-4
```

## Available model configs

### `pdff_additive_attention.yaml`

Default sequence model:

- CNN: `ghlobus.models.TvCnn`
- temporal module: `ghlobus.models.BasicAdditiveAttention`
- head: `liverpdff.models.MLPRegressor`

### `pdff_milattention.yaml`

Gated MIL-style sequence pooling:

- CNN: `ghlobus.models.TvCnn`
- temporal module: `ghlobus.models.MilAttention`
- head: `liverpdff.models.MLPRegressor`

### `pdff_lstm.yaml`

Sequence LSTM over frame embeddings:

- CNN: `ghlobus.models.TvCnn`
- temporal module: `liverpdff.models.TemporalLSTM`
- head: `liverpdff.models.MLPRegressor`

### `pdff_meanpool.yaml`

Simple baseline:

- CNN: `ghlobus.models.TvCnn`
- temporal module: `liverpdff.models.TemporalMeanPooling`
- head: `liverpdff.models.MLPRegressor`

### `pdff_convlstm.yaml`

Feature-map temporal model:

- CNN: `ghlobus.models.TvCnnFeatureMap`
- temporal module: `ghlobus.models.TvConvLSTM`
- head: `liverpdff.models.MLPRegressor`

Use this when you want spatial-temporal recurrence instead of frame-vector aggregation.

## Compatible modular patterns

### Vector CNN + vector temporal module

Use:

- `ghlobus.models.TvCnn`
- `ghlobus.models.BasicAdditiveAttention`
- `ghlobus.models.MilAttention`
- `liverpdff.models.TemporalLSTM`
- `liverpdff.models.TemporalMeanPooling`

These expect CNN output of shape `(B, L, E)`.

### Feature-map CNN + spatial-temporal module

Use:

- `ghlobus.models.TvCnnFeatureMap`
- `ghlobus.models.TvConvLSTM`

These expect CNN output of shape `(B, L, C, H, W)`.

## Data module behavior

The PDFF datamodule:

- loads labels from the Excel `Subject ID` and `PDFF` columns
- performs patient-level splits
- defaults to `split_mode=random_stratified`
- stratifies by PDFF bins from the original end-to-end script
- can save train/val patient ID CSVs for reproducibility
- can apply weighted sampling on the training set using PDFF bins

## Common command examples

### Additive attention with smaller batch

```bash
PYTHONNOUSERSITE=1 python -m liverpdff.training.train fit \
  --config liverpdff/training/configs/pdff_additive_attention.yaml \
  --data.init_args.batch_size 4
```

### LSTM model

```bash
PYTHONNOUSERSITE=1 python -m liverpdff.training.train fit \
  --config liverpdff/training/configs/pdff_lstm.yaml
```

### ConvLSTM model

```bash
PYTHONNOUSERSITE=1 python -m liverpdff.training.train fit \
  --config liverpdff/training/configs/pdff_convlstm.yaml
```

### Fixed validation split

```bash
PYTHONNOUSERSITE=1 python -m liverpdff.training.train fit \
  --config liverpdff/training/configs/pdff_additive_attention.yaml \
  --data.init_args.split_mode fixed_val \
  --data.init_args.split_csv_dir ./outputs/liverpdff/splits
```

### Restrict to selected sweeps

```bash
PYTHONNOUSERSITE=1 python -m liverpdff.training.train fit \
  --config liverpdff/training/configs/pdff_additive_attention.yaml \
  --data.init_args.include_sweeps '[1,2,3,4,5]'
```

## Files

- training entrypoint: `liverpdff/training/train.py`
- default datamodule: `liverpdff/data/VideoDataModuleTraining.py`
- regression model: `liverpdff/models/Cnn2RnnRegressor.py`
- reusable utilities: `liverpdff/utilities/pdff_utils.py`


## For runnning sh file

cd /mnt/castaneda_lab/GitHub/OBUS-GHL

MAX_EPOCHS=100 \
DEVICES=2 \
ACCELERATOR=gpu \
STRATEGY=ddp \
LIMIT_TRAIN_BATCHES=1.0 \
LIMIT_VAL_BATCHES=1.0 \
WANDB_MODE=online \
./liverpdff/training/run_experiments_sequential.sh


Only specific configs

MAX_EPOCHS=100 \
DEVICES=2 \
ACCELERATOR=gpu \
STRATEGY=ddp \
LIMIT_TRAIN_BATCHES=1.0 \
LIMIT_VAL_BATCHES=1.0 \
WANDB_MODE=online \
./liverpdff/training/run_experiments_sequential.sh \
  liverpdff/training/configs/pdff_additive_attention.yaml \
  liverpdff/training/configs/pdff_lstm.yaml

MAX_EPOCHS=100 \
DEVICES=1 \
ACCELERATOR=gpu \
STRATEGY=auto \
LIMIT_TRAIN_BATCHES=1.0 \
LIMIT_VAL_BATCHES=1.0 \
WANDB_MODE=online \
./liverpdff/training/run_experiments_sequential.sh \
  liverpdff/training/configs/pdff_additive_attention.yaml \

MAX_EPOCHS=100 \
DEVICES=1 \
ACCELERATOR=gpu \
STRATEGY=auto \
LIMIT_TRAIN_BATCHES=1.0 \
LIMIT_VAL_BATCHES=1.0 \
WANDB_MODE=online \
./liverpdff/training/run_experiments_sequential.sh \
  liverpdff/training/configs/pdff_lstm.yaml \
