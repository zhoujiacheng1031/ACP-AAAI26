# MACAP Results Directory

This directory contains experimental results, model checkpoints, and evaluation outputs for MACAP.

## Structure

```
results/
├── experiments/          # Main experiment results
│   ├── checkpoints/     # Model checkpoints
│   ├── logs/           # Training and evaluation logs
│   ├── metrics/        # Performance metrics
│   └── configs/        # Saved configurations
├── ablation/           # Ablation study results
├── baselines/          # Baseline comparisons
└── analysis/           # Analysis and visualizations
```

## File Types

- `*.pt` - PyTorch model checkpoints
- `*.json` - Configuration files and metrics
- `*.log` - Training and evaluation logs
- `*.png`, `*.pdf` - Plots and visualizations
- `*.csv` - Tabular results