# Ray Tune Hyperparameter Optimization for APNTSMixer

This document explains how to use Ray Tune for automated hyperparameter optimization of the APNTSMixer model on medical time series data.

## Overview

Ray Tune is integrated into the APNTSMixer training pipeline to automatically find optimal hyperparameters that minimize validation Mean Squared Error (MSE). This systematic approach can significantly improve model performance compared to manual hyperparameter tuning.

## Quick Start

### 1. Basic Ray Tune Optimization

```bash
cd tPatchGNN
python run_models.py --use_ray_tune --model APNTSMixer --dataset physionet
```

### 2. Customized Optimization

```bash
python run_models.py \
    --use_ray_tune \
    --model APNTSMixer \
    --dataset physionet \
    --tune_samples 100 \
    --tune_epochs 50 \
    --tune_grace_period 10 \
    --tune_reduction_factor 2 \
    --history 24 \
    --n 100000 \
    --gpu 0
```

## Command Line Arguments

### Ray Tune Specific Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_ray_tune` | flag | False | Enable Ray Tune hyperparameter optimization |
| `--tune_samples` | int | 50 | Number of hyperparameter combinations to try |
| `--tune_epochs` | int | 30 | Maximum epochs per trial |
| `--tune_grace_period` | int | 5 | Minimum epochs before early stopping |
| `--tune_reduction_factor` | int | 2 | Reduction factor for ASHA scheduler |

### Important Base Arguments for Ray Tune

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | tPatchGNN | Must be "APNTSMixer" for Ray Tune |
| `--dataset` | str | physionet | Dataset to use (physionet, mimic, ushcn) |
| `--history` | int | 24 | History window size |
| `--n` | int | 1e8 | Dataset size (use substantial size for reliable results) |
| `--gpu` | str | "0" | GPU device to use |
| `--seed` | int | 1 | Random seed for reproducibility |

## Hyperparameter Search Space

Ray Tune will automatically search over the following hyperparameters:

### Learning Parameters
- **Learning Rate (`lr`)**: Log-uniform between 1e-4 and 1e-1
- **Weight Decay (`w_decay`)**: Uniform between 0.0 and 0.1
- **Batch Size (`batch_size`)**: Choice of [128, 256, 512]

### Model Architecture
- **Hidden Dimension (`hid_dim`)**: Choice of [32, 64, 96, 128, 192, 256]
- **Number of Layers (`nlayer`)**: Choice of [1, 2, 3, 4]
- **Number of Patches (`npatch`)**: Choice of [16, 20, 24, 32, 40]
- **Use Attention (`use_attention`)**: Choice of [True, False]

### Training Parameters
- **Patience (`patience`)**: Choice of [5, 8, 10, 12]

## Optimization Process

### 1. ASHA Scheduler
Ray Tune uses the Asynchronous Successive Halving Algorithm (ASHA) for efficient early stopping:
- Allocates more resources to promising trials
- Terminates poorly performing trials early
- Significantly reduces total optimization time

### 2. Optuna Search Algorithm
Uses Optuna's Tree-structured Parzen Estimator (TPE) for intelligent hyperparameter selection:
- Learns from previous trials
- Focuses search on promising regions
- More efficient than random or grid search

### 3. Automatic Result Tracking
- All trials are logged and tracked
- Best configuration is automatically identified
- Results are saved to files for future use

## Output and Results

### During Optimization
Ray Tune will display:
- Progress of individual trials
- Current best performance
- Resource allocation across trials

### After Completion
Ray Tune provides:
- **Best hyperparameter configuration**
- **Final validation metrics** (MSE, MAE, RMSE, MAPE)
- **Configuration file** saved as `best_config_{dataset}_{model}.txt`

### Example Output
```
==============================================================
RAY TUNE OPTIMIZATION RESULTS
==============================================================
Best trial config: {
    'lr': 0.0087, 'w_decay': 0.0234, 'hid_dim': 128, 
    'nlayer': 2, 'npatch': 24, 'batch_size': 256, 
    'use_attention': True, 'patience': 8
}
Best trial final validation MSE: 0.004923
Best trial final validation MAE: 0.051204
Best trial final validation RMSE: 0.070163
Best trial final validation MAPE: 12.34%
Best trial reached epoch: 27
==============================================================
```

## Using the Best Configuration

After Ray Tune completes, use the best configuration for final training:

```bash
python run_models.py \
    --model APNTSMixer \
    --dataset physionet \
    --lr 0.0087 \
    --w_decay 0.0234 \
    --hid_dim 128 \
    --nlayer 2 \
    --npatch 24 \
    --batch_size 256 \
    --use_attention True \
    --patience 8 \
    --history 24 \
    --epoch 100 \
    --n 100000
```

## Performance Tips

### 1. Resource Configuration
- **GPU Memory**: Ensure sufficient GPU memory for parallel trials
- **CPU Cores**: More cores allow more parallel trials
- **Ray Resources**: Ray will automatically detect and use available resources

### 2. Search Efficiency
- Start with fewer samples (20-30) for initial exploration
- Increase samples (50-100+) for thorough optimization
- Use longer grace periods for stable datasets
- Use shorter grace periods for quick convergence

### 3. Dataset Considerations
- Use substantial dataset size (`--n 100000+`) for reliable validation metrics
- Ensure consistent data preprocessing across trials
- Consider multiple dataset splits for robust evaluation

## Advanced Usage

### 1. Custom Search Space
To modify the search space, edit the `get_apn_tsmixer_search_space()` function in `run_models.py`:

```python
def get_apn_tsmixer_search_space():
    search_space = {
        "lr": tune.loguniform(1e-5, 1e-2),  # Narrower range
        "hid_dim": tune.choice([64, 128, 256]),  # Fewer choices
        # Add other parameters...
    }
    return search_space
```

### 2. Different Schedulers
Replace ASHA with other schedulers:

```python
# Population Based Training
from ray.tune.schedulers import PopulationBasedTraining
scheduler = PopulationBasedTraining(...)

# Hyperband
from ray.tune.schedulers import HyperBandScheduler
scheduler = HyperBandScheduler(...)
```

### 3. Multi-Objective Optimization
Optimize for multiple metrics simultaneously by modifying the `tune.report()` call and search configuration.

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce `tune_samples` or batch sizes in search space
   - Use smaller models or reduce `hid_dim` range
   - Enable gradient checkpointing if available

2. **Ray Initialization Errors**
   - Ensure Ray is properly installed: `pip install ray[tune]`
   - Check for port conflicts if running multiple Ray instances
   - Try `ray stop` before running new optimization

3. **Slow Convergence**
   - Increase `tune_grace_period` for more stable early stopping
   - Reduce `tune_reduction_factor` for less aggressive pruning
   - Check if the search space is too large

4. **Poor Results**
   - Verify the search space includes reasonable ranges
   - Ensure sufficient training data (`--n` parameter)
   - Check that base model architecture is sound

### Performance Monitoring
- Monitor GPU utilization during optimization
- Check Ray dashboard at `http://localhost:8265` for detailed trial information
- Use `ray status` to check cluster resource usage

## Integration with Existing Workflow

Ray Tune seamlessly integrates with the existing APNTSMixer training pipeline:
- All existing command-line arguments remain compatible
- No changes needed to model architecture code
- Results can be directly used with existing evaluation scripts
- Compatible with existing logging and checkpoint systems

## Example Scripts

The repository includes:
- `run_ray_tune_example.py`: Interactive example script
- Command templates in this README
- Integration with existing `run_models.py`

Start with the example script to get familiar with the Ray Tune workflow, then customize for your specific needs.
