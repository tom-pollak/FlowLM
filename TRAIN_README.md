# FlowLM Training System

This document describes the production-grade training system for FlowLM and LLaDA experiments.

## Quick Start

### Using Preset Configurations

```bash
# Train LLaDA baseline
uv run python train.py --preset llada

# Train FlowLM experiment
uv run python train.py --preset flowlm
```

### Using Custom Configurations

```bash
# Train with custom YAML config
uv run python train.py --config configs/my_config.yaml

# Resume training from checkpoint
uv run python train.py --config configs/flowlm.yaml --resume checkpoints/flowlm/checkpoint-1000
```

## Configuration System

The training system uses structured configurations with OmegaConf and dataclasses for type safety and validation.

### Configuration Structure

```python
@dataclass
class FlowLMConfig:
    model: ModelConfig          # Model settings
    dataset: DatasetConfig      # Dataset and preprocessing
    masking: MaskingConfig      # Masking strategy
    training: TrainingConfig    # Training hyperparameters
    dataloader: DataLoaderConfig # DataLoader settings
    evaluation: EvaluationConfig # Evaluation settings
    inference: InferenceConfig  # Inference testing
    logging: LoggingConfig      # Logging and wandb
```

### Preset Configurations

- **LLaDA**: `--preset llada` - Baseline masked language modeling approach
- **FlowLM**: `--preset flowlm` - Novel any-position diffusion approach

### Custom YAML Configurations

YAML configurations only need to specify values that differ from defaults:

```yaml
# configs/my_experiment.yaml
masking:
  strategy: "FLOWLM_STRATEGY"
  min_ratio: 0.2

training:
  batch_size: 64
  learning_rate: 1e-4

logging:
  wandb:
    name: "my-experiment"
    tags: ["custom", "experiment"]
```

## Features

### Wandb Integration

- Automatic experiment tracking with `report_to="wandb"`
- Configurable project names, run names, and tags
- Model artifact logging
- Inference examples logged as tables

### Training Features

- HuggingFace Trainer integration
- Model compilation with `torch.compile`
- Gradient checkpointing
- Mixed precision training (bf16)
- Automatic checkpoint saving and resuming

### Inference Testing

- Automatic testing of both LLaDA and FlowLM inference modes
- Configurable test prompts
- Confidence threshold and replacement strategies for FlowLM

### Masking Strategies

- **BERT_STRATEGY**: 80% mask, 10% random, 10% unchanged
- **LLADA_STRATEGY**: 90% mask, 10% random, 0% unchanged
- **FLOWLM_STRATEGY**: 30% mask, 70% random, 0% unchanged

## Directory Structure

```
├── train.py                    # Main training script
├── configs/
│   ├── llada.yaml             # LLaDA baseline config
│   └── flowlm.yaml            # FlowLM experiment config
├── src/flowlm/
│   ├── config.py              # Structured configuration classes
│   ├── data.py                # Data processing utilities
│   ├── inference.py           # Inference utilities
│   └── evaluation.py          # Evaluation utilities
├── tests/
│   ├── test_config.py         # Configuration system tests
│   └── test_train_integration.py # Training integration tests
└── checkpoints/               # Model checkpoints (created during training)
    ├── llada/
    └── flowlm/
```

## Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test files
uv run pytest tests/test_config.py -v
uv run pytest tests/test_train_integration.py -v

# Run integration tests only
uv run pytest tests/ -m integration -v
```

## Key Differences: LLaDA vs FlowLM

### LLaDA (Baseline)
- High masking ratios (15%-99%)
- Only masked positions get predicted during inference
- Standard masked language modeling approach

### FlowLM (Experimental)
- Uses FLOWLM_STRATEGY masking (30% mask, 70% random tokens)
- Any position can be refined during inference
- Confidence-based token replacement
- Novel "any-position diffusion" approach

## Monitoring and Logging

All experiments are logged to Weights & Biases with:
- Training and validation curves
- Hyperparameter tracking
- Model artifacts
- Inference examples
- Comprehensive configuration logging

Check your wandb dashboard for real-time training progress and results comparison between LLaDA and FlowLM approaches.
