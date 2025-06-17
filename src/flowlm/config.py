"""Structured configuration classes for FlowLM training."""

from dataclasses import dataclass, field
from typing import List, Optional

from .data import MaskEnum, MaskingRatio


@dataclass
class ModelConfig:
    """Model configuration."""

    model_id: str = "answerdotai/ModernBERT-large"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    low_cpu_mem_usage: bool = False
    compile: bool = True
    compile_mode: str = "max-autotune"


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    name: str = "allenai/tulu-3-sft-mixture-0225"
    split: str = "train"
    cache_dir: str = "./data"
    max_length: int = 512
    train_split: float = 0.95
    num_proc: int = 24


@dataclass
class MaskingConfig:
    """Masking strategy configuration."""

    strategy: MaskEnum = MaskEnum.LLADA
    ratio: MaskingRatio = MaskingRatio(min_ratio=0.15, max_ratio=0.99)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 32
    num_epochs: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    gradient_clip_val: float = 1.0


@dataclass
class DataLoaderConfig:
    """DataLoader configuration."""

    num_workers: int = 16
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    eval_steps: int = 200
    num_eval_batches: int = 8


@dataclass
class InferenceConfig:
    """Inference testing configuration."""

    test_prompts: List[str] = field(
        default_factory=lambda: [
            "What is the capital of France?",
            "Explain photosynthesis in simple terms.",
            "Write a short story about a robot.",
        ]
    )
    answer_length: int = 16
    test_both_modes: bool = True
    confidence_threshold: float = 0.7
    max_replacements: int = 2


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""

    project: str = "flowlm"
    name: Optional[str] = None
    log_model: bool = True


@dataclass
class HuggingFaceConfig:
    """Hugging Face configuration."""

    log_model: bool = True
    repo_id: str = "flowlm"
    private: bool = False
    model_name: str = "modernbert-large-dllm"


@dataclass
class LoggingConfig:
    """Logging configuration."""

    wandb: WandbConfig = field(default_factory=WandbConfig)
    hf: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    log_every: int = 25
    save_dir: str = "checkpoints"


@dataclass
class FlowLMConfig:
    """Complete FlowLM training configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
