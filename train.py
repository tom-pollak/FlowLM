#!/usr/bin/env python3
"""
Production-grade training script for FlowLM and LLaDA approaches.
Supports both masked language modeling (LLaDA) and any-position diffusion (FlowLM).
"""

import os
import argparse
from functools import partial
from typing import Dict, Tuple

import torch
import wandb
from omegaconf import OmegaConf
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset, Dataset

from flowlm import (
    apply_random_mask,
    iterative_decode,
    MaskingStrategy,
    MaskingRatio,
    BERT_STRATEGY,
    LLADA_STRATEGY,
    FLOWLM_STRATEGY,
    FlowLMConfig,
    get_llada_config,
    get_flowlm_config,
)


def load_config(config_path: str) -> FlowLMConfig:
    """Load and merge configuration from YAML file with structured config."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML config
    yaml_config = OmegaConf.load(config_path)

    # Create structured config
    structured_config = OmegaConf.structured(FlowLMConfig)

    # Merge YAML config into structured config
    config = OmegaConf.merge(structured_config, yaml_config)

    return config


def get_masking_strategy(strategy_name: str) -> MaskingStrategy:
    """Get masking strategy by name."""
    strategies = {
        "BERT_STRATEGY": BERT_STRATEGY,
        "LLADA_STRATEGY": LLADA_STRATEGY,
        "FLOWLM_STRATEGY": FLOWLM_STRATEGY,
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unknown masking strategy: {strategy_name}")

    return strategies[strategy_name]


def prepare_dataset(config: FlowLMConfig, tokenizer) -> Tuple[Dataset, Dataset]:
    """Load and prepare training/validation datasets."""
    print(f"Loading dataset: {config.dataset.name}")

    # Load raw dataset
    raw_ds = load_dataset(
        config.dataset.name,
        split=config.dataset.split,
        cache_dir=config.dataset.cache_dir,
    )

    # Ensure we have a Dataset object, not DatasetDict
    if hasattr(raw_ds, 'keys'):
        # If it's a DatasetDict, take the first available split
        raw_ds = raw_ds[list(raw_ds.keys())[0]]

    print(f"Dataset size: {len(raw_ds)}")

    # Get masking configuration
    strategy = get_masking_strategy(config.masking.strategy)
    masking_ratio = MaskingRatio(
        min_ratio=config.masking.min_ratio,
        max_ratio=config.masking.max_ratio
    )

    # Create masking function
    mask_fn = partial(
        apply_random_mask,
        tokenizer=tokenizer,
        strategy=strategy,
        ratio=masking_ratio,
        max_len=config.dataset.max_length,
    )

    # Process dataset
    proc_ds = raw_ds.map(
        mask_fn,
        remove_columns=raw_ds.column_names,
        num_proc=config.dataset.num_proc,
    )
    proc_ds.set_format(type="torch")

    # Train/validation split
    train_size = int(config.dataset.train_split * len(proc_ds))
    train_ds = proc_ds.shuffle(seed=42).select(range(train_size))
    val_ds = proc_ds.select(range(train_size, len(proc_ds)))

    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")

    return train_ds, val_ds


def setup_wandb(config: FlowLMConfig) -> None:
    """Initialize wandb logging."""
    wandb.init(
        project=config.logging.wandb.project,
        name=config.logging.wandb.name,
        tags=config.logging.wandb.tags,
        config=OmegaConf.to_container(config, resolve=True),
    )

    print(f"Initialized wandb project: {config.logging.wandb.project}")


def test_inference(
    model, tokenizer, config: FlowLMConfig, device: str
) -> Dict[str, str]:
    """Test model inference with configured prompts."""
    print("\\nTesting inference...")

    results = {}

    for prompt in config.inference.test_prompts:
        print(f"\\nTesting prompt: {prompt}")

        # Test LLaDA-style inference
        result_llada = iterative_decode(
            model,
            tokenizer,
            prompt,
            answer_length=config.inference.answer_length,
            device=device,
            mask_only=True,
        )
        results[f"llada_{prompt[:20]}"] = result_llada
        print(f"LLaDA result: {result_llada}")

        # Test FlowLM-style inference if configured
        if config.inference.test_both_modes:
            result_flowlm = iterative_decode(
                model,
                tokenizer,
                prompt,
                answer_length=config.inference.answer_length,
                device=device,
                mask_only=False,
                confidence_threshold=config.inference.confidence_threshold,
                max_replacements=config.inference.max_replacements,
            )
            results[f"flowlm_{prompt[:20]}"] = result_flowlm
            print(f"FlowLM result: {result_flowlm}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train FlowLM or LLaDA models")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["llada", "flowlm"],
        help="Use a preset configuration (llada or flowlm)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume training from checkpoint directory"
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = load_config(args.config)
        print(f"Loaded config from: {args.config}")
    elif args.preset == "llada":
        config = OmegaConf.structured(get_llada_config())
        print("Using LLaDA preset configuration")
    elif args.preset == "flowlm":
        config = OmegaConf.structured(get_flowlm_config())
        print("Using FlowLM preset configuration")
    else:
        parser.error("Must specify either --config or --preset")

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize wandb
    setup_wandb(config)

    # Load model and tokenizer
    print(f"Loading model: {config.model.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_id)

    model_kwargs = {
        "torch_dtype": getattr(torch, config.model.torch_dtype),
        "device_map": config.model.device_map,
        "low_cpu_mem_usage": config.model.low_cpu_mem_usage,
    }

    model = AutoModelForMaskedLM.from_pretrained(
        config.model.model_id, **model_kwargs
    )

    print(f"Mask token: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")

    # Prepare datasets
    train_ds, val_ds = prepare_dataset(config, tokenizer)

    # Setup training arguments
    output_dir = config.logging.save_dir
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        logging_steps=config.logging.log_every,
        eval_strategy="steps",
        eval_steps=config.evaluation.eval_steps,
        save_steps=config.evaluation.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=config.dataloader.num_workers,
        dataloader_pin_memory=config.dataloader.pin_memory,
        gradient_checkpointing=True,
        bf16=config.model.torch_dtype == "bfloat16",
        report_to="wandb",
        run_name=config.logging.wandb.name,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.0,  # We handle masking in preprocessing
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Compile model if specified
    if config.model.compile:
        print("Compiling model...")
        torch.compile(model, mode=config.model.compile_mode)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        # Start training
        print("Starting training...")
        trainer.train()

    # Final evaluation
    print("\\nRunning final evaluation...")
    eval_results = trainer.evaluate()
    print("Final evaluation results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")

    # Test inference
    inference_results = test_inference(model, tokenizer, config, device)

    # Log inference results to wandb
    if wandb.run is not None:
        wandb.log({"inference_examples": inference_results})

    # Save final model
    print(f"\\nSaving model to: {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Log model as wandb artifact if configured
    if config.logging.wandb.log_model and wandb.run is not None:
        run_name = wandb.run.name or "model"
        artifact = wandb.Artifact(
            name=f"model-{run_name}",
            type="model",
            description=f"Trained {config.masking.strategy} model",
        )
        artifact.add_dir(output_dir)
        wandb.log_artifact(artifact)

    print("Training completed successfully!")
    wandb.finish()


if __name__ == "__main__":
    main()
