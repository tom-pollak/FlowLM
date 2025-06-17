#!/usr/bin/env python3
"""
Production-grade training script for FlowLM and LLaDA approaches.
Supports both masked language modeling (LLaDA) and any-position diffusion (FlowLM).
"""

import os
import argparse
from functools import partial

import torch
import wandb
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback

from datasets import load_dataset, Dataset, DatasetDict

from flowlm import (
    FlowLMConfig,
    apply_random_mask,
    test_inference,
    accuracy_buckets,
)


class FlowLMCallback(TrainerCallback):
    def __init__(self, model, tokenizer, config: FlowLMConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def on_evaluate(self, *args, **kwargs):
        inference_results = test_inference(
            self.model,
            self.tokenizer,
            self.config,
        )
        self.trainer.log({"inference_results": inference_results})  # type: ignore


def hf_trainer_accuracy(eval_preds):
    logits, labels = eval_preds
    attn = torch.ones_like(labels)
    global_acc, bucket_acc = accuracy_buckets(logits, labels, attn)
    return {"global_acc": global_acc, "bucket_acc": bucket_acc}


def prepare_dataset(config: FlowLMConfig, tokenizer) -> DatasetDict:
    """Load and prepare training/validation datasets."""
    print(f"Loading dataset: {config.dataset.name}")

    # Load raw dataset
    raw_ds = load_dataset(
        config.dataset.name,
        split=config.dataset.split,
        cache_dir=config.dataset.cache_dir,
    )
    assert isinstance(raw_ds, Dataset)

    print(f"Dataset size: {len(raw_ds)}")

    mask_fn = partial(
        apply_random_mask,
        tokenizer=tokenizer,
        strategy=config.masking.strategy.value,
        ratio=config.masking.ratio,
        max_len=config.dataset.max_length,
    )

    # Process dataset
    proc_ds = raw_ds.map(
        mask_fn,
        remove_columns=raw_ds.column_names,
        num_proc=config.dataset.num_proc,
        batched=True,
    ).with_format(type="torch")

    # Train/validation split
    dd = proc_ds.train_test_split(train_size=config.dataset.train_split)

    print(f"Training samples: {len(dd['train'])}")
    print(f"Validation samples: {len(dd['test'])}")

    return dd


def main():
    parser = argparse.ArgumentParser(description="Train FlowLM or LLaDA models")
    parser.add_argument("--config", type=str, help="Path to configuration YAML file")
    parser.add_argument(
        "--resume", type=str, help="Resume training from checkpoint directory"
    )

    args = parser.parse_args()

    if not args.config:
        parser.error("Must specify --config")
    elif not os.path.exists(args.config):
        parser.error(f"Config file not found: {args.config}")

    # Load config
    config: FlowLMConfig = OmegaConf.merge(  # type: ignore
        OmegaConf.structured(FlowLMConfig), OmegaConf.load(args.config)
    )

    output_dir = config.logging.save_dir
    if os.path.exists(output_dir):
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    os.makedirs(output_dir)

    # Initialize wandb
    wandb.init(
        project=config.logging.wandb.project,
        name=config.logging.wandb.name,
        config=OmegaConf.to_container(config, resolve=True),  # type: ignore
    )

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"Loading model: {config.model.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_id)

    model_kwargs = {
        "torch_dtype": getattr(torch, config.model.torch_dtype),
        "device_map": config.model.device_map,
        "low_cpu_mem_usage": config.model.low_cpu_mem_usage,
    }

    model = AutoModelForMaskedLM.from_pretrained(config.model.model_id, **model_kwargs)

    print(f"Mask token: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")

    # Prepare datasets
    dd = prepare_dataset(config, tokenizer)

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type="cosine",
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
        torch_compile=config.model.compile,
        torch_compile_mode=config.model.compile_mode,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.0,  # We handle masking in preprocessing
    )

    # Callback
    flowlm_callback = FlowLMCallback(model, tokenizer, config)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dd["train"],
        eval_dataset=dd["test"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=hf_trainer_accuracy,
        callbacks=[flowlm_callback],
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming training from: {args.resume}")
    else:
        print("Starting training...")

    trainer.train(resume_from_checkpoint=args.resume)

    # Save final model
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

    if config.logging.hf.log_model:
        trainer.push_to_hub(
            repo_id=config.logging.hf.repo_id,
            private=config.logging.hf.private,
            model_name=config.logging.hf.model_name,
        )

    wandb.finish()


if __name__ == "__main__":
    main()
