# %%
"""
Baseline ModernBERT diffusion LLM training experiment.
Replicates the mb_dllm notebook training with variable masking ratios.
"""

import os
import random
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from flowlm import apply_random_mask, evaluate_model, iterative_decode, BERT_STRATEGY

# %%
# Configuration
model_id = "answerdotai/ModernBERT-large"
dataset_name = "allenai/tulu-3-sft-mixture-0225"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training hyperparameters
max_len = 512
mask_ratio_min = 0.15
mask_ratio_max = 0.99
batch_size = 32
num_epochs = 1
learning_rate = 2e-5
weight_decay = 0.01
warmup_ratio = 0.06
log_every = 200

print(f"Using device: {device}")

# %%
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

print(f"Mask token: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")

# %%
# Load and process dataset
raw_ds = load_dataset(dataset_name, split="train", cache_dir="./data")
print(f"Dataset size: {len(raw_ds)}")


# Create masking function with our parameters
def mask_fn(example):
    return apply_random_mask(
        example,
        tokenizer,
        strategy=BERT_STRATEGY,  # 80% mask, 10% random, 10% unchanged
        max_len=max_len,
    )


# Process dataset
proc_ds = raw_ds.map(mask_fn, remove_columns=raw_ds.column_names, num_proc=4)
proc_ds.set_format(type="torch")

# %%
# Visualize a sample
sample = random.choice(proc_ds)
decoded_input = tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
print("Sample input with masks:")
print(decoded_input[:500] + "..." if len(decoded_input) > 500 else decoded_input)

# %%
# Train/validation split
train_size = int(0.95 * len(proc_ds))
train_ds = proc_ds.shuffle(seed=42).select(range(train_size))
val_ds = proc_ds.select(range(train_size, len(proc_ds)))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_ds)}")
print(f"Validation samples: {len(val_ds)}")

# %%
# Setup training
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
total_steps = len(train_loader) * num_epochs
warmup_steps = int(warmup_ratio * total_steps)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

# %%
# Training loop
model.train()
global_step = 0
losses, val_losses = [], []
accs, val_accs = [], []

for epoch in range(num_epochs):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    running_loss = 0.0
    running_acc = 0.0

    for step, batch in enumerate(pbar, 1):
        global_step += 1
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        # Track metrics (simple accuracy calculation)
        with torch.no_grad():
            predictions = outputs.logits.argmax(-1)
            mask = batch["labels"] != -100
            correct = (predictions == batch["labels"]) & mask
            acc = correct.sum().item() / mask.sum().item() if mask.sum() > 0 else 0.0

        running_loss += loss.item()
        running_acc += acc
        losses.append(loss.item())
        accs.append(acc)

        # Update progress bar
        if step % 20 == 0:
            pbar.set_postfix({"loss": running_loss / step, "acc": running_acc / step})

        # Validation
        if global_step % log_every == 0:
            val_loss, val_acc, bucket_acc = evaluate_model(model, val_loader, device)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print(
                f"\nStep {global_step:6d} | "
                f"train_loss {running_loss / step:.4f} train_acc {running_acc / step:.3f} | "
                f"val_loss {val_loss:.4f} val_acc {val_acc:.3f} | "
                f"bucket_acc {[f'{x:.3f}' for x in bucket_acc]}\n"
            )

            model.train()

# %%
# Plot training curves
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(losses, label="Train Loss", alpha=0.7)
val_x = [i * log_every for i in range(1, len(val_losses) + 1)]
plt.plot(val_x, val_losses, label="Validation Loss", linewidth=2)
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accs, label="Train Accuracy", alpha=0.7)
plt.plot(val_x, val_accs, label="Validation Accuracy", linewidth=2)
plt.xlabel("Training Steps")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# %%
# Save model
save_dir = "modernbert-diffusion-finetuned"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Model saved to {save_dir}")

# %%
# Test inference with both modes
print("Testing LLaDA-style inference (mask_only=True):")
test_prompt = "What is the capital of France?"
result_llada = iterative_decode(
    model,
    tokenizer,
    test_prompt,
    answer_length=16,
    device=device,
    mask_only=True,  # LLaDA mode
)
print(f"LLaDA result: {result_llada}")

print("\nTesting FlowLM-style inference (mask_only=False):")
result_flowlm = iterative_decode(
    model,
    tokenizer,
    test_prompt,
    answer_length=16,
    device=device,
    mask_only=False,  # FlowLM mode
    confidence_threshold=0.7,
    max_replacements=2,
)
print(f"FlowLM result: {result_flowlm}")

# %%
