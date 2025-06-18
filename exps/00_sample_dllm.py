# %%
import os, random, itertools, math, torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "tommyp111/modernbert-flowlm"
tokenizer = AutoTokenizer.from_pretrained(model_id)
SEP_ID, CLS_ID, MASK_ID = (
    tokenizer.sep_token_id,
    tokenizer.cls_token_id,
    tokenizer.mask_token_id,
)
model = AutoModelForMaskedLM.from_pretrained(model_id, device_map=device)
model.eval()

# %%
from flowlm.inference import iterative_decode

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
    mask_only=False,  # FlowLM mode
    confidence_threshold=0.7,
    max_replacements=2,
)
print(f"FlowLM result: {result_flowlm}")
