import random
from dataclasses import dataclass
from typing import Dict, List, Any
from transformers.tokenization_utils import PreTrainedTokenizer


@dataclass(frozen=True)
class MaskingStrategy:
    mask_prob: float  # Probability of [MASK] token
    random_prob: float  # Probability of random token
    unchanged_prob: float  # Probability of keeping original

    def __post_init__(self):
        # Validate probabilities sum to 1.0
        total = self.mask_prob + self.random_prob + self.unchanged_prob
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Probabilities must sum to 1.0, got {total}")


@dataclass(frozen=True)
class MaskingRatio:
    min_ratio: float
    max_ratio: float

    def __post_init__(self):
        if not (0.0 <= self.min_ratio <= self.max_ratio <= 1.0):
            raise ValueError(f"Invalid ratio range: {self.min_ratio}-{self.max_ratio}")


# Common strategies
BERT_STRATEGY = MaskingStrategy(mask_prob=0.8, random_prob=0.1, unchanged_prob=0.1)
LLADA_STRATEGY = MaskingStrategy(mask_prob=0.9, random_prob=0.1, unchanged_prob=0.0)
FLOWLM_STRATEGY = MaskingStrategy(mask_prob=0.3, random_prob=0.7, unchanged_prob=0.0)
PURE_MASK = MaskingStrategy(mask_prob=1.0, random_prob=0.0, unchanged_prob=0.0)
PURE_RANDOM = MaskingStrategy(mask_prob=0.0, random_prob=1.0, unchanged_prob=0.0)


def format_dialogue(
    messages: List[Dict[str, str]], tokenizer: PreTrainedTokenizer
) -> str:
    """
    Convert dialogue messages to flat string with explicit [SEP] boundaries.
    Expects first msg=user, second msg=assistant.
    """
    user_msg = messages[0]["content"].strip()
    assistant_msg = messages[1]["content"].strip()
    return f"User: {user_msg} {tokenizer.sep_token} Assistant: {assistant_msg}"


def apply_random_mask(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    strategy: MaskingStrategy,
    ratio: MaskingRatio = MaskingRatio(min_ratio=0.15, max_ratio=0.99),
    max_len: int = 512,
):
    """
    Apply random masking to dialogue data with variable mask ratio.
    Only processes tokens in the assistant response region.

    Args:
        strategy: MaskingStrategy defining how to replace selected tokens
        ratio: MaskingRatio defining what fraction of tokens to select
        max_len: Maximum sequence length
    """
    text = format_dialogue(example["messages"], tokenizer)
    enc = tokenizer(text, truncation=True, max_length=max_len, padding="max_length")
    ids: list[int] = enc["input_ids"]
    labels = [-100] * len(ids)  # -100 -> ignored by CE-loss

    # Find assistant region (everything after first [SEP])
    if tokenizer.sep_token_id not in ids:
        return {**enc, "labels": labels}

    assert isinstance(tokenizer.mask_token_id, int)
    assert isinstance(tokenizer.sep_token_id, int)

    sep_pos = ids.index(tokenizer.sep_token_id)
    candidates = [
        i
        for i in range(sep_pos + 1, len(ids))
        if ids[i]
        not in (tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id)
    ]

    if not candidates:
        return {**enc, "labels": labels}

    # Variable mask ratio
    mask_ratio = random.uniform(ratio.min_ratio, ratio.max_ratio)
    n_mask = max(1, int(len(candidates) * mask_ratio))
    chosen = random.sample(candidates, n_mask)

    for idx in chosen:
        labels[idx] = ids[idx]  # Remember ground-truth
        dice = random.random()
        if dice < strategy.mask_prob:
            ids[idx] = tokenizer.mask_token_id
        elif dice < strategy.mask_prob + strategy.random_prob:
            ids[idx] = random.randint(0, tokenizer.vocab_size - 1)
        # else: keep unchanged (strategy.unchanged_prob)

    enc["input_ids"] = ids
    enc["labels"] = labels
    return enc
