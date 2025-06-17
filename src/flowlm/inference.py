import torch
from typing import List
from transformers import PreTrainedTokenizer, PreTrainedModel


def iterative_decode(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    answer_length: int = 32,
    device: str = "cuda",
    mask_only: bool = True,
    confidence_threshold: float = 0.5,
    max_replacements: int = 1,
) -> str:
    """
    Iteratively decode by replacing tokens based on confidence.

    Args:
        mask_only: If True (LLaDA mode), only replace masked positions.
                  If False (FlowLM mode), can replace any position in answer region.
        confidence_threshold: For FlowLM mode, only replace tokens below this confidence.
        max_replacements: Maximum number of tokens to replace per step.
    """
    formatted_prompt = f"User: {prompt} {tokenizer.sep_token} Assistant:"
    prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)

    # Build initial sequence with masks for answer
    ids = (
        [tokenizer.cls_token_id]
        + prompt_ids
        + [tokenizer.sep_token_id]
        + [tokenizer.mask_token_id] * answer_length
        + [tokenizer.sep_token_id]
    )

    # Define answer region boundaries
    answer_start = len([tokenizer.cls_token_id] + prompt_ids + [tokenizer.sep_token_id])
    answer_end = answer_start + answer_length

    num_steps = (
        answer_length if mask_only else answer_length // 2
    )  # FlowLM needs fewer steps

    for step in range(num_steps):
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids]).to(device)).logits

        out_probs = torch.softmax(logits[0], dim=-1)

        if mask_only:
            # LLaDA mode: only consider masked positions
            mask_locs = (torch.tensor(ids) == tokenizer.mask_token_id).nonzero(
                as_tuple=True
            )[0]
            if len(mask_locs) == 0:
                break  # No more masks to replace
            candidate_positions = mask_locs
        else:
            # FlowLM mode: consider all positions in answer region
            candidate_positions = torch.arange(
                answer_start, answer_end, device=out_probs.device
            )

            # Filter by confidence threshold
            confidences = out_probs.max(dim=-1)[0]
            low_conf_mask = confidences < confidence_threshold
            candidate_positions = candidate_positions[
                low_conf_mask[candidate_positions]
            ]

            if len(candidate_positions) == 0:
                break  # All tokens are confident enough

        # Get probabilities for candidate positions
        candidate_probs = out_probs[candidate_positions]
        candidate_max_probs = candidate_probs.max(dim=-1)[0]

        # Select top positions to replace (sorted by confidence)
        num_to_replace = min(max_replacements, len(candidate_positions))
        if mask_only:
            # LLaDA: replace highest confidence mask
            replace_indices = candidate_max_probs.argsort(descending=True)[
                :num_to_replace
            ]
        else:
            # FlowLM: replace lowest confidence tokens
            replace_indices = candidate_max_probs.argsort(descending=False)[
                :num_to_replace
            ]

        # Make replacements
        for idx in replace_indices:
            pos = candidate_positions[idx]
            new_token = candidate_probs[idx].argmax().item()
            ids[pos] = new_token

    return tokenizer.decode(ids)
