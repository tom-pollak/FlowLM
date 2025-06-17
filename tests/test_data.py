"""Unit tests for data processing functions."""

import pytest
from transformers import AutoTokenizer
from flowlm import (
    apply_random_mask,
    format_dialogue,
    BERT_STRATEGY,
    FLOWLM_STRATEGY,
    PURE_MASK,
    PURE_RANDOM,
    MaskingRatio,
)


@pytest.fixture
def tokenizer():
    """Shared tokenizer fixture."""
    return AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")


@pytest.fixture
def test_example():
    """Shared test example fixture."""
    return {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ]
    }


class TestFormatDialogue:
    """Test dialogue formatting function."""

    def test_format_dialogue(self, tokenizer, test_example):
        """Test basic dialogue formatting."""
        result = format_dialogue(test_example["messages"], tokenizer)
        expected = (
            f"User: What is 2+2? {tokenizer.sep_token} Assistant: The answer is 4."
        )
        assert result == expected


class TestApplyRandomMask:
    """Test random masking function."""

    def test_basic_masking(self, tokenizer, test_example):
        """Test basic masking functionality."""
        result = apply_random_mask(
            test_example, tokenizer, BERT_STRATEGY, MaskingRatio(0.5, 0.5), max_len=64
        )

        # Check output structure
        assert "input_ids" in result
        assert "labels" in result
        assert "attention_mask" in result

        # Check that some tokens are labeled for prediction
        labeled_count = sum(1 for x in result["labels"] if x != -100)
        assert labeled_count > 0

    def test_pure_mask_strategy(self, tokenizer, test_example):
        """Test pure masking strategy."""
        result = apply_random_mask(
            test_example, tokenizer, PURE_MASK, MaskingRatio(0.5, 0.5), max_len=64
        )

        # All labeled positions should be [MASK] tokens
        input_ids = result["input_ids"]
        labels = result["labels"]

        for i, label in enumerate(labels):
            if label != -100:  # This position was masked
                assert input_ids[i] == tokenizer.mask_token_id

    def test_pure_random_strategy(self, tokenizer, test_example):
        """Test pure random token strategy."""
        result = apply_random_mask(
            test_example, tokenizer, PURE_RANDOM, MaskingRatio(0.5, 0.5), max_len=64
        )

        # All labeled positions should be random tokens (not masks)
        input_ids = result["input_ids"]
        labels = result["labels"]

        for i, label in enumerate(labels):
            if label != -100:  # This position was corrupted
                assert input_ids[i] != tokenizer.mask_token_id
                assert input_ids[i] != label  # Should be different from original

    def test_different_strategies_different_results(self, tokenizer, test_example):
        """Test that different strategies produce different results."""
        # Run multiple times to ensure strategies behave differently
        differences_found = 0
        n_trials = 10
        n_diffs_threshold = 9

        for seed in range(n_trials):  # Try multiple seeds
            import random

            random.seed(seed)

            bert_result = apply_random_mask(
                test_example,
                tokenizer,
                BERT_STRATEGY,
                MaskingRatio(0.5, 0.5),
                max_len=64,
            )

            random.seed(seed)  # Reset to same seed
            flowlm_result = apply_random_mask(
                test_example,
                tokenizer,
                FLOWLM_STRATEGY,
                MaskingRatio(0.5, 0.5),
                max_len=64,
            )

            if bert_result["input_ids"] != flowlm_result["input_ids"]:
                differences_found += 1

        # Should find differences in at least half the trials
        assert differences_found >= n_diffs_threshold, (
            f"Only found {differences_found} differences in 10 trials"
        )

    def test_masking_ratio_effect(self, tokenizer, test_example):
        """Test that masking ratio affects number of masked tokens."""
        low_ratio_result = apply_random_mask(
            test_example, tokenizer, BERT_STRATEGY, MaskingRatio(0.2, 0.2), max_len=64
        )

        high_ratio_result = apply_random_mask(
            test_example, tokenizer, BERT_STRATEGY, MaskingRatio(0.8, 0.8), max_len=64
        )

        low_labeled = sum(1 for x in low_ratio_result["labels"] if x != -100)
        high_labeled = sum(1 for x in high_ratio_result["labels"] if x != -100)

        # Higher ratio should result in more labeled positions
        assert high_labeled > low_labeled

    def test_only_assistant_tokens_labeled(self, tokenizer, test_example):
        """Test that only assistant response tokens get labeled."""
        result = apply_random_mask(
            test_example, tokenizer, BERT_STRATEGY, MaskingRatio(0.5, 0.5), max_len=64
        )

        # Find the first [SEP] token position
        input_ids = result["input_ids"]
        labels = result["labels"]

        sep_pos = input_ids.index(tokenizer.sep_token_id)

        # All labeled positions should be after the first [SEP]
        for i, label in enumerate(labels):
            if label != -100:
                assert i > sep_pos, (
                    f"Token at position {i} before [SEP] at {sep_pos} was labeled"
                )
