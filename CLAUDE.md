# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlowLM is a research project exploring a novel approach to diffusion-based language models using "corruption-and-correction" training. The goal is to enable models to refine ANY token position during inference, not just masked positions like traditional approaches.

**Primary Research Plan**: See `research_idea.md` for the complete theoretical foundation and methodology for FlowLM's "any-position diffusion" approach.

**Reference Implementation**: The `mb_dllm/` directory contains a ModernBERT-based implementation that follows LLaDA-style training. This serves as inspiration and reference for understanding how diffusion-style language model training works, but is NOT the exact FlowLM approach described in the research plan.

## Key Concepts from research_idea.md

### FlowLM vs LLaDA

- **LLaDA limitation**: Only masked [M] positions get predicted; other tokens can only change if remasked in separate steps
- **FlowLM innovation**: Any token can be refined in the same prediction step based on confidence scores
- **Training approach**: Corruption-and-correction using both masking AND random token substitution
- **Inference**: All positions evaluated for potential improvement each step, with low-confidence tokens replaced

### Architecture Goals

- Train model to predict clean tokens from corrupted input (masks + random substitutions)
- During inference, iteratively refine all positions based on confidence/fit
- More similar to image diffusion (refines ALL pixels) than traditional masked LM

## Codebase Structure

### Core Files

- `research_idea.md`: **Primary reference** - contains the complete FlowLM methodology, benefits, and challenges
- `src/flowlm/`: **Main implementation directory** - contains all reusable FlowLM components and modules
- `exps/`: **Experiments directory** - contains experimental code and interactive Python files (using # %% cells, NOT .ipynb notebooks)
- `mb_dllm/`: Reference implementation using ModernBERT with LLaDA-style training
  - `MB_dLLM_sample.ipynb`: Demonstrates iterative inference with confidence-based token replacement
  - `mb_llm_train.ipynb`: Shows variable masking ratio training on instruction data
  - `README.md`: Notes that this is adjacent research, not the exact FlowLM approach
- `papers/`: Reference papers (LLaDA, ModernBERT)

### Development Guidelines

- **Reusable code**: Place all reusable components, models, and utilities in `src/flowlm/`
- **Experiments**: Use `exps/` for experimental code and prototyping
- **Interactive development**: Use # %% cell markers in .py files for notebook-like interactive development (DO NOT create .ipynb files)
- **Logging**: Use wandb for experiment tracking and metrics logging
- **Debugging**: Print statements should only be used for debugging purposes; use proper logging otherwise

### Implementation Notes

The `mb_dllm/` implementation uses:

- ModernBERT as base model with MLM head
- Variable masking ratios (15%-99%) on assistant responses
- Standard masked language modeling loss
- Iterative inference replacing highest-confidence masked positions

This differs from the full FlowLM vision which would:

- Train with both masks AND random token corruption
- Refine any position (not just masked) during inference
- Use confidence scores to determine which tokens to update
