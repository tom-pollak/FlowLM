# FlowLM

## Lit Review

### Previous Approach (LLaDA)

Masked diffusion: Only predicts masked [M] positions, unmasked tokens are frozen

```
Current state: [The] [M] [M] [fast] [.]
                            ↑
                    only this gets predicted

Prediction step: [The] [cat] [jumps] [fast] [.]
                                ↑
                        new prediction here

Remasking step: [The] [M] [jumps] [fast] [.]
                       ↑
                 "cat" remasked, but only between steps
```

Problem: "cat" can only change if remasked in separate step

## Method

> Corruption-and-correction training

Any-position diffusion: Can refine ANY token in same prediction step.

```
Input:  [The] [banana] [jumps] [fast] [.]
Model thinks: "wait, 'banana' doesn't fit well with 'jumps fast'"
Output: [The] [dog] [jumps] [fast] [.]
```

- Training: Start with masks or random vocab tokens.
- Inference: Every token evaluated for potential improvement each step

```python
## Training

# Instead of just masking
x_corrupted = randomly_mask(x_clean)

# Also randomly corrupt some unmasked tokens
x_corrupted = randomly_substitute_tokens(x_corrupted, vocab, corruption_rate=0.15)

# Train to predict clean version from corrupted
loss = predict_clean_tokens(model, x_corrupted, x_clean)

## Inference

# Each step: predict improvements for ALL positions
for step in range(num_steps):
    predictions = model.predict_all_positions(current_sequence)
    confidence_scores = model.get_confidence(predictions)

    # Update low-confidence positions
    for i, (pred, conf) in enumerate(zip(predictions, confidence_scores)):
        if conf < threshold:
            current_sequence[i] = pred
```

## Benefits

- Could fix early mistakes based on later context
- Closer to Real Diffusion: Image diffusion refines ALL pixels, not just masked ones
- This would refine ALL tokens based on confidence/fit

## Challenges

If we start with just random vocab tokens, it might be hard to keep the model on the "rails". I guess the "conditioning" would be the original prompt. Would we need both `[MASK]` and random vocab tokens?
