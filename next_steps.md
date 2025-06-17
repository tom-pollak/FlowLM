# FlowLM Next Steps

## Current Status ✅

- **Infrastructure**: Unified LLaDA/FlowLM codebase in `src/flowlm/`
- **Baseline**: LLaDA replication working (`exps/00_train_mb_dllm.py`)
- **FlowLM Support**: Data corruption (`corruption_rate`) and any-position inference (`mask_only=False`) implemented

## Immediate Next Experiments

### 1. Full FlowLM Training Experiment
**File**: `exps/01_train_flowlm.py`

```python
# Key changes from baseline:
corruption_rate = 0.15  # Add random token substitution
mask_ratio_min = 0.3    # Lower masking ratio 
mask_ratio_max = 0.7    # Balanced with corruption
```

**Questions to answer:**
- Does FlowLM training converge as stably as LLaDA?
- What's the optimal `corruption_rate` vs `mask_ratio` balance?
- How does training loss compare between approaches?

### 2. Confidence Analysis Experiment
**File**: `exps/02_confidence_analysis.py`

**Research goals:**
- Measure correlation between model confidence and actual token quality
- Test different confidence estimation methods (max prob, entropy, etc.)
- Find optimal confidence thresholds for different tasks

**Metrics to track:**
- Confidence-accuracy correlation
- Calibration plots (predicted vs actual accuracy)
- Token-level improvement rates at different confidence levels

### 3. Inference Strategy Comparison
**File**: `exps/03_inference_strategies.py`

**Test variations:**
- Single vs multi-token replacement per step
- Different confidence thresholds (0.3, 0.5, 0.7, 0.9)
- Greedy vs confidence-based replacement ordering
- Fixed vs adaptive number of inference steps

## Key Research Questions

### From `research_idea.md` Challenges:

#### Q1: Keeping Models "On Rails"
> *"If we start with just random vocab tokens, it might be hard to keep the model on the rails"*

**Experiments needed:**
- Compare initialization strategies:
  - Pure masks (current LLaDA)
  - Pure random tokens
  - Hybrid masks + random tokens
  - Prompt-conditioned random sampling

**File**: `exps/04_initialization_strategies.py`

#### Q2: Optimal Corruption Strategy
> *"Would we need both [MASK] and random vocab tokens?"*

**Test matrix:**
```
| Mask Ratio | Corruption Rate | Strategy |
|------------|-----------------|----------|
| 0.8        | 0.0            | LLaDA    |
| 0.5        | 0.2            | Hybrid   |
| 0.3        | 0.4            | FlowLM   |
| 0.0        | 0.8            | Pure Random |
```

#### Q3: Context-Aware Refinement
> *"Could fix early mistakes based on later context"*

**Analysis needed:**
- Track which tokens get refined and why
- Measure improvement quality vs position in sequence  
- Test on tasks requiring global coherence

## Technical Implementation Tasks

### 1. Extend `src/flowlm/evaluation.py`
```python
# Add FlowLM-specific metrics
def refinement_quality_metrics(before_tokens, after_tokens, ground_truth):
    """Measure how FlowLM refinements improve quality"""
    
def confidence_calibration_analysis(confidences, accuracies):
    """Analyze how well confidence predicts accuracy"""
    
def token_change_analysis(initial, refined, steps):
    """Track which tokens change and when"""
```

### 2. Create Analysis Tools
**File**: `src/flowlm/analysis.py`
- Visualization of refinement process
- Confidence score distributions
- Token change heatmaps
- Convergence pattern analysis

### 3. Enhanced Data Processing
**Extensions to `src/flowlm/data.py`:**
- Smart corruption (avoid corrupting key prompt tokens)
- Task-specific corruption strategies
- Dynamic corruption rates based on sequence position

## Evaluation Framework

### Automatic Metrics
1. **Perplexity**: Standard language modeling metric
2. **Refinement Improvement**: Quality delta between initial and final sequences
3. **Convergence Efficiency**: Steps needed to reach stable output
4. **Confidence Calibration**: How well confidence predicts quality

### Human Evaluation
1. **Coherence**: Do FlowLM outputs maintain better global coherence?
2. **Error Correction**: Can humans identify where FlowLM fixes LLaDA mistakes?
3. **Quality Preference**: Side-by-side comparison of LLaDA vs FlowLM outputs

### Diagnostic Tests
**File**: `exps/05_diagnostic_tests.py`
- Inconsistency detection (can FlowLM fix contradictions?)
- Context integration (does later context improve early tokens?)
- Domain transfer (robustness across different text types)

## Experimental Priority

**Week 1-2: Core FlowLM Training**
- [ ] `exps/01_train_flowlm.py` - Full FlowLM training
- [ ] `exps/02_confidence_analysis.py` - Confidence-quality correlation

**Week 3-4: Inference Optimization** 
- [ ] `exps/03_inference_strategies.py` - Strategy comparison
- [ ] `exps/04_initialization_strategies.py` - On-rails investigation

**Week 5-6: Analysis & Evaluation**
- [ ] Implement analysis tools (`src/flowlm/analysis.py`)
- [ ] `exps/05_diagnostic_tests.py` - Comprehensive evaluation
- [ ] Human evaluation setup

## Success Criteria

**FlowLM should demonstrate:**
1. **Training Stability**: Converges as well as LLaDA baseline
2. **Quality Improvement**: Better outputs on coherence/consistency metrics
3. **Efficiency**: Fewer inference steps needed for same quality
4. **Robustness**: Works across different domains and tasks
5. **Interpretability**: Clear understanding of what/when tokens get refined

## Risk Mitigation

**Potential Issues:**
- Training instability with high corruption rates
- Inference getting stuck in local minima
- Overconfident predictions leading to poor refinement decisions

**Monitoring:**
- Loss curves and gradient norms during training
- Convergence patterns during inference
- Confidence calibration metrics

---

*This roadmap provides 6+ weeks of focused research to validate and optimize the FlowLM approach against the LLaDA baseline.*