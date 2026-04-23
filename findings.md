# TCE-Lite Experimental Findings

## Experiment Overview

This document reports the initial experimental findings from the TCE-Lite proof-of-concept implementation. The experiment compares TCE-Lite against a simple baseline MLP on a synthetic multimodal classification task under various corruption conditions.

## Experimental Setup

### Architecture (Initial)

**TCE-Lite v1:**
- Image manifold encoder: 2-layer MLP (64→32→16)
- Text manifold encoder: 2-layer MLP (32→16→8)
- SIP fusion layer: Concatenation + linear (24→16)
- Classifier head: Linear (16→10)
- Parameters: ~3.8K

**BaselineMLP:**
- Simple concatenation of image and text inputs (96-dim)
- 2 hidden layers (96→64→64→10)
- Parameters: ~11K

### Architecture (Improved - v2)

**TCE-Lite v2:**
- Image manifold encoder: 2-layer MLP (64→56→40)
- Text manifold encoder: 2-layer MLP (32→28→20)
- SIP fusion layer: GraphFusionSIP with cross-modal attention
  - Cross-modal attention (image↔text)
  - Self-attention for each modality
  - Fusion MLP with learned gating (48-dim)
- Classifier head: Linear (48→10)
- Parameters: ~11K (matched to baseline)

**BaselineMLP:** (unchanged)
- Parameters: ~11K

### Dataset (Initial v1)
- SyntheticMultimodalDataset: 5,000 samples
- Image dimension: 64
- Text dimension: 32
- Classes: 10
- Label encoding: Both modalities contained full label information (redundant)
- Train/validation split: 80/20

### Dataset (Improved v2)
- Same base structure but redesigned label encoding
- Label split design: Neither modality alone can determine class
  - Image encodes coarse class: label // 2 (0-4, 5 groups)
  - Text encodes fine class: label % 2 (0-1, 2 groups)
  - Combined: 5 × 2 = 10 final classes
- Added interaction patterns between modalities
- Now requires BOTH modalities for correct classification
- Train/validation split: 80/20

### Training
- Optimizer: Adam (lr=1e-3)
- Epochs: 20
- Batch size: 64
- Device: CPU

### Corruption Conditions
1. **Clean**: No corruption
2. **Noisy Image**: Add Gaussian noise (σ=0.5) to image modality
3. **Text Dropout**: Randomly zero out 50% of text features
4. **Missing Image**: Zero out entire image modality
5. **Missing Text**: Zero out entire text modality

## Results

### Initial Results (v1)

| Model        | Clean   | Noisy Img | Txt Drop | Miss Img | Miss Txt |
|--------------|---------|-----------|----------|----------|----------|
| Baseline     | 1.0000  | 0.9670    | 1.0000   | 1.0000   | 1.0000   |
| TCE-Lite v1  | 1.0000  | 0.5980    | 0.9950   | 0.5010   | 0.9840   |

*Note: Baseline maintained 100% on missing modalities because either modality alone was sufficient to classify.*

### Improved Results (v2)

| Model        | Clean   | Noisy Img | Txt Drop | Miss Img | Miss Txt |
|--------------|---------|-----------|----------|----------|----------|
| Baseline     | 1.0000  | 0.7740    | 0.8990   | 0.2910   | 0.5950   |
| TCE-Lite v2  | 1.0000  | 0.5750    | 0.8530   | 0.2190   | 0.5090   |

*Dataset now requires both modalities - baseline drops significantly on missing data as expected.*

## Analysis

### Key Findings (v2 - Fair Comparison)

1. **Clean Performance**: Both models achieve perfect accuracy (100%) on clean validation data with matched parameter counts (~11K).

2. **Dataset Requires Both Modalities**: The new split-label design successfully creates a task where:
   - Baseline drops to 29.1% on missing image (from 100% in v1)
   - Baseline drops to 59.5% on missing text (from 100% in v1)
   - This confirms the task now genuinely requires multimodal fusion

3. **Robustness Comparison**: 
   - BaselineMLP still outperforms TCE-Lite v2 under all corruption conditions
   - TCE-Lite v2 shows similar degradation patterns to v1
   - Graph-based attention fusion did not improve robustness over simple concatenation

### Interpretation

**This remains a negative result** even with improved experimental design:
- Parameter counts matched (~11K each)
- Dataset requires genuine multimodal fusion
- Graph-based SIP fusion with cross-modal attention

Despite these improvements, TCE-Lite v2 does **not** outperform the simple concatenation baseline. The baseline's early fusion approach remains more robust to corruption.

### Possible Explanations (Updated for v2)

1. **Over-separation persists**: Even with cross-modal attention, separate encoding before fusion may create representations that are too modality-specific to compensate effectively when one input is corrupted.

2. **Attention mechanism limitations**: The graph attention SIP may not be learning effective cross-modal mappings, or the attention weights may become unstable under corruption.

3. **Synthetic task structure**: The categorical split design (coarse+fine) may favor early concatenation approaches that can more easily learn the modulo arithmetic required.

4. **Training dynamics**: The baseline's single-stream architecture may simply optimize better for this task, or the corruption patterns may not be what the TCE architecture was hypothesized to handle.

## Next Steps

### Immediate
- Report these negative results honestly
- Consider architectural modifications to improve robustness

### Potential Improvements (v3 Ideas)
1. **Skip connections**: Add residual connections between encoders and fusion layer
2. **Corruption-aware training**: Train with random corruption augmentation
3. **Different fusion strategies**: Try weighted averaging, bilinear pooling, or transformer-style fusion
4. **Modality-specific dropout**: Regularize by randomly dropping modalities during training
5. **Larger fusion dimension**: Increase the SIP representation capacity
6. **Alternative graph structures**: Try GNN-style message passing instead of attention

### Research Implications

These findings suggest that the naive implementation of separate manifold encoders with simple fusion does not automatically confer robustness advantages. The full TCE architecture with Hierarchical Waffle Cubical Manifold (HWCM) structures and more sophisticated transition mappings may be necessary to realize the hypothesized benefits.

## Conclusion

The TCE-Lite prototype successfully runs on a MacBook Air and provides a baseline for experimentation. 

**Version 1** showed negative results but had methodological issues:
- TCE-Lite had fewer parameters than baseline (3.8K vs 11K)
- Dataset allowed single-modality classification (neither required both)

**Version 2** addressed these issues:
- Matched parameter counts (~11K each)
- Redesigned dataset requiring genuine multimodal fusion
- Implemented GraphFusionSIP with cross-modal attention

**Result**: TCE-Lite v2 still does not outperform the simple concatenation baseline under corruption.

This suggests that the hypothesized robustness benefits of manifold-style architectures may require:
- More sophisticated fusion mechanisms beyond attention
- Different task structures that better exploit the architecture
- The full HWCM graph structures envisioned in the research paper
- Or the hypothesis itself may need refinement

The negative results are being reported honestly as part of the research process.
