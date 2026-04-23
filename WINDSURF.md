# Windsurf Handoff: TCE-Lite Proof of Concept

This file gives Windsurf the rules, scope, and exact implementation prompt for continuing the TCE repository.

## Project

Repository: `plugmakr/TCE`

Goal: turn the current scaffold into a runnable **TCE-Lite proof-of-concept** that can run on a MacBook Air.

This is not a production ML system yet. It is an experimental prototype to test whether a topology-inspired multimodal architecture can degrade more gracefully than simple baselines under noisy or missing inputs.

---

## Hard Rules

1. **Do not overbuild.**
   - No web app.
   - No database.
   - No API server.
   - No cloud dependencies.
   - No large datasets.
   - No huge frameworks.

2. **Keep it MacBook Air friendly.**
   - Small synthetic dataset.
   - Small neural networks.
   - Fast CPU-compatible training.
   - MPS support is fine, but code must also run on CPU.

3. **Keep dependencies minimal.**
   Use only:
   - torch
   - numpy
   - tqdm
   - scikit-learn
   - matplotlib only if needed for optional plots

4. **Everything must run from the repo root.**

   ```bash
   python experiments/train_synthetic.py
   ```

5. **Do not remove the research framing.**
   The code should include comments explaining how components map to TCE concepts.

6. **Prefer readable code over clever code.**
   This repo is for learning, testing, and external review.

7. **Every experiment should compare against a baseline.**
   TCE-Lite only matters if compared to simple models.

---

## Current Repo Structure

```text
README.md
requirements.txt
src/
  tce_lite.py
  baseline.py
experiments/
  synthetic_dataset.py
  train_synthetic.py
paper/
  optional research paper files may be added later
```

---

## Install

```bash
git clone https://github.com/plugmakr/TCE.git
cd TCE
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python experiments/train_synthetic.py
```

---

## TCE-Lite Architecture Target

Build a minimal architecture with:

1. **Image manifold encoder**
   - Processes image-like numeric vectors.
   - Represents the spatial / HWCM side of the architecture.
   - Does not need full HWCM graph yet; start with a lightweight encoder and comments showing where HWCM logic will later go.

2. **Text manifold encoder**
   - Processes symbolic/text-like numeric vectors.
   - Represents the symbolic manifold side.

3. **SIP fusion layer**
   - Combines manifold-level summaries.
   - Acts as the shared integration point.

4. **Classifier head**
   - Predicts the synthetic class label.

5. **BaselineMLP**
   - Simple concatenation baseline.
   - Same input data, comparable hidden size.

---

## Required Experiment

Update `experiments/train_synthetic.py` so it trains and evaluates:

- BaselineMLP
- TCE-Lite

Evaluate both under:

1. clean input
2. noisy image
3. text dropout
4. missing image
5. missing text

Print a simple results table like:

```text
Model        Clean   Noisy Image   Text Dropout   Missing Image   Missing Text
Baseline     0.98    0.72          0.75           0.45            0.50
TCE-Lite     0.98    0.80          0.82           0.55            0.60
```

The numbers do not need to match this example. The goal is to generate honest measured results.

---

## Exact Prompt for Windsurf

Use the following as the implementation instruction:

```text
You are working inside the GitHub repo plugmakr/TCE.

Goal: turn the current scaffold into a runnable TCE-Lite proof-of-concept.

Do not overbuild. Keep everything small enough to run on a MacBook Air.

Tasks:
1. Review README.md, requirements.txt, src/, and experiments/.
2. Fix any import/path issues.
3. Replace or upgrade src/tce_lite.py with a true TCE-Lite model containing:
   - image manifold encoder
   - text manifold encoder
   - SIP fusion layer
   - classifier head
4. Keep src/baseline.py as the simple baseline model.
5. Update experiments/train_synthetic.py so it trains and evaluates:
   - BaselineMLP
   - TCE-Lite
6. Add corruption tests:
   - clean
   - noisy image
   - text dropout
   - missing image
   - missing text
7. Print a simple results table.
8. Keep dependencies minimal: torch, numpy, tqdm, scikit-learn, matplotlib only if needed.
9. Make sure this runs with:
   python experiments/train_synthetic.py

Important:
- Do not add huge frameworks.
- Do not add real datasets yet.
- Do not create a web app.
- Keep the first experiment small and debuggable.
- Add comments explaining what each part represents in TCE terms.
- Prefer simple, readable code.
```

---

## Success Criteria

A successful Phase 2 commit should allow a user to run:

```bash
python experiments/train_synthetic.py
```

and receive:

- training logs
- clean accuracy
- corruption test accuracy
- comparison between baseline and TCE-Lite

---

## Research Interpretation Rule

Do not claim TCE is proven.

The correct interpretation is:

> If TCE-Lite degrades more gracefully than the baseline under corruption, this provides early evidence that structured manifold-style fusion may improve robustness.

If it does not outperform baseline, report that honestly.
