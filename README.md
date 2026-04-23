# Topological Cognitive Engine (TCE)

**TCE** is an experimental research project exploring multi-manifold neural architectures for robust multimodal reasoning.

This repository contains:

- A research-paper draft for the **Topological Cognitive Engine**
- The upgraded **Hierarchical Waffle Cubical Manifold (HWCM)** design
- A lightweight **TCE-Lite** prototype scaffold intended to run on a normal laptop, including a MacBook Air

## Core Idea

Most deep learning architectures assume one dominant geometry: token sequences, grids, or fixed graphs. TCE explores the idea of using multiple graph-based manifolds, each with different inductive biases, connected by smooth transition mappings and shared integration points.

## Current Status

This is early research / prototype work.

The paper is conceptual and architectural. The code is a starting point for experiments, not proof of performance yet.

## Quick Start

```bash
git clone https://github.com/plugmakr/TCE.git
cd TCE
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python experiments/train_synthetic.py
```

On Apple Silicon, PyTorch may use MPS automatically where available.

## Repository Layout

```text
paper/
  TCE_research_paper.md
  main.tex
  bibliography.bib
src/
  tce_lite.py
experiments/
  train_synthetic.py
requirements.txt
```

## First Experimental Goal

Build and test **TCE-Lite** against simple baselines on a small synthetic multimodal task.

The first useful result would be evidence that TCE-Lite degrades more gracefully than simple concatenation under noisy or missing modality conditions.

## License

License not selected yet.
