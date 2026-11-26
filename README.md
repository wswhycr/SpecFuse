# SpecFuse: Graph–Spectral Cross-Attention Fusion for Ordered Strain-Pair Antigenic Distance Prediction

## Overview

SpecFuse predicts the antigenic distance between an ordered pair of influenza
strains — a source (antiserum strain) and a reference (virus strain) — from
haemagglutination-inhibition (HI) measurements. Each strain is represented by
two graph tokens, taken from a strain-graph encoder, and four spectral tokens,
obtained by a band-wise DCT of the Fourier spectrum of the same features. The
two token types are fused by directional cross-attention, in which source
tokens query reference tokens, so that the source–reference orientation implied
by the SLT target is preserved inside the fusion stage rather than only in the
input format.

## Model Architecture

![Model Architecture](model.png)

*Figure 1: SpecFuse pipeline. The gated local–Fourier adapter refines the raw
per-strain features, the strain-graph encoder produces the graph embeddings,
and the directional cross-attention head fuses two graph tokens with four
spectral tokens.*

## Repository contents

```
SpecFuse/
├── graphData.py      strain-graph loader (xlsx + ProtVec)
├── adapter.py        gated local–Fourier adapter
├── spectral.py       spectral token construction + dataset assembly
├── model.py          SpecFuse head and ablation variants
├── train.py          training entry point
├── eval.py           evaluation entry point (deterministic or MC-dropout)
├── plm_baseline.py   protein-language-model baselines
├── model.png         architecture figure
├── data/             raw SLT tables and ProtVec embeddings
├── splits/           fixed inner-train / inner-val / test index sets
├── splits_asym/      pair splits used by the PLM comparison
├── embeddings/       frozen PLM embeddings (5 models x 3 datasets)
├── frozen/           frozen adapter and graph-embedding artefacts
└── MANIFEST.csv / DATA_PROVENANCE.md
```

## Environment

```
Python            3.10.12
PyTorch           2.14.1+cu130
NumPy             2.2.6
pandas            2.x
scikit-learn      1.x
torch-geometric   2.x
CUDA              13.0
GPU               NVIDIA RTX 4090 (24 GB)
```

## Installation

```bash
git clone <repository-url>
cd SpecFuse
pip install -r requirements.txt
```
