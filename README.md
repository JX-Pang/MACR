# MACR: Mechanism-Aware Cognitive Recalibration for Cold-Start Cognitive Diagnosis

Official anonymous implementation of **MACR** (*Mechanism-Aware Cognitive Recalibration for Cold-Start Cognitive Diagnosis*).

---

## 📌 Framework Overview

Under zero-shot exercise cold-start scenarios, target exercises arrive without historical response logs. MACR addresses the **Language-Centric Cognitive Fallacy** via a three-stage behavior-grounded architecture:

1. **Stage 1: Metric-Preserving Student Anchoring**  
   Learns continuous student anchors $\mathbf{A} = [\mathbf{a}_1, \dots, \mathbf{a}_N] \in \mathbb{R}^{N \times d}$ directly from historical response logs, preserving fine-grained continuous proficiencies in an ordinal metric space without coarse textual summarization.
2. **Stage 2: Behavior-Grounded Exercise Recalibration**  
   Projects LLM exercise and knowledge concept semantics into measurement variables $\boldsymbol{\phi}_j = \{\mathbf{r}_j, \mathbf{b}_j, s_j\}$, calibrated via dual-space (measurement and behavior) contrastive objectives ($\mathcal{L}_{\text{meas}}$ and $\mathcal{L}_{\text{beh}}$) over mechanism-equivalent positive pairs and surface-similar hard negatives.
3. **Stage 3: Parallel-Form Reliability Inference**  
   Constructs an aggregation pool $\mathcal{V}_j = \{x_j^{(0)}, x_j^{(1)}, \dots, x_j^{(T)}\}$ containing the original exercise stem and $T$ mechanism-preserving isomorphic variants, performing unsupervised consensus-weighted aggregation to suppress phrasing outliers.

---

## 🛠️ Requirements & Environment Setup

```bash
pip install -r requirements.txt
```

### Core Dependencies
- Python >= 3.9
- PyTorch >= 2.0
- Transformers >= 4.36
- PEFT >= 0.7
- Safetensors
- Scikit-learn, NumPy, PyYAML, tqdm

### Default Models & Hardware
- **Default Backbone LLM**: `meta-llama/Llama-3.2-3B-Instruct` ($d_h = 3072$).  
  If using a local model directory, set the environment variable:
  ```bash
  export MACR_MODEL_PATH="/path/to/local/Llama-3.2-3B-Instruct"
  ```
- **Hardware**: Tested on NVIDIA RTX 3090 Ti GPUs.

---

## 📊 Dataset & Preprocessing

We benchmark on **XES3G5M** and **NIPS34**. The raw data format, filtering criteria, and partitioning follow the **LMCD** baseline ([TAL-auroraX/LMCD](https://github.com/TAL-auroraX/LMCD)):
- Missing KC tags are filtered out.
- For XES3G5M, hierarchical KC trees are flattened into a 1,175-dimensional KC space.
- Sparsity filtering: Students with $< 15$ interaction logs and exercises with $< 10$ responses are removed.
- Mathematical expressions are normalized into standard LaTeX.
- 5-fold exercise cold-start evaluation: 20% held out as target exercises ($\mathcal{E}^t$), remaining 80% source exercises partitioned into 90% training ($\mathcal{E}^s_{\text{train}}$) and 10% validation ($\mathcal{E}^s_{\text{val}}$).

To reproduce preprocessing from raw data:
```bash
python scripts/run_phase0_preprocess.py --raw-dir data/raw/XES3G5M --output-dir data/XES3G5M
```

---

## 🚀 End-to-End Pipeline

### Step 1: Stage 1 Student Anchoring & Semantic Prior Extraction
Extracts LLM semantic embeddings ($h_j$ and $C_j$) and trains the Stage 1 Teacher CDM (student anchors $\mathbf{A}$ and monotonic interaction parameters $\psi$):
```bash
python scripts/run_phase1_anchoring.py --config configs/macr_xes3g5m.yaml --cdm-type ncd
```

### Step 2: Build Contrastive Pairs and Parallel Forms
Constructs mechanism-equivalent positive pairs and surface-similar hard-negative pairs for Stage 2, and generates parallel forms for Stage 3:
```bash
# Build Stage 2 contrastive pairs
python scripts/build_contrastive_pairs.py --config configs/macr_xes3g5m.yaml

# Generate Stage 3 parallel-form variants (Heuristic, Logic Distillation, Isomorphic)
# Note: set ZHIPUAI_API_KEY for online rewriting, or use --allow-fallback for smoke test
python scripts/build_parallel_forms.py --config configs/macr_xes3g5m.yaml --split test
```

### Step 3: Stage 2 Exercise Recalibration Training
Trains the parameter-efficient LoRA adapter and measurement projection heads with response reconstruction and dual contrastive objectives ($\mathcal{L}_{\text{sim}} = \mathcal{L}_{\text{resp}} + \lambda_m \mathcal{L}_{\text{meas}} + \lambda_b \mathcal{L}_{\text{beh}}$):
```bash
python scripts/train_macr.py --config configs/macr_xes3g5m.yaml
```

### Step 4: Evaluation
Evaluate zero-shot diagnostic performance on unseen cold-start target exercises:
```bash
# Standard single-form evaluation
python scripts/evaluate_macr.py --config configs/macr_xes3g5m.yaml --split test

# Parallel-form reliability-weighted evaluation (Stage 3)
python scripts/evaluate_parallel_forms.py --config configs/macr_xes3g5m.yaml --split test
```

Student anchors $\mathbf{A}$ and interaction parameters $\psi$ are frozen after Stage 1 to serve as the fixed behavioral coordinate system.
