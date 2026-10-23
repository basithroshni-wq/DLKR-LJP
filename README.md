# ⚖️ DLKR-LJP V6.0 Flagship: Dynamic Legal Knowledge Retrieval for Legal Judgment Prediction
### Dual-Code Indian Criminal Jurisprudence (IPC 1860 & BNS 2023) with Multi-Model Neural Cascade & GraphRAG

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/basithroshni-wq/DLKR-LJP/blob/main/DLKR_LJP_Colab_Runner_V6.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)

---

## 🔬 Abstract & Architecture Overview

**DLKR-LJP V6.0** is an academic flagship neuro-symbolic framework for Legal Judgment Prediction (LJP) tailored for the Indian legal system. It bridges the statutory gap between the historical **Indian Penal Code (IPC 1860)** and the newly enacted **Bharatiya Nyaya Sanhita (BNS 2023)**.

Traditional legal NLP models suffer from token representation collapse (single `[CLS]` bottleneck) and fatal undercharging errors. DLKR-LJP resolves this through a **Multi-Model Neural Reasoning Cascade** combined with a **GraphRAG Statutory Knowledge DAG**, domain-native **InLegalBERT Mean Pooling**, **Cost-Sensitive Asymmetric Loss**, and a **Judicial Consistency Verifier (JCV)**.

```
                              [ Legal Incident Narrative / FIR Text ]
                                                 │
                                                 ▼
                             [ Neural Legal Reasoner Cascade ]
                    (Nemotron 30B / 550B MoE Reasoning + Statutory Normalizer)
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
      [ GraphRAG Statutory Knowledge DAG ]              [ InLegalBERT Mean-Pooling Encoder ]
     (s.300 Exceptions 1-5, Supreme Court Ratios:          (Attention-weighted across 512 tokens)
      Pulicherla, Virsa Singh, Ghapoo Yadav)                              │
                        │                                                 ▼
                        │                              [ Relevance & Confidence Gate (RCG) ]
                        │                              (Prunes irrelevant statutory noise)
                        │                                                 │
                        │                                                 ▼
                        │                               [ Prefix Knowledge Fusion Layer ]
                        │                                                 │
                        │                                                 ▼
                        │                              [ Multi-Task Prediction Heads (MHPN) ]
                        │                               - Head 1: Law Articles (IPC / BNS)
                        │                               - Head 2: Crime Charge Categorization
                        │                               - Head 3: Bail Qualification
                        │                               - Head 4: Penalty Severity Tiers
                        │                                                 │
                        └────────────────────────┬────────────────────────┘
                                                 ▼
                              [ Judicial Consistency Verifier (JCV) ]
                             - Enforces statutory penalty ceilings (e.g. 10 yrs)
                             - Procedural floor limits (Arnesh Kumar 7-yr rule)
                             - Quashes civil commercial contract disputes (s.482 / s.528)
                                                 │
                                                 ▼
                          [ 7-Point Traceable Evidence & Adjudication ]
```

---

## 🏛️ Core Architectural Pillars

### 1. Neural Legal Reasoner (Multi-Model Cascade)
- **Primary Reasoner**: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` (1–2 sec sub-second neural inference).
- **Flagship Cascade**: `nvidia/nemotron-3-ultra-550b-a55b:free` (550B MoE reasoning for complex edge cases).
- **Statutory Normalizer**: Automatically reconciles natural language outputs into canonical statutory codes (`IPC_304_PART2_BNS_105`, `IPC_302_BNS_103`, etc.).
- **Deterministic Offline Prober**: Standalone fallback guaranteeing execution even when offline.

### 2. GraphRAG Statutory Knowledge Graph (DAG Engine)
- **Topological DAG Traversal**: Evaluates statutory dependency pathways (e.g., `IPC 300` -> `Exception 4` -> `IPC 304 Part II`).
- **Supreme Court Landmark Precedents**:
  - *Pulicherla Surakarayanam v. State of A.P.* (2006): Intention gathered from weapon spontaneity and strike location.
  - *Virsa Singh v. State of Punjab* (1958): Objective bodily injury standard for murder.
  - *Ghapoo Yadav v. State of M.P.* (2003): Four conjunctive criteria for Section 300 Exception 4.
  - *Arnesh Kumar v. State of Bihar* (2014): Mandatory arrest safeguards for offences with penalty $\le$ 7 years.
  - *State of Haryana v. Bhajan Lal* (1992): Quashing purely civil complaints u/s 482 CrPC / s.528 BNSS.

### 3. Dual-Code Statutory Harmonization (IPC 1860 & BNS 2023)
- Features automated temporal regime detection routing incidents occurring prior to July 1, 2024 to the Indian Penal Code (IPC 1860) and post-July 1, 2024 to Bharatiya Nyaya Sanhita (BNS 2023).

### 4. Judicial Consistency Verifier (JCV)
- Strict symbolic auditor that audits predictions against statutory maximum penalties and legal threshold criteria, preventing sentencing overreach or illegal charge combinations.

---

## ⚖️ Statutory Penalty Severity Tiers

| Severity Bracket | Statutory Ceiling | Example Offences | Procedural Safeguard |
| :--- | :--- | :--- | :--- |
| **Tier 1** | Up to 3 Years | Simple Hurt (IPC 323), Simple Theft (IPC 379), Negligent Driving (IPC 304A) | Summary trial eligible |
| **Tier 2** | 3 to 7 Years | Cheating (IPC 420), Criminal Breach of Trust (IPC 406), Forgery (IPC 468) | *Arnesh Kumar* notice u/s 41A CrPC / 35(3) BNSS |
| **Tier 3** | 7 to 10 Years | Culpable Homicide Part II (IPC 304), Robbery (IPC 392), Dowry Death (IPC 304B) | Sessions Court trial |
| **Tier 4** | Life Imprisonment / Death | Premeditated Murder (IPC 302), Rape (IPC 376), Forgery of Will (IPC 467) | Non-bailable, Special Court trial |
| **Quashed** | No Penal Term | Pure commercial contract dispute without deception | Quashed u/s 482 CrPC / s.528 BNSS (*Bhajan Lal*) |

---

## 🚀 Quickstart Guide

### Option A: Google Colab (1-Click Run)
Open the official notebook [`DLKR_LJP_Colab_Runner_V6.ipynb`](DLKR_LJP_Colab_Runner_V6.ipynb) in Colab:
```python
# 1. Clone repository from GitHub
!git clone https://github.com/basithroshni-wq/DLKR-LJP.git
%cd DLKR-LJP

# 2. Install dependencies
!pip install -q transformers sentence-transformers chromadb gradio accelerate bitsandbytes networkx

# 3. Launch Gradio Dashboard
!python dlkr_ljp_v5_final.py --share
```

### Option B: Kaggle Notebook Execution
Open or import [`DLKR_LJP_Kaggle_Runner.ipynb`](DLKR_LJP_Kaggle_Runner.ipynb) into Kaggle:
```python
# 1. Clone repository into Kaggle working directory
!git clone https://github.com/basithroshni-wq/DLKR-LJP.git /kaggle/working/DLKR-LJP
%cd /kaggle/working/DLKR-LJP

# 2. Install lightweight dependencies
!pip install -q transformers sentence-transformers gradio networkx

# 3. Adjudicate any narrative or launch Gradio inline
from dlkr_ljp_v5_final import predict_case, SAMPLE_KAKKANAD_EXCEPTION4
result = predict_case(SAMPLE_KAKKANAD_EXCEPTION4)
print(result['predictions']['head1_articles'][0]['display_name'])
```

### Option C: Local Python Execution
```bash
git clone https://github.com/basithroshni-wq/DLKR-LJP.git
cd DLKR-LJP

pip install -r requirements.txt

# Run interactive web UI
python dlkr_ljp_v5_final.py --share
```

### Python API Usage
```python
from dlkr_ljp_v5_final import get_pipeline

pipeline = get_pipeline()

case_narrative = (
    "During a sudden quarrel at a tea stall over water, the accused stabbed "
    "the victim once with a knife in heat of passion. The victim died. No premeditation."
)

result = pipeline.run(case_narrative)
preds = result["predictions"]

print(f"Primary Law Article : {preds['head1_articles'][0]['display_name']}")
print(f"Criminal Charge     : {preds['head2_charges'][0]['charge_name']}")
print(f"Penalty Bracket     : {preds['primary_penalty']['tier']}")
print(f"Consistency Score   : {preds.get('judicial_consistency_index', 100)}%")
```

---

## 📁 Repository Structure

```
DLKR-LJP/
├── DLKR_LJP_Colab_Runner_V6.ipynb     # 1-Click Google Colab Runner
├── dlkr_ljp_v5_final.py               # Main Unified Pipeline & Gradio UI
├── llm_reasoner.py                    # Multi-Model Neural Cascade Engine
├── neural_attention_kernel.py         # Latent Attention Dispatcher Bridge
├── config.py                          # Hyperparameters & Statutory Mappings
├── requirements.txt                   # Project Dependencies
├── data_pipeline/
│   ├── ildc_streamer.py               # ILDC 35k Judgments Streamer
│   ├── rhetorical_extractor.py        # Pre-trial Fact Extraction
│   └── temporal_regime.py             # IPC 1860 vs BNS 2023 Detection
├── graph_engine/
│   ├── statutory_graph.py             # Section 300 DAG Traversal
│   └── precedent_ratios.py            # Supreme Court Case Ratios
├── models/
│   ├── inlegalbert_mean_encoder.py    # Mean-Pooling Encoder
│   ├── multi_task_heads.py            # Multi-Task Prediction Heads
│   └── cost_sensitive_loss.py         # Asymmetric Loss Function
├── training/
│   └── train_heads.py                 # Fast Head Fine-Tuning
├── verification/
│   └── jcv_auditor.py                 # Judicial Consistency Auditor
└── presentation_slides.html           # Project Presentation Deck
```

---

## 📜 Academic Citation

```bibtex
@misc{dlkrljp2026,
  title={DLKR-LJP V6: GraphRAG-Augmented Dual-Code Jurisprudence and Multi-Model Reasoning for Legal Judgment Prediction},
  author={DLKR-LJP Research Team},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/basithroshni-wq/DLKR-LJP}}
}
```
