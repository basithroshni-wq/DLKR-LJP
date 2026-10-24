# Graph Report - DLKR-LJP-main  (2026-10-24)

## Corpus Check
- 29 files · ~62,418 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 286 nodes · 449 edges · 19 communities (13 shown, 1 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 7 edges (avg confidence: 0.94)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6250a0bf`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- QwenLegalReasoner
- Any
- train_heads.py
- config.py
- MultiTaskLegalPredictor
- rhetorical_extractor.py
- ⚖️ DLKR-LJP V6.0 Flagship: Dynamic Legal Knowledge Retrieval for Legal Judgment Prediction
- ⚖️ DLKR-LJP V5.1 (Flagship Top Model): Dynamic Legal Knowledge Retrieval for Legal Judgment Prediction
- dlkr_ljp_v5_final.py
- AtomicLegalFactExtractor
- IldcCorpusStreamer
- statutory_verifier.py
- NeuralAttentionResolver
- repackage_all.py

## God Nodes (most connected - your core abstractions)
1. `QwenLegalReasoner` - 16 edges
2. `AtomicCaseFacts` - 15 edges
3. `MultiTaskLegalPredictor` - 12 edges
4. `AtomicLegalFactExtractor` - 10 edges
5. `PretrainedLegalReasoner` - 9 edges
6. `DynamicLegalKnowledgeRetriever` - 9 edges
7. `StatutoryTwoStageVerifier` - 9 edges
8. `IldcCorpusStreamer` - 8 edges
9. `predict_case()` - 8 edges
10. `StatutoryKnowledgeGraph` - 8 edges

## Surprising Connections (you probably didn't know these)
- `DynamicLjpPipeline` --uses--> `StatutoryKnowledgeGraph`  [INFERRED]
  dlkr_ljp_v5_final.py → graph_engine/statutory_graph.py
- `StatutoryTwoStageVerifier` --uses--> `QwenLegalReasoner`  [INFERRED]
  verification/statutory_verifier.py → llm_reasoner.py
- `train_mhpn_heads()` --calls--> `IldcCorpusStreamer`  [EXTRACTED]
  training/train_heads.py → data_pipeline/ildc_streamer.py
- `JudicialConsistencyVerifier` --uses--> `AtomicCaseFacts`  [INFERRED]
  verification/jcv_auditor.py → data_pipeline/rhetorical_extractor.py
- `run_all_cases()` --calls--> `predict_case()`  [EXTRACTED]
  test_25_plus_cases.py → dlkr_ljp_v5_final.py

## Import Cycles
- None detected.

## Communities (19 total, 1 thin omitted)

### Community 0 - "QwenLegalReasoner"
Cohesion: 0.11
Nodes (17): discover_gemini_models(), normalize_penalty_tier(), normalize_statutory_code(), Any, QwenLegalReasoner, ===============================================================================…, Normalizes natural language IPC/BNS strings to canonical DLKR codes., Normalizes natural language penalty descriptions to canonical tiers. (+9 more)

### Community 1 - "Any"
Cohesion: 0.12
Nodes (10): AtomicCaseFacts, DynamicLegalKnowledgeRetriever, JudicialConsistencyVerifier, PrefixKnowledgeFusion, PretrainedLegalReasoner, Any, DLKR-LJP Neural Legal Core: Engine 1: InLegalBERT Neural Transformer Core…, Synthesizes structured judicial reasoning from verified legal facts. (+2 more)

### Community 2 - "train_heads.py"
Cohesion: 0.10
Nodes (17): get_cost_sensitive_criteria(), ===============================================================================…, Returns loss functions with statutory severity weights for Head 1 and Head 2., InLegalBertMeanEncoder, Any, ===============================================================================…, Encodes a single text narrative into a 768-dim tensor vector., Loads InLegalBERT tokenizer and transformer backbone. (+9 more)

### Community 3 - "config.py"
Cohesion: 0.10
Nodes (20): ===============================================================================…, build_statutes_kb(), ===============================================================================…, Generates statutes_40_registry.json and statutes_40_kb.csv., AtomicCaseFacts, Any, find_applicable_precedents(), PrecedentRatio (+12 more)

### Community 4 - "MultiTaskLegalPredictor"
Cohesion: 0.18
Nodes (8): MultiTaskLegalHead, MultiTaskLegalPredictor, Path, Trained Multi-Task Neural Classification Head on top of InLegalBERT (768-dim…, Multi-Head Prediction Network (MHPN) combining: - Trained InLegalBERT Multi-…, Computes attention-weighted mean token pooling over the last hidden state.…, Synthesizes 240+ rich, realistic Indian criminal case scenarios across all…, Fine-tunes multi-task neural heads on 240+ augmented statutory exemplars using…

### Community 5 - "rhetorical_extractor.py"
Cohesion: 0.16
Nodes (13): ===============================================================================…, Extracts pre-trial factual narrative and structured legal facts., Strips judicial holdings, appellate arguments, and sentencing lines to return…, RhetoricalFactExtractor, detect_incident_date(), format_concordance_display(), Any, ===============================================================================… (+5 more)

### Community 6 - "⚖️ DLKR-LJP V6.0 Flagship: Dynamic Legal Knowledge Retrieval for Legal Judgment Prediction"
Cohesion: 0.12
Nodes (16): 1. Neural Legal Reasoner (Multi-Model Cascade), 2. GraphRAG Statutory Knowledge Graph (DAG Engine), 3. Dual-Code Statutory Harmonization (IPC 1860 & BNS 2023), 4. Judicial Consistency Verifier (JCV), 🔬 Abstract & Architecture Overview, 📜 Academic Citation, 🏛️ Core Architectural Pillars, ⚖️ DLKR-LJP V6.0 Flagship: Dynamic Legal Knowledge Retrieval for Legal Judgment Prediction (+8 more)

### Community 7 - "⚖️ DLKR-LJP V5.1 (Flagship Top Model): Dynamic Legal Knowledge Retrieval for Legal Judgment Prediction"
Cohesion: 0.12
Nodes (16): 1. GraphRAG Statutory Knowledge Graph (DAG Engine), 1. Run Full 27-Case Statutory Benchmark Suite, 2. Launch Interactive Gradio Web Dashboard, 2. Universal Qwen-2.5-7B Neural Reasoner & Multi-Backend Architecture, 3. Dual-Code Statutory Harmonization (IPC 1860 & BNS 2023), 3. Google Colab GPU Execution, 4. Interactive Academic Presentation Deck, 4. Judicial Consistency Verifier (JCV) & Precedent Grounding (+8 more)

### Community 8 - "dlkr_ljp_v5_final.py"
Cohesion: 0.13
Nodes (18): build_gradio_dashboard(), DynamicLjpPipeline, get_active_llm_status(), get_pipeline(), _patch_gradio_compatibility(), predict_case(), process_case(), ===============================================================================… (+10 more)

### Community 9 - "AtomicLegalFactExtractor"
Cohesion: 0.18
Nodes (4): AtomicLegalFactExtractor, OpenNyAILegalNER, Open-source Pre-trained Indian Legal NER Engine (OpenNyAI / Supreme Court of…, Generalized Legal Fact Extractor with Scoped Negation Handling, Date Regime…

### Community 10 - "IldcCorpusStreamer"
Cohesion: 0.21
Nodes (8): IldcCorpusStreamer, Any, Path, ===============================================================================…, Streams full judgments from the Exploration-Lab/ILDC dataset on Hugging Face.…, Manages local and remote real-world Indian court judgment corpora., Loads the cached dataset slice, or generates a comprehensive 2,500-case factual…, Generates realistic, varied Indian court judgment factual narratives covering…

### Community 11 - "statutory_verifier.py"
Cohesion: 0.11
Nodes (20): generate_synthetic_dataset(), ===============================================================================…, Generates synthetic FIR narratives across the 40 statutes., run_evaluation(), compute_dense_score(), get_statutory_verifier(), parse_date_from_text(), preprocess_narrative() (+12 more)

### Community 13 - "repackage_all.py"
Cohesion: 0.53
Nodes (5): main(), make_zip(), Path, ===============================================================================…, should_exclude()

## Knowledge Gaps
- **26 isolated node(s):** `Dual-Code Indian Criminal Jurisprudence (IPC 1860 & BNS 2023) with Multi-Model Neural Cascade & GraphRAG`, `🔬 Abstract & Architecture Overview`, `1. Neural Legal Reasoner (Multi-Model Cascade)`, `2. GraphRAG Statutory Knowledge Graph (DAG Engine)`, `3. Dual-Code Statutory Harmonization (IPC 1860 & BNS 2023)` (+21 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 127 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `StatutoryKnowledgeGraph` connect `config.py` to `dlkr_ljp_v5_final.py`, `Any`?**
  _High betweenness centrality (0.137) - this node is a cross-community bridge._
- **Why does `normalize_statutory_code()` connect `QwenLegalReasoner` to `dlkr_ljp_v5_final.py`, `statutory_verifier.py`, `MultiTaskLegalPredictor`?**
  _High betweenness centrality (0.095) - this node is a cross-community bridge._
- **Why does `normalize_penalty_tier()` connect `QwenLegalReasoner` to `dlkr_ljp_v5_final.py`, `statutory_verifier.py`, `MultiTaskLegalPredictor`?**
  _High betweenness centrality (0.095) - this node is a cross-community bridge._
- **What connects `Dual-Code Indian Criminal Jurisprudence (IPC 1860 & BNS 2023) with Multi-Model Neural Cascade & GraphRAG`, `🔬 Abstract & Architecture Overview`, `1. Neural Legal Reasoner (Multi-Model Cascade)` to the rest of the system?**
  _26 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `QwenLegalReasoner` be split into smaller, more focused modules?**
  _Cohesion score 0.11264367816091954 - nodes in this community are weakly interconnected._
- **Should `Any` be split into smaller, more focused modules?**
  _Cohesion score 0.12298387096774194 - nodes in this community are weakly interconnected._
- **Should `train_heads.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09523809523809523 - nodes in this community are weakly interconnected._