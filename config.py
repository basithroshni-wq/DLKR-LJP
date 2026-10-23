"""
========================================================================================
DLKR-LJP V5.1 STATE-OF-THE-ART - GLOBAL CONFIGURATION & STATUTORY TAXONOMY
========================================================================================
Comprehensive Dual-Code Statutory Registry for Indian Criminal Jurisprudence
Covering: Indian Penal Code (IPC 1860) & Bharatiya Nyaya Sanhita (BNS 2023)
========================================================================================
"""

import os
import sys
import datetime
from pathlib import Path

# Paths & Directories
PROJECT_DIR = Path(__file__).resolve().parent
RUNTIME_DATA_DIR = PROJECT_DIR / "runtime_data"
CHECKPOINTS_DIR = RUNTIME_DATA_DIR / "checkpoints"
CACHE_DIR = RUNTIME_DATA_DIR / "cache"
CHROMA_DIR = RUNTIME_DATA_DIR / "chroma_legal_kb"

for d in [RUNTIME_DATA_DIR, CHECKPOINTS_DIR, CACHE_DIR, CHROMA_DIR]:
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

# Environment Detection
IS_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
HEAD_WEIGHTS_FILENAME = "legal_bert_heads_v5_1.pt"

# Transformer & Embedding Models
BACKBONE_MODEL_NAME = "law-ai/InLegalBERT"
EMBEDDING_MODEL_NAME = "law-ai/InLegalBERT"
FALLBACK_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INLEGAL_LLAMA_MODEL_NAME = os.environ.get("INLEGAL_LLAMA_MODEL", "sudipto-ducs/InLegalLLaMA-Instruct")
DEFAULT_REASONER_MODEL = os.environ.get("DLKR_REASONER_MODEL", INLEGAL_LLAMA_MODEL_NAME)
FALLBACK_REASONER_MODEL = os.environ.get("FALLBACK_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
ENABLE_4BIT_QUANTIZATION = True

# DLKR Neural Transformer Core Configuration
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
GEMINI_API_KEY_1 = os.environ.get("GEMINI_API_KEY_1", os.environ.get("GEMINI_API_KEY", ""))
GEMINI_API_KEY_2 = os.environ.get("GEMINI_API_KEY_2", "")
GEMINI_API_KEY_3 = os.environ.get("GEMINI_API_KEY_3", "")
GEMINI_API_KEY = GEMINI_API_KEY_1

# Statutory Temporal Cutoff: Bharatiya Nyaya Sanhita (BNS) came into force on July 1, 2024
BNS_EFFECTIVE_DATE = datetime.date(2024, 7, 1)

# Dynamic Relevance & Confidence Gate Hyperparameters
SIMILARITY_THRESHOLD_TAU = 0.38
MIN_CONFIDENCE_FLOOR = 0.30
TOP_K_RETRIEVAL = 6
MAX_GATED_KNOWLEDGE_ITEMS = 3
ENABLE_ADAPTIVE_GATING = True

# Standardized Penalty Tiers
PENALTY_TIERS = [
    "Tier 1: Up to 3 Years (Minor / Summary Offence)",
    "Tier 2: 3 to 7 Years (Arnesh Kumar Threshold / Medium Penalty)",
    "Tier 3: 7 to 10 Years (Serious Offence / Substantial Imprisonment)",
    "Tier 4: Life Imprisonment / Capital Sentence (Heinous Offence)"
]

TIER_LEVEL_ORDINAL = {
    "Tier 1: Up to 3 Years (Minor / Summary Offence)": 1,
    "Tier 2: 3 to 7 Years (Arnesh Kumar Threshold / Medium Penalty)": 2,
    "Tier 3: 7 to 10 Years (Serious Offence / Substantial Imprisonment)": 3,
    "Tier 4: Life Imprisonment / Capital Sentence (Heinous Offence)": 4
}

# Standardized 12 Crime Categories
CRIMINAL_CHARGES = [
    "Theft & Dishonest Misappropriation of Movable Property",
    "Robbery / Extortion with Force",
    "Murder / Intentional Homicide",
    "Culpable Homicide Not Amounting to Murder",
    "Death by Rash and Negligent Act (Vehicular / Industrial)",
    "Attempt to Murder",
    "Voluntarily Causing Grievous Hurt by Dangerous Weapons",
    "Voluntarily Causing Hurt by Dangerous Weapons or Means",
    "Cheating & Dishonestly Inducing Delivery of Property",
    "Forgery of Valuable Security, Will, or Public Record",
    "Dowry Death / Cruelty by In-Laws",
    "Rape / Aggravated Sexual Assault"
]

# Comprehensive 40-Statute Closed Scope Dual-Code Registry (IPC 1860 & BNS 2023)
try:
    _reg_json = PROJECT_DIR / "data_pipeline" / "statutes_40_registry.json"
    if _reg_json.exists():
        import json
        with open(_reg_json, "r", encoding="utf-8") as _f:
            STATUTES_REGISTRY = json.load(_f)
    else:
        from data_pipeline.build_statutes_40 import STATUTES_40 as STATUTES_REGISTRY
except Exception:
    from data_pipeline.build_statutes_40 import STATUTES_40 as STATUTES_REGISTRY

LAW_ARTICLES = list(STATUTES_REGISTRY.keys())

# Cost Weights for Multi-Task Heads
ARTICLE_COST_WEIGHTS = {
    "IPC_302_BNS_103": 8.0,
    "IPC_304_PART1_BNS_105": 6.0,
    "IPC_304_PART2_BNS_105": 6.0,
    "IPC_307_BNS_109": 6.0,
    "IPC_308_BNS_110": 4.0,
    "IPC_326_BNS_118_2": 5.0,
    "IPC_376_BNS_64": 8.0,
    "IPC_376D_BNS_70": 9.0,
    "IPC_304B_BNS_80": 6.0,
    "IPC_392_BNS_309": 5.0,
    "IPC_395_BNS_310": 7.0,
    "IPC_397_BNS_311": 7.0,
    "IPC_467_BNS_338": 5.0,
    "IPC_409_BNS_316_5": 5.0,
    "IPC_304A_BNS_106": 4.0,
    "IPC_420_BNS_318": 3.0,
    "IPC_468_BNS_336": 3.0,
    "IPC_471_BNS_340": 3.0,
    "IPC_379_BNS_303": 2.0,
    "IPC_380_BNS_305": 3.0,
    "IPC_381_BNS_306": 3.0,
    "IPC_384_BNS_308_2": 3.0,
    "IPC_406_BNS_316": 2.5,
    "IPC_411_BNS_317_2": 2.0,
    "IPC_498A_BNS_85": 3.0,
    "IPC_324_BNS_118_1": 2.5,
    "IPC_325_BNS_117_2": 3.5,
    "IPC_323_BNS_115": 1.5,
    "IPC_341_BNS_126_2": 1.0,
    "IPC_342_BNS_127_2": 1.5,
    "IPC_354_BNS_74": 3.5,
    "IPC_354A_BNS_75": 2.0,
    "IPC_354D_BNS_78": 2.5,
    "IPC_506_BNS_351": 2.5,
    "IPC_506_PART1_BNS_351_2": 2.0,
    "IPC_506_PART2_BNS_351_3": 3.5,
    "IPC_509_BNS_79": 2.0,
    "IPC_147_BNS_191_2": 2.0,
    "IPC_148_BNS_191_3": 3.0,
    "IPC_160_BNS_194": 1.0,
    "NONE_CIVIL_DISPUTE": 1.0,
    "NONE_OUT_OF_DOMAIN": 1.0
}
for art in STATUTES_REGISTRY:
    if art not in ARTICLE_COST_WEIGHTS:
        ARTICLE_COST_WEIGHTS[art] = 2.0
