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

# Comprehensive Statutory Articles Registry
STATUTES_REGISTRY = {
    "IPC_379_BNS_303": {
        "article_code": "IPC_379_BNS_303",
        "ipc_section": "Section 379 IPC",
        "bns_section": "Section 303(2) BNS",
        "title": "Punishment for Theft",
        "charge": "Theft & Dishonest Misappropriation of Movable Property",
        "penalty_tier": "Tier 1: Up to 3 Years (Minor / Summary Offence)",
        "max_imprisonment_years": 3,
        "is_cognizable": True,
        "is_bailable": False,
        "essential_elements": [
            "Dishonest intention to take property",
            "Movable property taken out of possession of another",
            "Taking without consent",
            "Moving the property in order to such taking"
        ],
        "statutory_exceptions": []
    },
    "IPC_392_BNS_309": {
        "article_code": "IPC_392_BNS_309",
        "ipc_section": "Section 392 IPC",
        "bns_section": "Section 309(4) BNS",
        "title": "Punishment for Robbery",
        "charge": "Robbery / Extortion with Force",
        "penalty_tier": "Tier 3: 7 to 10 Years (Serious Offence / Substantial Imprisonment)",
        "max_imprisonment_years": 10,
        "is_cognizable": True,
        "is_bailable": False,
        "essential_elements": [
            "Commission of theft or extortion",
            "Causing or attempting to cause death, hurt, or wrongful restraint",
            "Fear of instant death, hurt, or restraint intentionally caused",
            "Act done for end of committing theft or carrying away property"
        ],
        "statutory_exceptions": []
    },
    "IPC_302_BNS_103": {
        "article_code": "IPC_302_BNS_103",
        "ipc_section": "Section 302 IPC",
        "bns_section": "Section 103(1) BNS",
        "title": "Punishment for Murder",
        "charge": "Murder / Intentional Homicide",
        "penalty_tier": "Tier 4: Life Imprisonment / Capital Sentence (Heinous Offence)",
        "max_imprisonment_years": 99,  # Capital / Life
        "is_cognizable": True,
        "is_bailable": False,
        "essential_elements": [
            "Intention of causing death",
            "Intention of causing bodily injury sufficient in ordinary course of nature to cause death",
            "Knowledge that act is so imminently dangerous that it must in all probability cause death",
            "Absence of any of the five exceptions to Section 300"
        ],
        "statutory_exceptions": ["Exception 1", "Exception 2", "Exception 3", "Exception 4", "Exception 5"]
    },
    "IPC_304_PART1_BNS_105": {
        "article_code": "IPC_304_PART1_BNS_105",
        "ipc_section": "Section 304 Part I IPC",
        "bns_section": "Section 105 BNS",
        "title": "Culpable Homicide Not Amounting to Murder (With Intention to Cause Bodily Injury)",
        "charge": "Culpable Homicide Not Amounting to Murder",
        "penalty_tier": "Tier 4: Life Imprisonment / Capital Sentence (Heinous Offence)",
        "max_imprisonment_years": 20,
        "is_cognizable": True,
        "is_bailable": False,
        "essential_elements": [
            "Death caused by act with intention of causing death or injury likely to cause death",
            "Mitigated by one of the exceptions to Section 300 (e.g. grave provocation or sudden quarrel)"
        ],
        "statutory_exceptions": []
    },
    "IPC_304_PART2_BNS_105": {
        "article_code": "IPC_304_PART2_BNS_105",
        "ipc_section": "Section 304 Part II IPC",
        "bns_section": "Section 105 BNS",
        "title": "Culpable Homicide Not Amounting to Murder (With Knowledge but Without Intention)",
        "charge": "Culpable Homicide Not Amounting to Murder",
        "penalty_tier": "Tier 3: 7 to 10 Years (Serious Offence / Substantial Imprisonment)",
        "max_imprisonment_years": 10,
        "is_cognizable": True,
        "is_bailable": False,
        "essential_elements": [
            "Act done with knowledge that it is likely to cause death",
            "Without intention to cause death or cause bodily injury likely to cause death",
            "Occurred in heat of passion, sudden quarrel, or mutual fight without premeditation"
        ],
        "statutory_exceptions": []
    },
    "IPC_304A_BNS_106": {
        "article_code": "IPC_304A_BNS_106",
        "ipc_section": "Section 304A IPC",
        "bns_section": "Section 106(1) BNS",
        "title": "Causing Death by Negligence (Rash Driving / Negligent Act)",
        "charge": "Death by Rash and Negligent Act (Vehicular / Industrial)",
        "penalty_tier": "Tier 1: Up to 3 Years (Minor / Summary Offence)",
        "max_imprisonment_years": 2,  # BNS 106(1) provides up to 5 years, IPC 304A up to 2 years
        "is_cognizable": True,
        "is_bailable": True,
        "essential_elements": [
            "Death of any person",
            "Act done by accused rashly or negligently",
            "Act not amounting to culpable homicide"
        ],
        "statutory_exceptions": []
    },
    "IPC_307_BNS_109": {
        "article_code": "IPC_307_BNS_109",
        "ipc_section": "Section 307 IPC",
        "bns_section": "Section 109 BNS",
        "title": "Attempt to Murder",
        "charge": "Attempt to Murder",
        "penalty_tier": "Tier 4: Life Imprisonment / Capital Sentence (Heinous Offence)",
        "max_imprisonment_years": 99,  # Up to 10 years, or life if hurt caused
        "is_cognizable": True,
        "is_bailable": False,
        "essential_elements": [
            "Act done with such intention or knowledge and under such circumstances that if death caused, would be murder",
            "Execution of act towards commission of murder",
            "Victim survived the assault"
        ],
        "statutory_exceptions": []
    },
    "IPC_326_BNS_118_2": {
        "article_code": "IPC_326_BNS_118_2",
        "ipc_section": "Section 326 IPC",
        "bns_section": "Section 118(2) BNS",
        "title": "Voluntarily Causing Grievous Hurt by Dangerous Weapons or Means",
        "charge": "Voluntarily Causing Grievous Hurt by Dangerous Weapons",
        "penalty_tier": "Tier 4: Life Imprisonment / Capital Sentence (Heinous Offence)",
        "max_imprisonment_years": 99,  # Imprisonment for life or up to 10 years
        "is_cognizable": True,
        "is_bailable": False,
        "essential_elements": [
            "Voluntarily causing grievous hurt",
            "By means of instrument for shooting, stabbing, cutting, or weapon used as offensive instrument likely to cause death"
        ],
        "statutory_exceptions": []
    },
    "IPC_324_BNS_118_1": {
        "article_code": "IPC_324_BNS_118_1",
        "ipc_section": "Section 324 IPC",
        "bns_section": "Section 118(1) BNS",
        "title": "Voluntarily Causing Hurt by Dangerous Weapons or Means",
        "charge": "Voluntarily Causing Hurt by Dangerous Weapons or Means",
        "penalty_tier": "Tier 1: Up to 3 Years (Minor / Summary Offence)",
        "max_imprisonment_years": 3,
        "is_cognizable": True,
        "is_bailable": False,
        "essential_elements": [
            "Voluntarily causing hurt",
            "By dangerous weapon or means",
            "Victim did not suffer grievous hurt or fatality"
        ],
        "statutory_exceptions": []
    },
    "IPC_323_BNS_115": {
        "article_code": "IPC_323_BNS_115",
        "ipc_section": "Section 323 IPC",
        "bns_section": "Section 115(2) BNS",
        "title": "Punishment for Voluntarily Causing Hurt",
        "charge": "Voluntarily Causing Hurt by Dangerous Weapons or Means",
        "penalty_tier": "Tier 1: Up to 3 Years (Minor / Summary Offence)",
        "max_imprisonment_years": 1,
        "is_cognizable": False,
        "is_bailable": True,
        "essential_elements": [
            "Voluntary causation of bodily pain, disease, or infirmity",
            "Without dangerous weapon or severe injury"
        ],
        "statutory_exceptions": []
    },
    "IPC_420_BNS_318": {
        "article_code": "IPC_420_BNS_318",
        "ipc_section": "Section 420 IPC",
        "bns_section": "Section 318(4) BNS",
        "title": "Cheating and Dishonestly Inducing Delivery of Property",
        "charge": "Cheating & Dishonestly Inducing Delivery of Property",
        "penalty_tier": "Tier 2: 3 to 7 Years (Arnesh Kumar Threshold / Medium Penalty)",
        "max_imprisonment_years": 7,
        "is_cognizable": True,
        "is_bailable": False,
        "essential_elements": [
            "Deception of any person",
            "Fraudulent or dishonest inducement to deliver any property",
            "Dishonest intention operating from the very inception"
        ],
        "statutory_exceptions": []
    },
    "IPC_467_BNS_338": {
        "article_code": "IPC_467_BNS_338",
        "ipc_section": "Section 467 IPC",
        "bns_section": "Section 338 BNS",
        "title": "Forgery of Valuable Security, Will, etc.",
        "charge": "Forgery of Valuable Security, Will, or Public Record",
        "penalty_tier": "Tier 4: Life Imprisonment / Capital Sentence (Heinous Offence)",
        "max_imprisonment_years": 99,  # Imprisonment for life or up to 10 years
        "is_cognizable": True,
        "is_bailable": False,
        "essential_elements": [
            "Forgery of document purporting to be valuable security, will, authority to adopt",
            "Dishonest or fraudulent making or alteration of document"
        ],
        "statutory_exceptions": []
    },
    "IPC_468_BNS_336": {
        "article_code": "IPC_468_BNS_336",
        "ipc_section": "Section 468 IPC",
        "bns_section": "Section 336(3) BNS",
        "title": "Forgery for Purpose of Cheating",
        "charge": "Forgery of Valuable Security, Will, or Public Record",
        "penalty_tier": "Tier 2: 3 to 7 Years (Arnesh Kumar Threshold / Medium Penalty)",
        "max_imprisonment_years": 7,
        "is_cognizable": True,
        "is_bailable": False,
        "essential_elements": [
            "Forgery of document",
            "Intention that document forged shall be used for purpose of cheating"
        ],
        "statutory_exceptions": []
    },
    "IPC_471_BNS_340": {
        "article_code": "IPC_471_BNS_340",
        "ipc_section": "Section 471 IPC",
        "bns_section": "Section 340(2) BNS",
        "title": "Using as Genuine a Forged Document",
        "charge": "Forgery of Valuable Security, Will, or Public Record",
        "penalty_tier": "Tier 2: 3 to 7 Years (Arnesh Kumar Threshold / Medium Penalty)",
        "max_imprisonment_years": 7,
        "is_cognizable": True,
        "is_bailable": False,
        "essential_elements": [
            "Fraudulently or dishonestly using as genuine any document known to be forged"
        ],
        "statutory_exceptions": []
    },
    "IPC_406_BNS_316": {
        "article_code": "IPC_406_BNS_316",
        "ipc_section": "Section 406 IPC",
        "bns_section": "Section 316(2) BNS",
        "title": "Punishment for Criminal Breach of Trust",
        "charge": "Theft & Dishonest Misappropriation of Movable Property",
        "penalty_tier": "Tier 1: Up to 3 Years (Minor / Summary Offence)",
        "max_imprisonment_years": 3,
        "is_cognizable": True,
        "is_bailable": False,
        "essential_elements": [
            "Entrustment with property or dominion over property",
            "Dishonest misappropriation or conversion to own use"
        ],
        "statutory_exceptions": []
    },
    "IPC_304B_BNS_80": {
        "article_code": "IPC_304B_BNS_80",
        "ipc_section": "Section 304B IPC",
        "bns_section": "Section 80 BNS",
        "title": "Dowry Death",
        "charge": "Dowry Death / Cruelty by In-Laws",
        "penalty_tier": "Tier 3: 7 to 10 Years (Serious Offence / Substantial Imprisonment)",
        "max_imprisonment_years": 99,  # Minimum 7 years up to life
        "is_cognizable": True,
        "is_bailable": False,
        "essential_elements": [
            "Death of woman caused by burns or bodily injury or occurs otherwise than under normal circumstances",
            "Within 7 years of marriage",
            "Shown that soon before death she was subjected to cruelty or harassment by husband or relatives for dowry"
        ],
        "statutory_exceptions": []
    },
    "IPC_498A_BNS_85": {
        "article_code": "IPC_498A_BNS_85",
        "ipc_section": "Section 498A IPC",
        "bns_section": "Section 85 BNS",
        "title": "Husband or Relative of Husband Subjecting Woman to Cruelty",
        "charge": "Dowry Death / Cruelty by In-Laws",
        "penalty_tier": "Tier 1: Up to 3 Years (Minor / Summary Offence)",
        "max_imprisonment_years": 3,
        "is_cognizable": True,
        "is_bailable": False,
        "essential_elements": [
            "Woman subjected to cruelty by husband or relative of husband",
            "Willful conduct likely to drive woman to suicide or grave injury",
            "Harassment with view to coercing unlawful demand for dowry"
        ],
        "statutory_exceptions": []
    },
    "IPC_376_BNS_64": {
        "article_code": "IPC_376_BNS_64",
        "ipc_section": "Section 376 IPC",
        "bns_section": "Section 64 BNS",
        "title": "Punishment for Rape",
        "charge": "Rape / Aggravated Sexual Assault",
        "penalty_tier": "Tier 4: Life Imprisonment / Capital Sentence (Heinous Offence)",
        "max_imprisonment_years": 99,  # Rigorous imprisonment not less than 10 years up to life
        "is_cognizable": True,
        "is_bailable": False,
        "essential_elements": [
            "Penetrative sexual act without consent or against will",
            "Absence of valid consent under Section 375 IPC / Section 63 BNS"
        ],
        "statutory_exceptions": []
    },
    "IPC_506_BNS_351": {
        "article_code": "IPC_506_BNS_351",
        "ipc_section": "Section 506 IPC",
        "bns_section": "Section 351(2) BNS",
        "title": "Punishment for Criminal Intimidation",
        "charge": "Voluntarily Causing Hurt by Dangerous Weapons or Means",
        "penalty_tier": "Tier 2: 3 to 7 Years (Arnesh Kumar Threshold / Medium Penalty)",
        "max_imprisonment_years": 7,  # If threat be to cause death or grievous hurt, up to 7 years
        "is_cognizable": False,
        "is_bailable": True,
        "essential_elements": [
            "Threat to cause injury to person, reputation, or property",
            "Intent to cause alarm or cause person to do any act he is not legally bound to do"
        ],
        "statutory_exceptions": []
    }
}

LAW_ARTICLES = list(STATUTES_REGISTRY.keys())

# Cost Weights for Multi-Task Heads
ARTICLE_COST_WEIGHTS = {
    "IPC_302_BNS_103": 8.0,
    "IPC_304_PART1_BNS_105": 6.0,
    "IPC_304_PART2_BNS_105": 6.0,
    "IPC_307_BNS_109": 6.0,
    "IPC_326_BNS_118_2": 5.0,
    "IPC_376_BNS_64": 8.0,
    "IPC_304B_BNS_80": 6.0,
    "IPC_392_BNS_309": 5.0,
    "IPC_467_BNS_338": 5.0,
    "IPC_304A_BNS_106": 4.0,
    "IPC_420_BNS_318": 3.0,
    "IPC_468_BNS_336": 3.0,
    "IPC_471_BNS_340": 3.0,
    "IPC_379_BNS_303": 2.0,
    "IPC_406_BNS_316": 2.0,
    "IPC_498A_BNS_85": 3.0,
    "IPC_324_BNS_118_1": 2.5,
    "IPC_323_BNS_115": 1.5,
    "IPC_506_BNS_351": 2.5
}
