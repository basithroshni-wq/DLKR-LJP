"""
========================================================================================
DLKR-LJP NEXT-GEN (DENUKE) - COST-SENSITIVE ASYMMETRIC LOSS MATRIX
========================================================================================
Applies Severity-Proportional positive weights to penalize false negatives on
heinous/fatal crimes (Murder, Culpable Homicide, Rape) up to 10x more than minor theft.
========================================================================================
"""

from typing import List, Dict, Any
try:
    from config import LAW_ARTICLES, CRIMINAL_CHARGES
except ImportError:
    try:
        from final.config import LAW_ARTICLES, CRIMINAL_CHARGES
    except ImportError:
        from denuke.config import LAW_ARTICLES, CRIMINAL_CHARGES

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

ARTICLE_SEVERITY_COSTS: Dict[str, float] = {
    "IPC_302_BNS_103": 10.0,       # Murder: Capital / Life Sentence
    "IPC_304_PART1_BNS_105": 8.0,   # Culpable Homicide (Intention)
    "IPC_304_PART2_BNS_105": 8.0,   # Culpable Homicide (Knowledge / Sudden Fight)
    "IPC_376_BNS_64": 9.0,         # Rape / Sexual Assault
    "IPC_304B_BNS_80": 8.5,        # Dowry Death
    "IPC_307_BNS_109": 7.5,        # Attempt to Murder
    "IPC_326_BNS_118_2": 5.0,      # Grievous Hurt by Dangerous Weapon
    "IPC_392_BNS_309": 4.0,        # Robbery
    "IPC_324_BNS_118_1": 2.5,      # Hurt by Dangerous Weapon
    "IPC_304A_BNS_106": 4.0,       # Death by Rash / Negligent Act
    "IPC_467_BNS_338": 2.5,        # Forgery of Valuable Security
    "IPC_420_BNS_318": 1.5,        # Cheating
    "IPC_379_BNS_303": 1.0         # Simple Theft
}

CHARGE_SEVERITY_COSTS: Dict[str, float] = {
    "Murder / Intentional Homicide": 10.0,
    "Culpable Homicide Not Amounting to Murder": 8.0,
    "Rape / Aggravated Sexual Assault": 9.0,
    "Dowry Death / Cruelty by In-Laws": 8.5,
    "Attempt to Murder": 7.5,
    "Voluntarily Causing Grievous Hurt by Dangerous Weapons": 5.0,
    "Robbery / Extortion with Force": 4.0,
    "Voluntarily Causing Hurt by Dangerous Weapons or Means": 2.5,
    "Death by Rash and Negligent Act (Vehicular / Industrial)": 4.0,
    "Forgery of Valuable Security, Will, or Public Record": 2.5,
    "Cheating & Dishonestly Inducing Delivery of Property": 1.5,
    "Theft & Dishonest Misappropriation of Movable Property": 1.0
}

def get_cost_sensitive_criteria(device: str = "cpu"):
    """
    Returns loss functions with statutory severity weights for Head 1 and Head 2.
    """
    if not HAS_TORCH:
        return None, None, None

    art_weights = torch.tensor(
        [ARTICLE_SEVERITY_COSTS.get(a, 1.0) for a in LAW_ARTICLES],
        dtype=torch.float32,
        device=device
    )
    chg_weights = torch.tensor(
        [CHARGE_SEVERITY_COSTS.get(c, 1.0) for c in CRIMINAL_CHARGES],
        dtype=torch.float32,
        device=device
    )

    crit_art = nn.BCEWithLogitsLoss(pos_weight=art_weights)
    crit_chg = nn.BCEWithLogitsLoss(pos_weight=chg_weights)
    crit_tier = nn.CrossEntropyLoss()

    return crit_art, crit_chg, crit_tier
