"""
========================================================================================
DLKR-LJP NEXT-GEN (DENUKE) - MULTI-TASK NEURAL PREDICTION HEADS
========================================================================================
Tri-Head Classification Network operating over Attention Mean-Pooled Legal Vectors:
- Head 1: Applicable Statutory Articles (Multi-Label Sigmoid)
- Head 2: Criminal Charge Taxonomy (Multi-Label Sigmoid)
- Head 3: Statutory Penalty Severity Tiers (Multi-Class Softmax)
========================================================================================
"""

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

class MultiTaskLegalHead(nn.Module if HAS_TORCH else object):
    def __init__(self, hidden_size: int = 768, num_articles: int = 13, num_charges: int = 12, num_tiers: int = 4):
        if not HAS_TORCH:
            return
        super().__init__()
        self.dropout = nn.Dropout(0.2)

        # Head 1: Statutory Articles
        self.article_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, num_articles)
        )

        # Head 2: Criminal Charges
        self.charge_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, num_charges)
        )

        # Head 3: Penalty Severity Tiers
        self.tier_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, num_tiers)
        )

    def forward(self, pooled_rep):
        if not HAS_TORCH:
            return None, None, None
        drop = self.dropout(pooled_rep)
        return (
            self.article_head(drop),
            self.charge_head(drop),
            self.tier_head(drop)
        )
