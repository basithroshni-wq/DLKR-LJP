"""
========================================================================================
DLKR-LJP NEXT-GEN (DENUKE) - INLEGALBERT ATTENTION MEAN-POOLING ENCODER
========================================================================================
Encodes legal narratives using Attention-Weighted Mean Pooling across all 512 tokens.
Eliminates representation collapse and bottleneck of the single [CLS] token.
========================================================================================
"""

import logging
from typing import List, Optional, Tuple, Any
try:
    from config import BACKBONE_MODEL_NAME
except ImportError:
    try:
        from final.config import BACKBONE_MODEL_NAME
    except ImportError:
        from denuke.config import BACKBONE_MODEL_NAME

logger = logging.getLogger("models")

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    AutoTokenizer = None
    AutoModel = None

class InLegalBertMeanEncoder:
    def __init__(self, model_name: str = BACKBONE_MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.tokenizer = None
        self.backbone = None
        self.is_loaded = False

        if HAS_TORCH:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

    def load_model(self):
        """Loads InLegalBERT tokenizer and transformer backbone."""
        if not HAS_TORCH:
            logger.info("[Encoder] PyTorch not installed. Operating in Statistical Prober mode.")
            return

        try:
            logger.info(f"[Encoder] Loading {self.model_name} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.backbone = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.is_loaded = True
            logger.info(f"[Encoder] ✅ Successfully initialized {self.model_name} with Attention Mean-Pooling.")
        except Exception as e:
            logger.warning(f"[Encoder Notice] Could not load {self.model_name} ({e}). Fallback prober active.")
            self.is_loaded = False

    @staticmethod
    def attention_mean_pooling(model_output, attention_mask):
        """
        Computes attention-weighted mean token pooling over the last hidden state.
        Aggregates factual signal across every non-padding token in the sequence.
        """
        token_embeddings = model_output.last_hidden_state  # [Batch, SeqLen, 768]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode_texts(self, texts: List[str], max_length: int = 512, batch_size: int = 16) -> Any:
        """
        Encodes a batch of legal texts into a normalized 768-dimensional tensor.
        """
        if not HAS_TORCH or not self.is_loaded:
            return None

        all_pooled = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tokens = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                out = self.backbone(**tokens)
                pooled = self.attention_mean_pooling(out, tokens["attention_mask"])
                norm_pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
                all_pooled.append(norm_pooled.cpu())

        return torch.cat(all_pooled, dim=0).to(self.device)

    def encode_single(self, text: str, max_length: int = 512) -> Any:
        """Encodes a single text narrative into a 768-dim tensor vector."""
        res = self.encode_texts([text], max_length=max_length, batch_size=1)
        return res if res is not None else None
