"""
========================================================================================
DLKR-LJP: Neural Attention Kernel Bridge (Transformer Execution Core)
========================================================================================
Bridge module orchestrating judicial attention and neural representation
via the high-performance QwenLegalReasoner (supporting Transformer, GPU, and Conceptual modes).
========================================================================================
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

try:
    from llm_reasoner import QwenLegalReasoner
except ImportError:
    try:
        from final.llm_reasoner import QwenLegalReasoner
    except ImportError:
        from denuke.models.llm_reasoner import QwenLegalReasoner

logger = logging.getLogger("neural_attention_kernel")

# Singleton instance
_REASONER_INSTANCE = None

def get_reasoner() -> QwenLegalReasoner:
    global _REASONER_INSTANCE
    if _REASONER_INSTANCE is None:
        _REASONER_INSTANCE = QwenLegalReasoner()
    return _REASONER_INSTANCE

def resolve_neural_attention_context(system_prompt: str, user_content: str, timeout: int = 15) -> Optional[Dict[str, Any]]:
    """
    Executes neural reasoning via DLKR Latent Judicial Transformer (GPU / Transformer / Conceptual).
    Backward-compatible entry point for dlkr_ljp_v5_final.py.
    """
    reasoner = get_reasoner()
    # Extract raw case narrative
    text = user_content
    if "Case Facts:" in user_content:
        text = user_content.split("Case Facts:", 1)[-1].strip()
    elif "Case Narrative:" in user_content:
        text = user_content.split("Case Narrative:", 1)[-1].strip()

    try:
        res = reasoner.reason(text)
        return res
    except Exception as e:
        logger.warning(f"[Neural Attention Kernel] Notice: {e}")
        return None

def get_active_model_info() -> str:
    status_str, is_active = get_reasoner().get_status()
    return status_str
