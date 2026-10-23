"""
========================================================================================
DLKR-LJP NEXT-GEN (DENUKE) - TEMPORAL STATUTORY REGIME DETECTOR
========================================================================================
Determines governing criminal code based on the July 1, 2024 transition threshold:
- Pre-July 1, 2024: Indian Penal Code (IPC 1860) governs (BNS equivalent displayed)
- Post-July 1, 2024: Bharatiya Nyaya Sanhita (BNS 2023) governs (IPC historical referenced)
========================================================================================
"""

import re
import datetime
from typing import Dict, Any, Optional, Tuple
try:
    from config import BNS_EFFECTIVE_DATE, STATUTES_REGISTRY
except ImportError:
    try:
        from final.config import BNS_EFFECTIVE_DATE, STATUTES_REGISTRY
    except ImportError:
        from denuke.config import BNS_EFFECTIVE_DATE, STATUTES_REGISTRY

DATE_PATTERNS = [
    # YYYY-MM-DD or YYYY/MM/DD
    r"\b(20[12]\d)[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b",
    # DD-MM-YYYY or DD/MM/YYYY
    r"\b(0[1-9]|[12]\d|3[01])[-/](0[1-9]|1[0-2])[-/](20[12]\d)\b",
    # Month Day, Year (e.g., July 15, 2024 or 15th August 2024)
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(20[12]\d)\b",
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(20[12]\d)\b",
    # Explicit Year only (e.g., "in the year 2024")
    r"\b(?:in\s+(?:the\s+year\s+)?)(20[12]\d)\b"
]

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
}

def detect_incident_date(text: str) -> Optional[datetime.date]:
    """Extracts the date of the incident from narrative text."""
    # Check DD-MM-YYYY
    m = re.search(r"\b(0?[1-9]|[12]\d|3[01])[-/.](0?[1-9]|1[0-2])[-/.](20[12]\d)\b", text)
    if m:
        try:
            return datetime.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass

    # Check Day Month Year
    m = re.search(r"\b(0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]+(20[12]\d)\b", text, re.I)
    if m:
        try:
            m_str = m.group(2).lower()[:3]
            m_num = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}[m_str]
            return datetime.date(int(m.group(3)), m_num, int(m.group(1)))
        except (ValueError, KeyError):
            pass

    # Check Year only
    m = re.search(r"\b(?:year|in|during)\s+(20[12]\d)\b", text, re.I)
    if m:
        yr = int(m.group(1))
        if yr > 2024:
            return datetime.date(yr, 8, 1)
        elif yr < 2024:
            return datetime.date(yr, 1, 1)

    return None

def resolve_temporal_regime(incident_date: Optional[datetime.date] = None, text: str = "") -> Dict[str, Any]:
    """
    Resolves whether IPC 1860 or BNS 2023 is the primary governing criminal statute.
    """
    if incident_date is None and text:
        incident_date = detect_incident_date(text)

    is_post_bns = False
    if incident_date:
        is_post_bns = incident_date >= BNS_EFFECTIVE_DATE
    else:
        # Check explicit text triggers for BNS enactment
        is_post_bns = bool(re.search(r"\b(?:bns|bharatiya\s+nyaya\s+sanhita|post-?july\s+2024)\b", text, re.I))

    if is_post_bns:
        return {
            "primary_code": "BNS_2023",
            "statute_name": "Bharatiya Nyaya Sanhita, 2023",
            "statute_short": "BNS 2023",
            "secondary_code": "IPC_1860",
            "secondary_name": "Indian Penal Code, 1860 (Historical Precursor)",
            "incident_date": str(incident_date) if incident_date else "Post-July 1, 2024 (Inferred)",
            "legal_note": "Governed by Bharatiya Nyaya Sanhita (BNS 2023), applicable to offences committed on or after July 1, 2024."
        }
    else:
        return {
            "primary_code": "IPC_1860",
            "statute_name": "Indian Penal Code, 1860",
            "statute_short": "IPC 1860",
            "secondary_code": "BNS_2023",
            "secondary_name": "Bharatiya Nyaya Sanhita, 2023 (Modern Equivalent)",
            "incident_date": str(incident_date) if incident_date else "Pre-July 1, 2024 (Historical Default)",
            "legal_note": "Governed by Indian Penal Code (IPC 1860). Modern equivalent provisions under BNS 2023 are harmonized for reference."
        }

def format_concordance_display(article_code: str, regime: Dict[str, Any]) -> Dict[str, str]:
    """Formats the display card showing both statutes with primary highlighted."""
    stat_data = STATUTES_REGISTRY.get(article_code, {})
    ipc_sec = stat_data.get("ipc_section", article_code)
    bns_sec = stat_data.get("bns_section", article_code)

    if regime["primary_code"] == "BNS_2023":
        primary_title = f"{bns_sec} (Primary)"
        secondary_title = f"Equivalent: {ipc_sec}"
        display_label = f"{bns_sec} [IPC: {ipc_sec}]"
    else:
        primary_title = f"{ipc_sec} (Primary)"
        secondary_title = f"BNS Equivalent: {bns_sec}"
        display_label = f"{ipc_sec} | {bns_sec}"

    return {
        "display_label": display_label,
        "primary_statute": primary_title,
        "secondary_statute": secondary_title,
        "ipc_section": ipc_sec,
        "bns_section": bns_sec,
        "regime": regime["primary_code"]
    }
