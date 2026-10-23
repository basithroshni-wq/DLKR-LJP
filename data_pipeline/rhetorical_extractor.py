"""
========================================================================================
DLKR-LJP NEXT-GEN (DENUKE) - ANTI-LEAKAGE RHETORICAL ROLE & FACT EXTRACTOR
========================================================================================
Extracts pure pre-trial incident facts from multi-page court judgments / FIR text.
Strictly strips judicial conclusions and sentences to eliminate circular label leakage.
========================================================================================
"""

import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
try:
    from data_pipeline.temporal_regime import detect_incident_date
except ImportError:
    try:
        from final.data_pipeline.temporal_regime import detect_incident_date
    except ImportError:
        from denuke.data_pipeline.temporal_regime import detect_incident_date

@dataclass
class AtomicCaseFacts:
    narrative_clean: str
    injury_severity: str = "None"  # "Fatal (Death)", "Grievous (Fracture/Vital)", "Simple", "None"
    weapon_used: Optional[str] = None
    weapon_category: str = "Unarmed"
    is_sudden_fight: bool = False
    has_premeditation: bool = False
    is_theft: bool = False
    is_robbery: bool = False
    is_cheating: bool = False
    has_forged_documents: bool = False
    is_rash_driving: bool = False
    is_dowry_death: bool = False
    is_sexual_offence: bool = False
    is_purely_civil: bool = False
    is_attempt_murder: bool = False
    incident_date: Optional[str] = None
    accused_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fatality": self.injury_severity == "Fatal (Death)",
            "injury_severity": self.injury_severity,
            "weapon_used": self.weapon_used,
            "weapon_category": self.weapon_category,
            "sudden_fight": self.is_sudden_fight,
            "is_sudden_fight": self.is_sudden_fight,
            "premeditation": self.has_premeditation,
            "has_premeditation": self.has_premeditation,
            "is_theft": self.is_theft,
            "is_robbery": self.is_robbery,
            "is_cheating": self.is_cheating,
            "has_forged_documents": self.has_forged_documents,
            "is_rash_driving": self.is_rash_driving,
            "is_dowry_death": self.is_dowry_death,
            "is_sexual_offence": self.is_sexual_offence,
            "is_purely_civil": self.is_purely_civil,
            "is_attempt_murder": self.is_attempt_murder,
            "vital_organ_struck": bool(self.injury_severity == "Fatal (Death)" or "vital" in self.narrative_clean.lower() or "head" in self.narrative_clean.lower() or "chest" in self.narrative_clean.lower()),
            "weapon_improvised": bool(self.weapon_used and any(k in str(self.weapon_used).lower() for k in ["stick", "lathi", "stave", "stone", "brick", "rod"])),
            "prior_enmity": self.has_premeditation,
            "lethal_weapon": bool(self.weapon_category in ["Sharp/Cutting", "Firearm"]),
            "heat_of_passion": self.is_sudden_fight,
            "no_premeditation": not self.has_premeditation,
            "single_or_proportional_blow": bool("single" in self.narrative_clean.lower() or "one blow" in self.narrative_clean.lower()),
            "contract_dispute": self.is_purely_civil,
            "violence": bool(self.injury_severity != "None" or self.weapon_used is not None or self.is_sexual_offence)
        }

class RhetoricalFactExtractor:
    """Extracts pre-trial factual narrative and structured legal facts."""

    def extract_pure_facts(self, raw_text: str) -> str:
        """
        Strips judicial holdings, appellate arguments, and sentencing lines
        to return pure factual incident history.
        """
        text = raw_text.strip()
        if len(text) < 400:
            return text

        # Find starting point of incident narrative
        start_patterns = [
            r"(?i)\b(?:the\s+prosecution\s+case\s+(?:is|was)|case\s+of\s+the\s+prosecution)\b",
            r"(?i)\b(?:briefly\s+stated|brief\s+facts|facts\s+giving\s+rise\s+to)\b",
            r"(?i)\b(?:according\s+to\s+the\s+f\.?i\.?r\.?|as\s+per\s+the\s+charge\s*sheet)\b",
            r"(?i)\b(?:the\s+incident\s+occurred\s+on|on\s+the\s+date\s+of\s+incident)\b"
        ]
        start_idx = 0
        for pat in start_patterns:
            m = re.search(pat, text)
            if m:
                start_idx = m.start()
                break

        # Find termination point (arguments or legal determinations)
        term_patterns = [
            r"(?i)\b(?:arguments?\s+of\s+(?:the\s+)?counsel|learned\s+counsel\s+submitted)\b",
            r"(?i)\b(?:points?\s+for\s+determination|charges?\s+were\s+framed)\b",
            r"(?i)\b(?:we\s+have\s+heard\s+the\s+parties|analysis\s+and\s+findings)\b",
            r"(?i)\b(?:the\s+trial\s+court\s+convicted|the\s+high\s+court\s+held)\b"
        ]
        end_idx = len(text)
        for pat in term_patterns:
            m = re.search(pat, text[start_idx:])
            if m and m.start() > 150:
                end_idx = start_idx + m.start()
                break

        facts_span = text[start_idx:end_idx].strip()
        return facts_span if len(facts_span) >= 100 else text[:1500]

    def extract(self, text: str) -> AtomicCaseFacts:
        pure_facts = self.extract_pure_facts(text)
        lower = pure_facts.lower()

        fatal_patterns = [
            r"\bsuccumbed\b", r"\bdeclared dead\b", r"\bpost-mortem\b", r"\bautopsy\b",
            r"\bfatal blows?\b", r"\bcaused death\b", r"\bdied at hospital\b", r"\bdied on the spot\b",
            r"\bhomicide\b", r"\bcorpse\b", r"\bkilling\b", r"\bmurdered\b", r"\bdied\b",
            r"\bkilled\b", r"\blethal blows?\b", r"\bdead\b"
        ]
        grievous_patterns = [
            r"\bfractures?\b", r"\bpermanent disfigurement\b", r"\bvital organs?\b",
            r"\bdeep incisions?\b", r"\bamputations?\b", r"\bsevere bleeding\b",
            r"\bintensive care\b", r"\bicu\b", r"\bcompound fractures?\b",
            r"\blife endangerment\b", r"\bgrievous\b"
        ]
        if any(re.search(p, lower) for p in fatal_patterns):
            severity = "Fatal (Death)"
        elif any(re.search(p, lower) for p in grievous_patterns):
            severity = "Grievous (Fracture/Vital)"
        elif any(k in lower for k in ["injured", "hurt", "bleeding", "wound", "bruise", "assaulted"]):
            severity = "Simple"
        else:
            severity = "None"

        # 2. Weapon detection
        weapon = None
        category = "Unarmed"
        sharp_weapons = ["knife", "dagger", "sword", "blade", "axe", "sickle", "machete", "kirpan", "chopper"]
        firearms = ["gun", "pistol", "revolver", "rifle", "firearm", "country-made pistol", "shotgun"]
        blunt_weapons = ["lathi", "stick", "iron rod", "wooden stave", "stave", "club", "bat", "brick", "stone", "bamboo stick"]

        for w in sharp_weapons:
            if re.search(r"\b" + w + r"\b", lower):
                weapon = w
                category = "Sharp/Cutting"
                break
        if not weapon:
            for w in firearms:
                if re.search(r"\b" + w + r"\b", lower):
                    weapon = w
                    category = "Firearm"
                    break
        if not weapon:
            for w in blunt_weapons:
                if re.search(r"\b" + w + r"\b", lower):
                    weapon = w
                    category = "Blunt Heavy"
                    break

        # 3. Sudden fight / Premeditation
        sudden_fight = bool(any(k in lower for k in [
            "sudden quarrel", "sudden fight", "heat of passion", "altercation",
            "spontaneous", "no premeditation", "no prior enmity", "heated argument", "verbal duel"
        ]))

        # Premeditation with negation check (e.g. "no prior enmity or premeditation")
        has_negation = bool(re.search(r"\b(?:no|without|absence of|not)\s+(?:prior enmity|premeditation|malice)\b", lower))
        premeditation = bool(any(k in lower for k in [
            "premeditated", "lying in wait", "ambush", "prior enmity", "conspiracy",
            "planned in advance", "hired killers", "malice aforethought", "recce", "rivalry"
        ])) and not has_negation

        # 4. Property & Financial
        theft = bool(any(k in lower for k in ["pickpocket", "stole", "theft", "unattended bicycle", "snatched without force", "movable property without consent"]))
        robbery = bool(any(k in lower for k in [
            "robbery", "knifepoint", "gunpoint", "forcibly took", "fear of instant death",
            "fear of instant hurt", "dacoity", "extorted"
        ])) or (any(k in lower for k in ["forcibly took", "under fear of death", "threat of death", "under fear of instant death"]) and any(k in lower for k in ["purse", "phone", "cash", "chain", "gold", "mobile phone"]))
        cheating = bool(any(k in lower for k in ["cheating", "fraud", "dishonestly induced", "fake investment", "deceived complainant", "misappropriated funds"]))
        forged = bool(any(k in lower for k in ["forged", "forgery", "fake document", "false deed", "fabricated will", "spurious stamp"]))

        # 5. Specialized Crimes
        rash_driving = bool(any(k in lower for k in ["rash driving", "negligent driving", "speeding truck", "hit and run", "vehicular accident", "motor vehicle collision"]))
        dowry = bool(any(k in lower for k in ["dowry", "cruelty by husband", "in-laws demanded", "burn injuries within seven years", "strangulation within 7 years of marriage"]))
        sexual_offence = bool(any(k in lower for k in ["rape", "sexual assault", "penetrative", "outraged her modesty", "ravished"]))
        attempt_murder = bool(any(k in lower for k in ["attempt to murder", "slit his throat but survived", "fired shot injuring shoulder", "escaped with bullet wound"]))

        # 6. Civil Dispute Quashing
        is_civil = bool(any(k in lower for k in [
            "commercial dispute", "breach of contract", "civil suit", "partnership reconciliation",
            "dealership agreement", "arbitration clause", "delayed supply", "unpaid invoice", "partition dispute"
        ]) and severity == "None" and not robbery and not sexual_offence)

        inc_date = detect_incident_date(pure_facts)

        return AtomicCaseFacts(
            narrative_clean=pure_facts,
            injury_severity=severity,
            weapon_used=weapon,
            weapon_category=category,
            is_sudden_fight=sudden_fight,
            has_premeditation=premeditation,
            is_theft=theft,
            is_robbery=robbery,
            is_cheating=cheating,
            has_forged_documents=forged,
            is_rash_driving=rash_driving,
            is_dowry_death=dowry,
            is_sexual_offence=sexual_offence,
            is_purely_civil=is_civil,
            is_attempt_murder=attempt_murder,
            incident_date=str(inc_date) if inc_date else None
        )
