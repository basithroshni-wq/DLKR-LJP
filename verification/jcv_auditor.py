"""
========================================================================================
DLKR-LJP NEXT-GEN (DENUKE) - JUDICIAL CONSISTENCY VERIFIER (JCV AUDITOR)
========================================================================================
Symbolic statutory verifier enforcing:
1. Statutory sentencing ceilings (e.g. 10 years for IPC 304 Part II)
2. Procedural arrest floors (Arnesh Kumar v. State of Bihar under 7 years)
3. Homicide supercedence (barring non-fatal hurt charges once fatality occurs)
4. Section 482 CrPC / Section 528 BNSS quashing on civil contract disputes
========================================================================================
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
try:
    from config import STATUTES_REGISTRY, TIER_LEVEL_ORDINAL
except ImportError:
    try:
        from final.config import STATUTES_REGISTRY, TIER_LEVEL_ORDINAL
    except ImportError:
        from denuke.config import STATUTES_REGISTRY, TIER_LEVEL_ORDINAL

try:
    from data_pipeline.rhetorical_extractor import AtomicCaseFacts
except ImportError:
    try:
        from final.data_pipeline.rhetorical_extractor import AtomicCaseFacts
    except ImportError:
        from denuke.data_pipeline.rhetorical_extractor import AtomicCaseFacts

try:
    from graph_engine.statutory_graph import GraphTraversalResult
except ImportError:
    try:
        from final.graph_engine.statutory_graph import GraphTraversalResult
    except ImportError:
        from denuke.graph_engine.statutory_graph import GraphTraversalResult

@dataclass
class JudicialAuditReport:
    is_valid: bool
    jci_score: float  # 0.0 to 100.0
    consistency_status: str  # "CERTIFIED_VALID", "RECONCILED", "NON_COGNIZABLE_QUASHED"
    statutory_ceiling_years: int
    reconciliation_applied: bool = False
    audit_notes: List[str] = field(default_factory=list)
    procedural_safeguards: List[str] = field(default_factory=list)

class JudicialConsistencyVerifier:
    """Audits multi-head neural predictions against codified statutory bounds."""

    def audit(
        self,
        predicted_article: str,
        predicted_charge: str,
        predicted_tier: str,
        atomic_facts: AtomicCaseFacts,
        graph_result: Optional[GraphTraversalResult] = None
    ) -> JudicialAuditReport:
        notes: List[str] = []
        safeguards: List[str] = []
        jci = 98.0
        reconciled = False
        status = "CERTIFIED_VALID"

        stat_info = STATUTES_REGISTRY.get(predicted_article, {})
        max_years = stat_info.get("max_imprisonment_years", 10)

        # 1. Civil Dispute Quashing Audit
        if atomic_facts.is_purely_civil and atomic_facts.injury_severity == "None":
            notes.append("Commercial/civil dispute identified. Quashing standard u/s 482 CrPC / s.528 BNSS applied.")
            return JudicialAuditReport(
                is_valid=True,
                jci_score=99.0,
                consistency_status="NON_COGNIZABLE_QUASHED",
                statutory_ceiling_years=0,
                reconciliation_applied=False,
                audit_notes=notes,
                procedural_safeguards=["Section 482 CrPC Inherent Powers Standard (State of Haryana v. Bhajan Lal)"]
            )

        # 2. Homicide Supercedence Check (Fatality vs. Hurt)
        # If victim is deceased, charging under simple/dangerous hurt (IPC 324/323) is a fatal jurisprudence error
        if atomic_facts.injury_severity == "Fatal (Death)":
            if predicted_article in ["IPC_324_BNS_118_1", "IPC_326_BNS_118_2", "IPC_323_BNS_115"]:
                reconciled = True
                status = "RECONCILED"
                notes.append("FATAL ERROR AVERTED: Neural model predicted non-fatal hurt on deceased victim. Overridden to Culpable Homicide under statutory doctrine.")

                if graph_result and graph_result.applied_exception:
                    predicted_article = "IPC_304_PART2_BNS_105"
                    max_years = 10
                    notes.append("Reconciled to IPC 304 Part II / BNS 105 under Section 300 Exception 4.")
                else:
                    predicted_article = "IPC_302_BNS_103"
                    max_years = 99
                    notes.append("Reconciled to IPC 302 / BNS 103 (Murder).")
                jci = 95.0

        # 3. Arnesh Kumar Procedural Floor Check (Offences under 7 years)
        pred_ordinal = TIER_LEVEL_ORDINAL.get(predicted_tier, 2)
        if pred_ordinal <= 2 or max_years <= 7:
            safeguards.append("Arnesh Kumar v. State of Bihar (2014) 8 SCC 273: Mandatory Section 41A CrPC notice before arrest.")
            notes.append("Offence falls within Arnesh Kumar protective ceiling (<= 7 years imprisonment).")

        # 4. Exception 4 Sentence Ceiling Verification
        if graph_result and graph_result.applied_exception:
            notes.append(f"Statutory mitigation applied: {graph_result.applied_exception}")
            if max_years > 10:
                max_years = 10
                notes.append("Sentencing ceiling capped at statutory limit of 10 years u/s 304 Part II.")

        return JudicialAuditReport(
            is_valid=True,
            jci_score=round(jci, 1),
            consistency_status=status,
            statutory_ceiling_years=max_years,
            reconciliation_applied=reconciled,
            audit_notes=notes,
            procedural_safeguards=safeguards
        )
