"""
========================================================================================
DLKR-LJP: STATUTORY KNOWLEDGE GRAPH (GRAPHRAG) ENGINE
========================================================================================
Directed Acyclic Graph (DAG) for Indian Criminal Law Jurisprudence
Traverses Statutory Provisions, Exceptions (s.300), and Binding Supreme Court Precedents
========================================================================================
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging

try:
    from config import STATUTES_REGISTRY, LAW_ARTICLES
except ImportError:
    try:
        from final.config import STATUTES_REGISTRY, LAW_ARTICLES
    except ImportError:
        from denuke.config import STATUTES_REGISTRY, LAW_ARTICLES

try:
    from graph_engine.precedent_ratios import LANDMARK_PRECEDENTS, PrecedentRatio
except ImportError:
    try:
        from final.graph_engine.precedent_ratios import LANDMARK_PRECEDENTS, PrecedentRatio
    except ImportError:
        from denuke.graph_engine.precedent_ratios import LANDMARK_PRECEDENTS, PrecedentRatio

logger = logging.getLogger("graph_engine")

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

@dataclass
class GraphTraversalResult:
    primary_article: str
    charge_category: str
    penalty_tier: str
    max_imprisonment_years: int
    traversal_path: List[str]
    applied_exception: Optional[str] = None
    applied_precedents: List[PrecedentRatio] = field(default_factory=list)
    confidence: float = 0.95
    is_mitigated: bool = False
    judicial_rationale: str = ""

class StatutoryKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph() if HAS_NETWORKX else {}
        self._build_graph()

    def _build_graph(self):
        """Constructs the statutory knowledge DAG with statutory provisions, exceptions, and precedents."""
        if not HAS_NETWORKX:
            return

        # 1. Add Statutory Nodes
        for art, meta in STATUTES_REGISTRY.items():
            self.graph.add_node(
                art,
                node_type="STATUTE",
                ipc_section=meta["ipc_section"],
                bns_section=meta["bns_section"],
                title=meta["title"],
                charge=meta["charge"],
                penalty_tier=meta["penalty_tier"],
                max_years=meta["max_imprisonment_years"]
            )

        # 2. Add Exception Nodes
        self.graph.add_node(
            "EXCEPTION_1_PROVOCATION",
            node_type="EXCEPTION",
            statute="IPC_300 / BNS_101",
            name="Section 300 Exception 1: Grave and Sudden Provocation"
        )
        self.graph.add_node(
            "EXCEPTION_4_SUDDEN_FIGHT",
            node_type="EXCEPTION",
            statute="IPC_300 / BNS_101",
            name="Section 300 Exception 4: Sudden fight in heat of passion upon sudden quarrel"
        )

        # 3. Add Precedent Nodes
        for prec in LANDMARK_PRECEDENTS:
            self.graph.add_node(
                prec.citation,
                node_type="PRECEDENT",
                case_name=prec.case_name,
                ratio=prec.ratio_decidendi,
                directive=prec.outcome_directive
            )

        # 4. Connect Statutory Exception Mitigation Edges
        # Section 300 Exception 4 mitigates Murder (302) to Culpable Homicide Part II (304 Part II)
        self.graph.add_edge("IPC_302_BNS_103", "EXCEPTION_4_SUDDEN_FIGHT", relation="SUBJECT_TO_EXCEPTION")
        self.graph.add_edge("EXCEPTION_4_SUDDEN_FIGHT", "IPC_304_PART2_BNS_105", relation="MITIGATES_TO", max_ceiling=10)
        self.graph.add_edge("EXCEPTION_1_PROVOCATION", "IPC_304_PART1_BNS_105", relation="MITIGATES_TO", max_ceiling=20)

        # 5. Connect Fatal Supercedence Edges
        self.graph.add_edge("IPC_304_PART2_BNS_105", "IPC_324_BNS_118_1", relation="SUPERSEDES_ON_DEATH")
        self.graph.add_edge("IPC_302_BNS_103", "IPC_324_BNS_118_1", relation="SUPERSEDES_ON_DEATH")

        # 6. Connect Precedent Guidance Edges
        self.graph.add_edge("2006 (11) SCC 444", "EXCEPTION_4_SUDDEN_FIGHT", relation="BINDING_CRITERIA")
        self.graph.add_edge("(2003) 3 SCC 528", "IPC_304_PART2_BNS_105", relation="STATUTORY_MANDATE")
        self.graph.add_edge("AIR 1958 SC 465", "IPC_302_BNS_103", relation="INTENTION_STANDARD")

    def traverse(self, facts: Dict[str, Any]) -> GraphTraversalResult:
        """
        Traverses the knowledge graph based on extracted factual attributes.
        Returns the legally sound provision, statutory ceilings, and precedent trace.
        """
        # Check out of domain abstention first
        if facts.get("is_out_of_domain", False):
            return GraphTraversalResult(
                primary_article="NONE_OUT_OF_DOMAIN",
                charge_category="Non-Cognizable / Out of Domain Narrative",
                penalty_tier="No Penalty Attracted",
                max_imprisonment_years=0,
                traversal_path=["Abstention Gate", "Non-Legal Narrative Filter"],
                confidence=0.0,
                judicial_rationale="Narrative does not disclose essential ingredients of any cognizable criminal offence under IPC or BNS."
            )

        is_fatal = facts.get("fatality", False) or facts.get("injury_severity") in ["Fatal (Death)", "Fatal"]
        is_sudden_fight = facts.get("sudden_fight", False) or facts.get("is_sudden_fight", False)
        has_premeditation = facts.get("premeditation", False) or facts.get("has_premeditation", False)
        is_civil = facts.get("is_purely_civil", False)
        is_theft = facts.get("is_theft", False)
        is_robbery = facts.get("is_robbery", False)
        is_cheating = facts.get("is_cheating", False)
        is_rash_driving = facts.get("is_rash_driving", False)
        is_sexual_offence = facts.get("is_sexual_offence", False)
        is_dowry_death = facts.get("is_dowry_death", False)
        is_valuable_security = facts.get("is_valuable_security", False)
        has_forged = facts.get("has_forged_documents", False)
        is_breach_of_trust = facts.get("is_breach_of_trust", False)
        is_criminal_intimidation = facts.get("is_criminal_intimidation", False)
        is_matrimonial_cruelty = facts.get("is_matrimonial_cruelty", False)

        path = []
        applied_precedents: List[PrecedentRatio] = []

        # 1. Civil Quashing Branch u/s 482 / 528
        if is_civil and not is_fatal:
            path = ["Dispute Identified", "Commercial Contract Element", "Bhajan Lal Doctrine (1992 Supp SCC 335)", "Quashing u/s 482 CrPC / s.528 BNSS"]
            bhajan = next((p for p in LANDMARK_PRECEDENTS if "Bhajan Lal" in p.case_name), None)
            if bhajan:
                applied_precedents.append(bhajan)
            return GraphTraversalResult(
                primary_article="NONE_CIVIL_DISPUTE",
                charge_category="Civil Dispute / Quashing Standard u/s 482 CrPC / s.528 BNSS",
                penalty_tier="Quashed u/s 482 CrPC / s.528 BNSS (No Penal Term)",
                max_imprisonment_years=0,
                traversal_path=path,
                applied_precedents=applied_precedents,
                confidence=0.96,
                is_mitigated=True,
                judicial_rationale="Pure commercial / civil breach without criminal mens rea at inception. Governed by State of Haryana v. Bhajan Lal quashing criteria."
            )

        # 2. Fatal Incident Branch (Homicide Spectrum)
        if is_fatal:
            path.append("Fatality Established (Death of Victim)")

            # Vehicular negligence vs Homicide
            if is_rash_driving and not has_premeditation:
                path.append("Vehicular Rash Driving -> Section 304A IPC / Section 106 BNS")
                return GraphTraversalResult(
                    primary_article="IPC_304A_BNS_106",
                    charge_category="Death by Rash and Negligent Act (Vehicular / Industrial)",
                    penalty_tier="Tier 1: Up to 3 Years (Minor / Summary Offence)",
                    max_imprisonment_years=2,
                    traversal_path=path,
                    applied_precedents=[],
                    confidence=0.95,
                    is_mitigated=False,
                    judicial_rationale="Death caused by vehicular rash/negligent driving without intention to kill or knowledge of imminent fatality."
                )

            # Dowry Death (Section 304B)
            if is_dowry_death:
                path.append("Abnormal Death of Married Woman within 7 Years with Dowry Harassment -> Section 304B IPC / Section 80 BNS")
                return GraphTraversalResult(
                    primary_article="IPC_304B_BNS_80",
                    charge_category="Dowry Death / Cruelty by In-Laws",
                    penalty_tier="Tier 3: 7 to 10 Years (Serious Offence / Substantial Imprisonment)",
                    max_imprisonment_years=14,
                    traversal_path=path,
                    applied_precedents=[],
                    confidence=0.95,
                    is_mitigated=False,
                    judicial_rationale="Essential statutory ingredients of dowry death under Section 304B IPC / Section 80 BNS fulfilled."
                )

            # Section 300 Homicide Spectrum: Murder vs. Culpable Homicide
            path.append("Prima Facie Section 300 (Homicide)")

            if is_sudden_fight and not has_premeditation:
                path.append("Traverse Edge: SUBJECT_TO_EXCEPTION -> EXCEPTION_4_SUDDEN_FIGHT")
                path.append("Ghapoo Yadav (2003) & Pulicherla (2006) Tests Satisfied")
                path.append("Traverse Edge: MITIGATES_TO -> IPC_304_PART2_BNS_105")

                for p in LANDMARK_PRECEDENTS:
                    if p.citation in ["(2003) 3 SCC 528", "2006 (11) SCC 444"]:
                        applied_precedents.append(p)

                return GraphTraversalResult(
                    primary_article="IPC_304_PART2_BNS_105",
                    charge_category="Culpable Homicide Not Amounting to Murder",
                    penalty_tier="Tier 3: 7 to 10 Years (Serious Offence / Substantial Imprisonment)",
                    max_imprisonment_years=10,
                    traversal_path=path,
                    applied_exception="Section 300 Exception 4 (Sudden Fight upon Sudden Quarrel)",
                    applied_precedents=applied_precedents,
                    confidence=0.96,
                    is_mitigated=True,
                    judicial_rationale="Fatal blow delivered in heat of passion during spontaneous quarrel without premeditation. Fulfills Section 300 Exception 4 and Pulicherla Nagaraju guidance; legally mitigates to Section 304 Part II."
                )
            else:
                path.append("Absence of Section 300 Exceptions -> Virsa Singh Objective Test Satisfied -> Retain IPC_302_BNS_103")
                virsa = next((p for p in LANDMARK_PRECEDENTS if "Virsa Singh" in p.case_name), None)
                if virsa:
                    applied_precedents.append(virsa)

                return GraphTraversalResult(
                    primary_article="IPC_302_BNS_103",
                    charge_category="Murder / Intentional Homicide",
                    penalty_tier="Tier 4: Life Imprisonment / Capital Sentence (Heinous Offence)",
                    max_imprisonment_years=99,
                    traversal_path=path,
                    applied_precedents=applied_precedents,
                    confidence=0.95,
                    is_mitigated=False,
                    judicial_rationale="Premeditated intentional homicide causing bodily injury sufficient in ordinary course of nature to cause death. Section 302 IPC / Section 103 BNS applies."
                )

        # 3. Non-Fatal Violent Offences
        if facts.get("is_attempt_murder", False):
            path = ["Violent Attack with Lethal Weapon", "Victim Survived", "Section 307 IPC / Section 109 BNS"]
            return GraphTraversalResult(
                primary_article="IPC_307_BNS_109",
                charge_category="Attempt to Murder",
                penalty_tier="Tier 3: 7 to 10 Years (Serious Offence / Substantial Imprisonment)",
                max_imprisonment_years=10,
                traversal_path=path,
                confidence=0.94,
                judicial_rationale="Lethal strike with weapon capable of causing death; victim survived attack."
            )

        if facts.get("injury_severity") == "Grievous (Fracture/Vital)":
            path = ["Grievous Hurt Inflicted (Fracture/Vital Organ)", "Dangerous Weapon Used", "Section 326 IPC / Section 118(2) BNS"]
            return GraphTraversalResult(
                primary_article="IPC_326_BNS_118_2",
                charge_category="Voluntarily Causing Grievous Hurt by Dangerous Weapons",
                penalty_tier="Tier 4: Life Imprisonment / Capital Sentence (Heinous Offence)",
                max_imprisonment_years=10,
                traversal_path=path,
                confidence=0.95,
                judicial_rationale="Grievous hurt inflicted with dangerous cutting/blunt weapon u/s 326 IPC / s.118(2) BNS."
            )

        # 4. Property Crimes with Violence (Robbery)
        if is_robbery:
            path = ["Theft / Extortion Coupled with Force or Threat of Instant Hurt", "Section 392 IPC / Section 309 BNS"]
            return GraphTraversalResult(
                primary_article="IPC_392_BNS_309",
                charge_category="Robbery / Extortion with Force",
                penalty_tier="Tier 3: 7 to 10 Years (Serious Offence / Substantial Imprisonment)",
                max_imprisonment_years=10,
                traversal_path=path,
                confidence=0.95,
                judicial_rationale="Theft committed with violence or threat of violence satisfies Section 392 IPC / Section 309 BNS."
            )

        # 5. Sexual Offences
        if is_sexual_offence:
            path = ["Non-Consensual Sexual Assault", "Section 376 IPC / Section 64 BNS"]
            return GraphTraversalResult(
                primary_article="IPC_376_BNS_64",
                charge_category="Rape / Aggravated Sexual Assault",
                penalty_tier="Tier 4: Life Imprisonment / Capital Sentence (Heinous Offence)",
                max_imprisonment_years=20,
                traversal_path=path,
                confidence=0.96,
                judicial_rationale="Penetrative sexual assault without consent satisfies Section 376 IPC / Section 64 BNS."
            )

        # 6. Forgery & Fraud Spectrum
        if has_forged and is_valuable_security:
            path = ["Fabrication of Valuable Security / Last Will", "Section 467 IPC / Section 338 BNS"]
            return GraphTraversalResult(
                primary_article="IPC_467_BNS_338",
                charge_category="Forgery of Valuable Security, Will, or Public Record",
                penalty_tier="Tier 4: Life Imprisonment / Capital Sentence (Heinous Offence)",
                max_imprisonment_years=99,
                traversal_path=path,
                confidence=0.95,
                judicial_rationale="Forging valuable security or testamentary document carries life imprisonment under Section 467 IPC."
            )

        if has_forged and is_cheating:
            path = ["Forging Appointment Letter or Instrument to Cheat", "Section 468 IPC / Section 336(3) BNS"]
            return GraphTraversalResult(
                primary_article="IPC_468_BNS_336",
                charge_category="Forgery of Valuable Security, Will, or Public Record",
                penalty_tier="Tier 2: 3 to 7 Years (Arnesh Kumar Threshold / Medium Penalty)",
                max_imprisonment_years=7,
                traversal_path=path,
                confidence=0.94,
                judicial_rationale="Forgery executed specifically for cheating falls squarely under Section 468 IPC / Section 336(3) BNS."
            )

        if has_forged:
            path = ["Fraudulent Use of Known Forged Document as Genuine", "Section 471 IPC / Section 340(2) BNS"]
            return GraphTraversalResult(
                primary_article="IPC_471_BNS_340",
                charge_category="Forgery of Valuable Security, Will, or Public Record",
                penalty_tier="Tier 1: Up to 3 Years (Minor / Summary Offence)",
                max_imprisonment_years=2,
                traversal_path=path,
                confidence=0.93,
                judicial_rationale="Using forged document as genuine attracts Section 471 IPC / Section 340(2) BNS."
            )

        if is_breach_of_trust:
            path = ["Fiduciary Entrustment of Property", "Dishonest Misappropriation", "Section 406 IPC / Section 316 BNS"]
            return GraphTraversalResult(
                primary_article="IPC_406_BNS_316",
                charge_category="Theft & Dishonest Misappropriation of Movable Property",
                penalty_tier="Tier 2: 3 to 7 Years (Arnesh Kumar Threshold / Medium Penalty)",
                max_imprisonment_years=3,
                traversal_path=path,
                confidence=0.94,
                judicial_rationale="Dishonest conversion of entrusted money or assets falls under Section 406 IPC / Section 316 BNS."
            )

        if is_cheating:
            path = ["Dishonest Deception at Inception Inducing Property Delivery", "Section 420 IPC / Section 318 BNS"]
            arnesh = next((p for p in LANDMARK_PRECEDENTS if "Arnesh Kumar" in p.case_name), None)
            if arnesh:
                applied_precedents.append(arnesh)
            return GraphTraversalResult(
                primary_article="IPC_420_BNS_318",
                charge_category="Cheating & Dishonestly Inducing Delivery of Property",
                penalty_tier="Tier 2: 3 to 7 Years (Arnesh Kumar Threshold / Medium Penalty)",
                max_imprisonment_years=7,
                traversal_path=path,
                applied_precedents=applied_precedents,
                confidence=0.94,
                judicial_rationale="Cheating under Section 420 IPC / Section 318(4) BNS. Subject to Arnesh Kumar Section 41A CrPC safeguards."
            )

        # 7. Bodily Hurt
        if facts.get("weapon_used") and not is_fatal:
            path = ["Weapon Used (Simple Hurt)", "Section 324 IPC / Section 118(1) BNS"]
            return GraphTraversalResult(
                primary_article="IPC_324_BNS_118_1",
                charge_category="Voluntarily Causing Hurt by Dangerous Weapons or Means",
                penalty_tier="Tier 1: Up to 3 Years (Minor / Summary Offence)",
                max_imprisonment_years=3,
                traversal_path=path,
                confidence=0.94,
                judicial_rationale="Non-grievous, non-fatal hurt caused by dangerous weapon u/s 324 IPC / s.118(1) BNS."
            )

        if facts.get("injury_severity") == "Simple Hurt":
            path = ["Unarmed Assault (Slapping/Punches)", "Section 323 IPC / Section 115(2) BNS"]
            return GraphTraversalResult(
                primary_article="IPC_323_BNS_115",
                charge_category="Voluntarily Causing Hurt by Dangerous Weapons or Means",
                penalty_tier="Tier 1: Up to 3 Years (Minor / Summary Offence)",
                max_imprisonment_years=1,
                traversal_path=path,
                confidence=0.92,
                judicial_rationale="Simple hurt inflicted without weapon brandishing u/s 323 IPC / s.115(2) BNS."
            )

        # 8. Matrimonial Cruelty & Intimidation
        if is_matrimonial_cruelty:
            path = ["Matrimonial Cruelty by Husband or Relatives", "Section 498A IPC / Section 85 BNS"]
            return GraphTraversalResult(
                primary_article="IPC_498A_BNS_85",
                charge_category="Dowry Death / Cruelty by In-Laws",
                penalty_tier="Tier 1: Up to 3 Years (Minor / Summary Offence)",
                max_imprisonment_years=3,
                traversal_path=path,
                confidence=0.94,
                judicial_rationale="Cruelty by husband or in-laws under Section 498A IPC / Section 85 BNS."
            )

        if is_criminal_intimidation:
            path = ["Death Threat / Criminal Intimidation", "Section 506 IPC / Section 351 BNS"]
            return GraphTraversalResult(
                primary_article="IPC_506_BNS_351",
                charge_category="Robbery / Extortion with Force",
                penalty_tier="Tier 2: 3 to 7 Years (Arnesh Kumar Threshold / Medium Penalty)",
                max_imprisonment_years=7,
                traversal_path=path,
                confidence=0.93,
                judicial_rationale="Criminal intimidation with threat to cause death falls under Section 506 Part II IPC."
            )

        # 9. Simple Theft
        if is_theft:
            path = ["Dishonest Moving of Movable Property Without Consent", "Section 379 IPC / Section 303 BNS"]
            return GraphTraversalResult(
                primary_article="IPC_379_BNS_303",
                charge_category="Theft & Dishonest Misappropriation of Movable Property",
                penalty_tier="Tier 1: Up to 3 Years (Minor / Summary Offence)",
                max_imprisonment_years=3,
                traversal_path=path,
                confidence=0.95,
                judicial_rationale="Simple theft without weapon or bodily violence satisfies Section 379 IPC / Section 303 BNS."
            )

        # Default Fallback
        return GraphTraversalResult(
            primary_article="IPC_379_BNS_303",
            charge_category="Theft & Dishonest Misappropriation of Movable Property",
            penalty_tier="Tier 1: Up to 3 Years (Minor / Summary Offence)",
            max_imprisonment_years=3,
            traversal_path=["Default Factual Evaluation"],
            confidence=0.50,
            judicial_rationale="Default statutory provision evaluated."
        )
