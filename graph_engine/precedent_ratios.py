"""
========================================================================================
DLKR-LJP NEXT-GEN (DENUKE) - LANDMARK SUPREME COURT PRECEDENT REGISTRY
========================================================================================
Codified Binding Judicial Ratios for Indian Criminal Law Reasoning
========================================================================================
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class PrecedentRatio:
    citation: str
    case_name: str
    year: int
    court: str
    statutory_provisions: List[str]
    legal_doctrine: str
    ratio_decidendi: str
    trigger_criteria: Dict[str, Any]
    outcome_directive: str

LANDMARK_PRECEDENTS: List[PrecedentRatio] = [
    PrecedentRatio(
        citation="2006 (11) SCC 444",
        case_name="Pulicherla Nagaraju alias Nagaraja Reddy v. State of A.P.",
        year=2006,
        court="Supreme Court of India",
        statutory_provisions=["IPC_302_BNS_103", "IPC_304_PART1_BNS_105", "IPC_304_PART2_BNS_105"],
        legal_doctrine="Intention Determination Factors for Murder vs. Culpable Homicide",
        ratio_decidendi="Intention to cause death must be gathered from all attendant circumstances: (1) nature of weapon used, (2) whether weapon carried or picked up spontaneously, (3) whether blow aimed at vital part, (4) amount of force employed, (5) whether act committed in course of sudden quarrel, and (6) whether offender took undue advantage.",
        trigger_criteria={
            "fatality": True,
            "vital_organ_struck": True,
            "weapon_improvised": True,
            "prior_enmity": False
        },
        outcome_directive="Mitigate from Murder (IPC 302) to Culpable Homicide Part II (IPC 304 Part II) when fatal strike was spontaneous single blow in sudden quarrel without premeditation."
    ),
    PrecedentRatio(
        citation="AIR 1958 SC 465",
        case_name="Virsa Singh v. State of Punjab",
        year=1958,
        court="Supreme Court of India",
        statutory_provisions=["IPC_302_BNS_103"],
        legal_doctrine="Four-Point Test for Clause 3 of Section 300 IPC",
        ratio_decidendi="For Section 300 Thirdly, prosecution must prove: (1) bodily injury present, (2) nature of injury proved objectively, (3) intention to inflict that particular bodily injury, (4) that injury is sufficient in ordinary course of nature to cause death.",
        trigger_criteria={
            "fatality": True,
            "premeditation": True,
            "lethal_weapon": True
        },
        outcome_directive="Uphold capital Murder (IPC 302 / BNS 103) conviction without exception mitigation."
    ),
    PrecedentRatio(
        citation="(2003) 3 SCC 528",
        case_name="Ghapoo Yadav and Others v. State of M.P.",
        year=2003,
        court="Supreme Court of India",
        statutory_provisions=["IPC_304_PART2_BNS_105"],
        legal_doctrine="Four Cumulative Ingredients of Section 300 Exception 4",
        ratio_decidendi="To invoke Exception 4 to Section 300, four conditions must simultaneously exist: (1) sudden fight, (2) absence of premeditation, (3) heat of passion upon sudden quarrel, and (4) offender did not take undue advantage or act in a cruel or unusual manner.",
        trigger_criteria={
            "sudden_fight": True,
            "heat_of_passion": True,
            "no_premeditation": True,
            "single_or_proportional_blow": True
        },
        outcome_directive="Mandatory application of IPC 304 Part II / BNS 105; statutory ceiling 10 years imprisonment."
    ),
    PrecedentRatio(
        citation="(2014) 8 SCC 273",
        case_name="Arnesh Kumar v. State of Bihar & Anr.",
        year=2014,
        court="Supreme Court of India",
        statutory_provisions=["IPC_498A_BNS_85", "IPC_420_BNS_318", "IPC_406_BNS_316"],
        legal_doctrine="Arrest Procedural Safeguards (Under 7 Years Ceiling)",
        ratio_decidendi="For offences punishable with imprisonment up to 7 years, police officer shall not automatically arrest accused without recording specific reasons under Section 41 CrPC (s.35 BNSS). Notice of appearance u/s 41A CrPC is mandatory.",
        trigger_criteria={
            "penalty_tier": "Tier 2: 3 to 7 Years (Arnesh Kumar Threshold / Medium Penalty)",
            "arrest_recommended": True
        },
        outcome_directive="Flag mandatory procedural compliance check and Arnesh Kumar protective threshold."
    ),
    PrecedentRatio(
        citation="1992 Supp (1) SCC 335",
        case_name="State of Haryana and Ors. v. Ch. Bhajan Lal & Ors.",
        year=1992,
        court="Supreme Court of India",
        statutory_provisions=["SECTION_482_CRPC_528_BNSS"],
        legal_doctrine="Seven Categories for Quashing FIR / Criminal Proceedings",
        ratio_decidendi="Where allegations in FIR even if taken at face value do not disclose cognizable offence, or where dispute is manifestly civil in nature attended with malafide, High Court must exercise inherent powers u/s 482 CrPC / 528 BNSS to prevent abuse of judicial process.",
        trigger_criteria={
            "is_purely_civil": True,
            "contract_dispute": True,
            "violence": False
        },
        outcome_directive="Apply Section 482 CrPC / Section 528 BNSS quashing recommendation; zero criminal penal term."
    )
]

def find_applicable_precedents(fact_dict: Dict[str, Any]) -> List[PrecedentRatio]:
    applicable = []
    for prec in LANDMARK_PRECEDENTS:
        match = True
        for k, v in prec.trigger_criteria.items():
            if fact_dict.get(k) != v:
                match = False
                break
        if match:
            applicable.append(prec)
    return applicable
