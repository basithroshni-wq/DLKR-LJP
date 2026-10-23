"""
========================================================================================
DLKR-LJP V5: Comprehensive 27-Case Statutory Benchmark Test Suite
========================================================================================
Validates multi-task legal judgment prediction across the full Indian penal spectrum:
- IPC 379 / BNS 303 (Simple Theft)
- IPC 392 / BNS 309 (Armed Robbery)
- IPC 302 / BNS 103 (Premeditated Murder)
- IPC 304 / BNS 105 (Culpable Homicide / Sudden Fight Exception 4)
- IPC 304A / BNS 106 (Fatal Vehicular Rash & Negligent Act)
- IPC 307 / BNS 109 (Attempt to Murder)
- IPC 326 / BNS 118(2) (Grievous Hurt with Dangerous Weapon)
- IPC 324 / BNS 118(1) (Simple Hurt with Dangerous Weapon)
- IPC 323 / BNS 115 (Simple Hurt Unarmed)
- IPC 420 / BNS 318 (Cheating & Financial Fraud)
- IPC 468 / BNS 336 (Forgery for Purpose of Cheating)
- IPC 467 / BNS 338 (Forgery of Valuable Security / Will)
- IPC 471 / BNS 340 (Using Forged Document as Genuine)
- IPC 406 / BNS 316 (Criminal Breach of Trust)
- IPC 304B / BNS 80 (Dowry Death)
- IPC 498A / BNS 85 (Matrimonial Cruelty)
- IPC 376 / BNS 64 (Rape / Sexual Assault)
- IPC 506 / BNS 351 (Criminal Intimidation)
- Section 482 CrPC / s.528 BNSS (Commercial Civil Dispute Quashing)
- Out of Domain Abstention Gate (Non-Cognizable Narrative)
========================================================================================
"""

import sys
import time
from dlkr_ljp_v5_final import predict_case

test_suite = [
    # 1-2. Theft (IPC 379 / BNS 303)
    ("Theft (Pickpocketing)", "A pickpocket secretly stole a commuter wallet containing Rs 8,000 in cash from his trousers pocket on a crowded suburban train without the commuter realizing.", "IPC_379_BNS_303"),
    ("Theft (Bicycle)", "The accused stole a bicycle parked outside the railway station without the owner consent and hid it in his shed. No weapon or force was involved.", "IPC_379_BNS_303"),

    # 3-4. Robbery (IPC 392 / BNS 309)
    ("Robbery at Knifepoint", "The accused intercepted a pedestrian on a deserted street at night, held a sharp dagger to his throat, and forcibly took his mobile phone and gold chain under threat of death.", "IPC_392_BNS_309"),
    ("Robbery with Assault", "Two men cornered a shopkeeper, brandished a knife, slapped and shoved him to the floor, and forcibly took the day cash collection from his cash drawer.", "IPC_392_BNS_309"),

    # 5-7. Premeditated Murder (IPC 302 / BNS 103)
    ("Premeditated Murder (Axe)", "Due to long-standing land rivalry, the accused lay in wait behind a tree armed with an axe and ambushed the victim, striking multiple fatal blows to his head and neck until he died.", "IPC_302_BNS_103"),
    ("Murder with Firearm", "The assailant brought an illegal country-made pistol to the victim shop and shot him in the chest at point-blank range, killing him on the spot.", "IPC_302_BNS_103"),
    ("Murder by Poison", "The accused deliberately laced the victim tea with a lethal dose of cyanide poison with premeditated intent to eliminate him. The victim died within an hour.", "IPC_302_BNS_103"),

    # 8-9. Culpable Homicide (IPC 304 / BNS 105)
    ("Sudden Fight Homicide", "During an impromptu heated quarrel over water sharing, the accused picked up a wooden club lying on the ground and struck the victim once on the head in the heat of passion without premeditation. The victim succumbed to head injuries.", "IPC_304_PART"),
    ("Mutual Scuffle Homicide", "A sudden physical altercation erupted in a tea stall between two labourers. In the sudden brawl without any prior planning, the accused shoved the victim hard against a sharp stone, causing fatal head trauma.", "IPC_304_PART"),

    # 10-11. Motor Accidents / Rash Driving (IPC 304A / BNS 106)
    ("Fatal Truck Accident", "A speeding truck driver driving rashly on the wrong side of the national highway rammed into a motorcycle, causing the rider to die on the spot.", "IPC_304A_BNS_106"),
    ("Hit and Run Death", "The driver drove his car recklessly at 100 km/h in a school zone, hit a pedestrian crossing the road, and fled the scene. The pedestrian died of injuries.", "IPC_304A_BNS_106"),

    # 12-13. Attempt to Murder (IPC 307 / BNS 109)
    ("Attempt to Murder (Pistol)", "The accused aimed a pistol at the victim chest shouting I will finish you and fired two gunshots. The victim received emergency surgical treatment and survived.", "IPC_307_BNS_109"),
    ("Attempt to Murder (Stabbing)", "With intent to kill, the assailant stabbed the victim repeatedly in the abdomen and neck with a sharp knife. The victim was rushed to the ICU and survived.", "IPC_307_BNS_109"),

    # 14-16. Bodily Hurt (IPC 326, 324, 323)
    ("Grievous Hurt with Iron Rod", "The accused attacked the complainant with an iron rod, fracturing both forearm bones and causing a compound fracture of the jaw bone.", "IPC_326_BNS_118_2"),
    ("Simple Hurt with Weapon", "During a fight, the accused slashed the victim arm with a small folding knife, causing a superficial bleeding laceration requiring 4 stitches.", "IPC_324_BNS_118_1"),
    ("Simple Hurt Unarmed", "In a dispute over parking, the accused slapped and punched the complainant multiple times, causing facial swelling and minor bruises.", "IPC_323_BNS_115"),

    # 17-20. Fraud & Forgery (IPC 420, 468, 467, 471)
    ("Cheating (Financial Scheme)", "The accused deceived depositors by promising 200 percent returns in 60 days through a fictitious real estate fund and dishonestly induced them to transfer Rs 40 lakhs before absconding.", "IPC_420_BNS_318"),
    ("Job Scam with Fake Letter", "The accused took Rs 6 lakh from the complainant promising a railway job, handed over a forged appointment letter and fake badge, and vanished.", "IPC_468_BNS_336"),
    ("Forged Will (Security)", "The nephew forged the signature of his deceased uncle on a fake last will and testament to claim exclusive ownership of ancestral land.", "IPC_467_BNS_338"),
    ("Using Forged Document", "The applicant knowingly submitted a fake fabricated mark sheet and counterfeit university certificate to gain employment.", "IPC_471_BNS_340"),

    # 21. Breach of Trust (IPC 406 / BNS 316)
    ("Criminal Breach of Trust", "The company cashier was entrusted with Rs 15 lakhs for depositing into the company bank account, but he dishonestly misappropriated and converted the cash for his personal use.", "IPC_406_BNS_316"),

    # 22-23. Matrimonial Offences (IPC 304B, 498A)
    ("Dowry Death", "Married for two years, the young bride was found dead with severe burn injuries following relentless physical torture by her husband and mother-in-law demanding a gold necklace and Rs 5 lakh dowry.", "IPC_304B_BNS_80"),
    ("Matrimonial Cruelty", "The husband and his relatives subjected the wife to constant physical assaults, verbal abuse, and mental torture demanding a car as additional dowry.", "IPC_498A_BNS_85"),

    # 24. Sexual Offences (IPC 376 / BNS 64)
    ("Rape / Sexual Assault", "The accused overpowered the woman in a secluded building and committed non-consensual penetrative sexual intercourse violating her bodily autonomy.", "IPC_376_BNS_64"),

    # 25. Criminal Intimidation (IPC 506 / BNS 351)
    ("Criminal Intimidation (Death Threat)", "The accused visited the merchant shop with associates and threatened to kill him and blow up his shop if he did not pay Rs 2 lakh protection money.", "IPC_506_BNS_351"),

    # 26. Civil Dispute Quashing (s.482 CrPC / s.528 BNSS)
    ("Commercial Contract Dispute", "Dispute over delayed delivery of industrial machinery and unpaid supply invoices under a commercial sales agreement. Both parties had agreed to an arbitration clause. No criminal deception occurred at inception.", "NONE_CIVIL_DISPUTE"),

    # 27. Out of Domain Abstention Gate
    ("Out of Domain Recipe", "To prepare traditional vegetable biryani, soak basmati rice for 30 minutes, saute chopped onions and green chilies in ghee, and simmer with seasonal vegetables and aromatic spices.", "NONE_OUT_OF_DOMAIN")
]

def run_all_cases():
    print("=" * 82)
    print("      DLKR-LJP V5: 27-CASE STATUTORY BENCHMARK VALIDATION RUNNER")
    print("=" * 82)
    sys.stdout.flush()

    passed_count = 0
    results = []
    t0 = time.time()

    for idx, (title, narrative, expected_code) in enumerate(test_suite, 1):
        case_t0 = time.time()
        res = predict_case(narrative)
        preds = res["predictions"]
        reason = res["legal_reasoning"]
        top_art = preds["head1_articles"][0]["article_id"]
        disp_name = preds["head1_articles"][0].get("display_name", top_art)
        top_chg = preds["head2_charges"][0]["charge_name"]
        top_tier = preds["primary_penalty"]["tier"]
        engine = reason.get("engine", "DLKR-LJP Pretrained Legal Reasoner")
        is_ood = preds.get("is_out_of_domain", False)
        is_exculp = preds.get("is_exculpatory", False)

        matched = False
        if expected_code == "NONE_OUT_OF_DOMAIN":
            matched = is_ood or ("OUT_OF_DOMAIN" in top_art)
        elif expected_code == "NONE_CIVIL_DISPUTE":
            matched = is_exculp or ("CIVIL" in top_art)
        else:
            matched = (expected_code in top_art)

        if matched:
            passed_count += 1
            status = "PASS"
        else:
            status = "FAIL"

        elapsed = time.time() - case_t0
        results.append({
            "idx": idx,
            "title": title,
            "expected": expected_code,
            "predicted_art": top_art,
            "disp_name": disp_name,
            "charge": top_chg,
            "tier": top_tier,
            "engine": engine,
            "status": status,
            "time": elapsed
        })
        print(f"[{idx:02d}/27] {status} ({elapsed:.1f}s) | {title[:28]:28s} | Expected: {expected_code[:16]:16s} | Got: {top_art[:20]:20s} | {top_tier[:25]}")
        sys.stdout.flush()

    total_time = time.time() - t0
    acc_pct = (passed_count / len(test_suite)) * 100
    print("=" * 82)
    print(f"TOTAL BENCHMARK RESULT: {passed_count}/{len(test_suite)} PASSED ({acc_pct:.1f}%) in {total_time:.1f}s")
    print("=" * 82)
    return results

if __name__ == "__main__":
    run_all_cases()
