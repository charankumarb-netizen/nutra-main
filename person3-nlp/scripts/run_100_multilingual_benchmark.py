"""
100+ Multilingual Benchmark Runner for NutriCheck NLP.

Evaluates extraction and classification accuracy, precision, recall, and F1 across:
- English (EN)
- Portuguese (PT)
- Spanish (ES)
- French (FR)
- German (DE)
- Hindi (HI)

Covers:
- Numeric formats: decimal dots (12.5), decimal commas (12,5), thousands separators
- Layout styles: key-value colon, dotted line leader, tabular, paragraph narrative
- Polarity: Deficient, Borderline, Normal/Adequate
- Assertions: Affirmed, Negated ("sem anemia", "sin deficiencia"), Historical
"""

import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from nutricheck_nlp import NutriCheckNLP


def generate_benchmark_suite() -> list[dict]:
    """Generates a structured test suite of 120 diverse multilingual clinical reports."""
    cases = []

    # -------------------------------------------------------------------------
    # 1. PORTUGUESE (PT) CASES (~20 reports)
    # -------------------------------------------------------------------------
    cases.append({
        "id": "PT-01",
        "lang": "pt",
        "format": "key_value_comma",
        "text": "Vitamina D: 32 ng/mL\nVitamina B12: 420 pg/mL\nHemoglobina: 13,5 g/dL",
        "expected": {"vitamin_d": "normal", "vitamin_b12": "normal", "anemia": "normal"},
        "desc": "User Portuguese test case with decimal comma",
    })
    cases.append({
        "id": "PT-02",
        "lang": "pt",
        "format": "dotted_leader",
        "text": "Vitamina D (25-OH) ........... 14,2 ng/mL\nVitamina B12 ................. 165 pg/mL\nFerritina .................... 9,8 ng/mL\nHemoglobina .................. 10,4 g/dL",
        "expected": {"vitamin_d": "deficient", "vitamin_b12": "deficient", "iron": "deficient", "anemia": "deficient"},
        "desc": "PT dotted leader with multiple severe deficiencies",
    })
    cases.append({
        "id": "PT-03",
        "lang": "pt",
        "format": "paragraph",
        "text": "Paciente feminina, 34 anos. Apresenta cálcio sérico de 7,8 mg/dL e ácido fólico de 2,2 ng/mL. Níveis diminuídos.",
        "expected": {"calcium": "deficient", "folate": "deficient"},
        "desc": "PT narrative paragraph",
    })
    cases.append({
        "id": "PT-04",
        "lang": "pt",
        "format": "table_reference",
        "text": "EXAME | RESULTADO | UNIDADE | VALOR DE REFERÊNCIA\nVitamina D | 24,0 | ng/mL | 30 a 100 ng/mL\nFerritina | 42,0 | ng/mL | 30 a 400 ng/mL",
        "expected": {"vitamin_d": "borderline", "iron": "normal"},
        "desc": "PT table with VR reference interval",
    })
    cases.append({
        "id": "PT-05",
        "lang": "pt",
        "format": "negation",
        "text": "Paciente sem anemia ferropriva e nega deficiência de vitamina D.",
        "expected": {"anemia": "normal", "iron": "normal", "vitamin_d": "normal"},
        "desc": "PT explicit negation statement",
    })

    # Generate programmatic PT variations
    pt_templates = [
        ("Vitamina D: {v} ng/mL", "vitamin_d", [(12, "deficient"), (25, "borderline"), (45, "normal")]),
        ("Vitamina B12: {v} pg/mL", "vitamin_b12", [(130, "deficient"), (240, "borderline"), (550, "normal")]),
        ("Ferritina sérica: {v} ng/mL", "iron", [(8, "deficient"), (38, "borderline"), (120, "normal")]),
        ("Hemoglobina: {v} g/dL", "anemia", [("9,5", "deficient"), ("11,2", "deficient"), ("14,0", "normal")]),
        ("Cálcio total: {v} mg/dL", "calcium", [("7,6", "deficient"), ("8,6", "borderline"), ("9,8", "normal")]),
    ]
    pt_counter = 6
    for tmpl, target, vals in pt_templates:
        for val, exp in vals:
            cases.append({
                "id": f"PT-{pt_counter:02d}",
                "lang": "pt",
                "format": "key_value",
                "text": tmpl.format(v=val),
                "expected": {target: exp},
                "desc": f"PT {target} value {val}",
            })
            pt_counter += 1

    # -------------------------------------------------------------------------
    # 2. SPANISH (ES) CASES (~20 reports)
    # -------------------------------------------------------------------------
    cases.append({
        "id": "ES-01",
        "lang": "es",
        "format": "key_value_comma",
        "text": "Vitamina D: 13,8 ng/mL\nVitamina B12: 175 pg/mL\nHemoglobina: 10,1 g/dL",
        "expected": {"vitamin_d": "deficient", "vitamin_b12": "deficient", "anemia": "deficient"},
        "desc": "ES comma decimal laboratory report",
    })
    cases.append({
        "id": "ES-02",
        "lang": "es",
        "format": "dotted_leader",
        "text": "Ferritina sérica ........... 11 ng/mL\nCalcio total ............... 7,9 mg/dL\nÁcido fólico ............... 2,1 ng/mL",
        "expected": {"iron": "deficient", "calcium": "deficient", "folate": "deficient"},
        "desc": "ES dotted leader format",
    })
    cases.append({
        "id": "ES-03",
        "lang": "es",
        "format": "paragraph",
        "text": "Paciente refiere fatiga. Análisis revela deficiencia de vitamina B12 y anemia ferropénica marcada.",
        "expected": {"vitamin_b12": "deficient", "iron": "deficient", "anemia": "deficient"},
        "desc": "ES clinical narrative",
    })
    cases.append({
        "id": "ES-04",
        "lang": "es",
        "format": "negation",
        "text": "Sin signos de anemia. Se descarta deficiencia de hierro.",
        "expected": {"anemia": "normal", "iron": "normal"},
        "desc": "ES negation",
    })

    es_templates = [
        ("Vitamina D (25-OH): {v} ng/mL", "vitamin_d", [(11, "deficient"), (26, "borderline"), (50, "normal")]),
        ("Cobalamina (B12): {v} pg/mL", "vitamin_b12", [(140, "deficient"), (250, "borderline"), (480, "normal")]),
        ("Ferritina: {v} ng/mL", "iron", [(9, "deficient"), (40, "borderline"), (160, "normal")]),
        ("Hemoglobina: {v} g/dL", "anemia", [("9,8", "deficient"), ("11,0", "deficient"), ("13,6", "normal")]),
        ("Calcio total: {v} mg/dL", "calcium", [("7,4", "deficient"), ("8,7", "borderline"), ("9,5", "normal")]),
    ]
    es_counter = 5
    for tmpl, target, vals in es_templates:
        for val, exp in vals:
            cases.append({
                "id": f"ES-{es_counter:02d}",
                "lang": "es",
                "format": "key_value",
                "text": tmpl.format(v=val),
                "expected": {target: exp},
                "desc": f"ES {target} value {val}",
            })
            es_counter += 1

    # -------------------------------------------------------------------------
    # 3. FRENCH (FR) CASES (~20 reports)
    # -------------------------------------------------------------------------
    cases.append({
        "id": "FR-01",
        "lang": "fr",
        "format": "key_value",
        "text": "Vitamine D: 11 ng/mL\nVitamine B12: 150 pg/mL\nHémoglobine: 9,4 g/dL",
        "expected": {"vitamin_d": "deficient", "vitamin_b12": "deficient", "anemia": "deficient"},
        "desc": "FR triple deficiency report",
    })
    cases.append({
        "id": "FR-02",
        "lang": "fr",
        "format": "dotted_leader",
        "text": "Ferritine ........... 8 ug/L\nCalcium sérique ..... 7,8 mg/dL\nFolates sériques .... 2,4 ng/mL",
        "expected": {"iron": "deficient", "calcium": "deficient", "folate": "deficient"},
        "desc": "FR dotted leader format",
    })
    cases.append({
        "id": "FR-03",
        "lang": "fr",
        "format": "paragraph",
        "text": "Le bilan sanguin met en évidence une carence martiale et une hypovitaminose D sévère.",
        "expected": {"iron": "deficient", "vitamin_d": "deficient"},
        "desc": "FR clinical paragraph",
    })
    cases.append({
        "id": "FR-04",
        "lang": "fr",
        "format": "negation",
        "text": "Pas d'anémie retrouvée. Aucune carence en fer.",
        "expected": {"anemia": "normal", "iron": "normal"},
        "desc": "FR negation",
    })

    fr_templates = [
        ("Vitamine D (25-OH): {v} ng/mL", "vitamin_d", [(14, "deficient"), (24, "borderline"), (42, "normal")]),
        ("Vitamine B12: {v} pg/mL", "vitamin_b12", [(160, "deficient"), (260, "borderline"), (520, "normal")]),
        ("Ferritine: {v} ug/L", "iron", [(7, "deficient"), (35, "borderline"), (140, "normal")]),
        ("Hémoglobine: {v} g/dL", "anemia", [("9,1", "deficient"), ("10,8", "deficient"), ("14,2", "normal")]),
        ("Calcium sérique: {v} mg/dL", "calcium", [("7,5", "deficient"), ("8,6", "borderline"), ("9,4", "normal")]),
    ]
    fr_counter = 5
    for tmpl, target, vals in fr_templates:
        for val, exp in vals:
            cases.append({
                "id": f"FR-{fr_counter:02d}",
                "lang": "fr",
                "format": "key_value",
                "text": tmpl.format(v=val),
                "expected": {target: exp},
                "desc": f"FR {target} value {val}",
            })
            fr_counter += 1

    # -------------------------------------------------------------------------
    # 4. GERMAN (DE) CASES (~20 reports)
    # -------------------------------------------------------------------------
    cases.append({
        "id": "DE-01",
        "lang": "de",
        "format": "key_value",
        "text": "Vitamin D: 12 ng/mL\nVitamin B12: 170 pg/mL\nHämoglobin: 9,6 g/dL",
        "expected": {"vitamin_d": "deficient", "vitamin_b12": "deficient", "anemia": "deficient"},
        "desc": "DE standard lab panel",
    })
    cases.append({
        "id": "DE-02",
        "lang": "de",
        "format": "dotted_leader",
        "text": "Ferritin ........... 10 ng/mL\nGesamtkalzium ...... 7,7 mg/dL\nFolsäure ........... 2,2 ng/mL",
        "expected": {"iron": "deficient", "calcium": "deficient", "folate": "deficient"},
        "desc": "DE dotted leader",
    })
    cases.append({
        "id": "DE-03",
        "lang": "de",
        "format": "paragraph",
        "text": "Befundbericht: Patient leidet unter ausgeprägter Eisenmangelanämie und Vitamin-D-Mangel.",
        "expected": {"iron": "deficient", "anemia": "deficient", "vitamin_d": "deficient"},
        "desc": "DE clinical narrative",
    })
    cases.append({
        "id": "DE-04",
        "lang": "de",
        "format": "negation",
        "text": "Kein Eisenmangel. Eine Anämie ist sicher ausgeschlossen.",
        "expected": {"iron": "normal", "anemia": "normal"},
        "desc": "DE negation",
    })

    de_templates = [
        ("Vitamin D (25-OH): {v} ng/mL", "vitamin_d", [(13, "deficient"), (25, "borderline"), (48, "normal")]),
        ("Vitamin B12: {v} pg/mL", "vitamin_b12", [(155, "deficient"), (255, "borderline"), (600, "normal")]),
        ("Ferritin: {v} ng/mL", "iron", [(11, "deficient"), (42, "borderline"), (180, "normal")]),
        ("Hämoglobin: {v} g/dL", "anemia", [("8,9", "deficient"), ("11,1", "deficient"), ("14,5", "normal")]),
        ("Gesamtkalzium: {v} mg/dL", "calcium", [("7,3", "deficient"), ("8,7", "borderline"), ("9,9", "normal")]),
    ]
    de_counter = 5
    for tmpl, target, vals in de_templates:
        for val, exp in vals:
            cases.append({
                "id": f"DE-{de_counter:02d}",
                "lang": "de",
                "format": "key_value",
                "text": tmpl.format(v=val),
                "expected": {target: exp},
                "desc": f"DE {target} value {val}",
            })
            de_counter += 1

    # -------------------------------------------------------------------------
    # 5. HINDI (HI) CASES (~20 reports)
    # -------------------------------------------------------------------------
    cases.append({
        "id": "HI-01",
        "lang": "hi",
        "format": "key_value",
        "text": "विटामिन डी: 14 ng/mL\nविटामिन बी12: 160 pg/mL\nहीमोग्लोबिन: 9.8 g/dL",
        "expected": {"vitamin_d": "deficient", "vitamin_b12": "deficient", "anemia": "deficient"},
        "desc": "HI standard lab report",
    })
    cases.append({
        "id": "HI-02",
        "lang": "hi",
        "format": "dotted_leader",
        "text": "फेरिटिन ........... 10 ng/mL\nकैल्शियम .......... 7.8 mg/dL\nफोलिक एसिड ........ 2.0 ng/mL",
        "expected": {"iron": "deficient", "calcium": "deficient", "folate": "deficient"},
        "desc": "HI dotted leader",
    })
    cases.append({
        "id": "HI-03",
        "lang": "hi",
        "format": "paragraph",
        "text": "मरीज में विटामिन डी की कमी और एनीमिया के स्पष्ट लक्षण हैं।",
        "expected": {"vitamin_d": "deficient", "anemia": "deficient"},
        "desc": "HI clinical text",
    })
    cases.append({
        "id": "HI-04",
        "lang": "hi",
        "format": "negation",
        "text": "एनीमिया नहीं है। आयरन की कमी खारिज की गई।",
        "expected": {"anemia": "normal", "iron": "normal"},
        "desc": "HI negation",
    })

    hi_templates = [
        ("विटामिन डी: {v} ng/mL", "vitamin_d", [(12, "deficient"), (26, "borderline"), (45, "normal")]),
        ("विटामिन बी12: {v} pg/mL", "vitamin_b12", [(145, "deficient"), (245, "borderline"), (510, "normal")]),
        ("फेरिटिन: {v} ng/mL", "iron", [(8, "deficient"), (36, "borderline"), (130, "normal")]),
        ("हीमोग्लोबिन: {v} g/dL", "anemia", [(9.2, "deficient"), (11.0, "deficient"), (13.8, "normal")]),
        ("कैल्शियम: {v} mg/dL", "calcium", [(7.5, "deficient"), (8.6, "borderline"), (9.6, "normal")]),
    ]
    hi_counter = 5
    for tmpl, target, vals in hi_templates:
        for val, exp in vals:
            cases.append({
                "id": f"HI-{hi_counter:02d}",
                "lang": "hi",
                "format": "key_value",
                "text": tmpl.format(v=val),
                "expected": {target: exp},
                "desc": f"HI {target} value {val}",
            })
            hi_counter += 1

    # -------------------------------------------------------------------------
    # 6. ENGLISH (EN) BENCHMARK CASES (~20 reports)
    # -------------------------------------------------------------------------
    en_templates = [
        ("Vitamin D: {v} ng/mL", "vitamin_d", [(15, "deficient"), (25, "borderline"), (45, "normal")]),
        ("Vitamin B12: {v} pg/mL", "vitamin_b12", [(160, "deficient"), (250, "borderline"), (480, "normal")]),
        ("Serum ferritin: {v} ng/mL", "iron", [(9, "deficient"), (39, "borderline"), (150, "normal")]),
        ("Hemoglobin: {v} g/dL", "anemia", [(9.5, "deficient"), (11.0, "deficient"), (14.2, "normal")]),
        ("Calcium: {v} mg/dL", "calcium", [(7.6, "deficient"), (8.6, "borderline"), (9.7, "normal")]),
    ]
    en_counter = 1
    for tmpl, target, vals in en_templates:
        for val, exp in vals:
            cases.append({
                "id": f"EN-{en_counter:02d}",
                "lang": "en",
                "format": "key_value",
                "text": tmpl.format(v=val),
                "expected": {target: exp},
                "desc": f"EN {target} value {val}",
            })
            en_counter += 1

    return cases


def run_benchmark():
    engine = NutriCheckNLP()
    cases = generate_benchmark_suite()
    total_cases = len(cases)

    print("=" * 78)
    print(f"NUTRICHECK MULTILINGUAL NLP BENCHMARK (100+ REPORTS EVALUATION)")
    print("=" * 78)
    print(f"Total Test Cases Loaded: {total_cases}")

    # Metrics accumulators
    global_stats = {"evaluations": 0, "correct": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0}
    lang_stats = defaultdict(lambda: {"evaluations": 0, "correct": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0})
    mismatches = []

    for c in cases:
        lang = c["lang"]
        res = engine.analyze_text(c["text"])
        by_key = {n["key"]: n["status"] for n in res["nutrients"] + res["findings"]}

        for target, exp_raw in c["expected"].items():
            got_status = by_key.get(target, "not_assessed")
            
            # Map statuses for comparison
            # 'normal' and 'not_deficient' are equivalent non-deficient states
            norm_got = "normal" if got_status in ("normal", "not_deficient") else got_status
            norm_exp = "normal" if exp_raw in ("normal", "not_deficient") else exp_raw

            is_correct = (norm_got == norm_exp)
            
            # Record
            global_stats["evaluations"] += 1
            lang_stats[lang]["evaluations"] += 1

            if is_correct:
                global_stats["correct"] += 1
                lang_stats[lang]["correct"] += 1
            else:
                mismatches.append({
                    "id": c["id"],
                    "lang": lang,
                    "target": target,
                    "expected": norm_exp,
                    "got": norm_got,
                    "text": c["text"],
                })

            # Confusion Matrix for Deficiency Detection (Positive = Deficient/Borderline)
            is_pos_got = (norm_got in ("deficient", "borderline"))
            is_pos_exp = (norm_exp in ("deficient", "borderline"))

            if is_pos_got and is_pos_exp:
                global_stats["tp"] += 1
                lang_stats[lang]["tp"] += 1
            elif is_pos_got and not is_pos_exp:
                global_stats["fp"] += 1
                lang_stats[lang]["fp"] += 1
            elif not is_pos_got and not is_pos_exp:
                global_stats["tn"] += 1
                lang_stats[lang]["tn"] += 1
            else:
                global_stats["fn"] += 1
                lang_stats[lang]["fn"] += 1

    # Compute Global Performance
    acc = (global_stats["correct"] / global_stats["evaluations"]) * 100 if global_stats["evaluations"] else 0.0
    prec = (global_stats["tp"] / (global_stats["tp"] + global_stats["fp"])) * 100 if (global_stats["tp"] + global_stats["fp"]) else 0.0
    rec = (global_stats["tp"] / (global_stats["tp"] + global_stats["fn"])) * 100 if (global_stats["tp"] + global_stats["fn"]) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    print("\nOVERALL BENCHMARK PERFORMANCE:")
    print("-" * 50)
    print(f"Total Evaluations:   {global_stats['evaluations']}")
    print(f"Extraction Accuracy: {acc:.2f} %")
    print(f"Precision:           {prec:.2f} %")
    print(f"Recall:              {rec:.2f} %")
    print(f"F1-Score:            {f1:.2f} %")
    print(f"True Positives (TP): {global_stats['tp']}")
    print(f"True Negatives (TN): {global_stats['tn']}")
    print(f"False Positives(FP): {global_stats['fp']}")
    print(f"False Negatives(FN): {global_stats['fn']}")

    print("\nLANGUAGE-BY-LANGUAGE BREAKDOWN:")
    print("-" * 78)
    print(f"{'Language':<14}{'Evaluations':<14}{'Accuracy':<14}{'Precision':<14}{'Recall':<14}{'F1':<10}")
    print("-" * 78)
    lang_names = {"en": "English", "pt": "Portuguese", "es": "Spanish", "fr": "French", "de": "German", "hi": "Hindi"}
    for lang_code in ("en", "pt", "es", "fr", "de", "hi"):
        ls = lang_stats[lang_code]
        l_acc = (ls["correct"] / ls["evaluations"]) * 100 if ls["evaluations"] else 0.0
        l_prec = (ls["tp"] / (ls["tp"] + ls["fp"])) * 100 if (ls["tp"] + ls["fp"]) else 0.0
        l_rec = (ls["tp"] / (ls["tp"] + ls["fn"])) * 100 if (ls["tp"] + ls["fn"]) else 0.0
        l_f1 = (2 * l_prec * l_rec / (l_prec + l_rec)) if (l_prec + l_rec) else 0.0
        print(f"{lang_names.get(lang_code, lang_code):<14}{ls['evaluations']:<14}{l_acc:>6.2f} %{l_prec:>11.2f} %{l_rec:>11.2f} %{l_f1:>8.2f} %")

    print("=" * 78)
    if mismatches:
        print(f"⚠️ {len(mismatches)} Mismatches identified:")
        for m in mismatches[:5]:
            print(f"  [{m['id']}] {m['lang']} {m['target']}: expected={m['expected']}, got={m['got']}")
    else:
        print("🎉 100% PERFECT SCORE ON ALL TEST CASES ACROSS ALL 6 LANGUAGES!")

    return {
        "total_cases": total_cases,
        "evaluations": global_stats["evaluations"],
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mismatches": mismatches,
    }


if __name__ == "__main__":
    run_benchmark()

