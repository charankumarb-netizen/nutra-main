"""
Automated Test Suite for Portuguese Laboratory Value Extraction & NLP Pipeline Fixes.

Covers:
1. Exact user test report (Pedro Santos) with:
   - Vitamina B12: 100 pg/mL -> DEFICIENT
   - Hemoglobina: 10,8 g/dL -> LOW / DEFICIENT (Male cutoff < 13.0)
   - VCM: 110 fL -> HIGH (elevated CBC measurement, macrocytosis)
   - Iron, Vitamin D, Calcium, Folate -> Not detected
   - Summary: Deficient/Low: 2, Borderline: 0, Normal: 0
2. 20+ diverse Portuguese reports with various layouts, formats, and patient details.
3. English reports to ensure backwards compatibility and no regression.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from nutricheck_nlp import NutriCheckNLP


def test_pedro_santos_report():
    """Verify the exact user test report."""
    report = """
Paciente: Pedro Santos

Idade: 38 anos | Sexo: Masculino

Sintomas: Formigamento nas mãos e nos pés, perda de memória, fraqueza, cansaço e dificuldade de equilíbrio.

Resultados Laboratoriais:
- Vitamina B12: 100 pg/mL
- Hemoglobina: 10,8 g/dL
- VCM: 110 fL

Observação Clínica: Os níveis baixos de vitamina B12 podem estar relacionados aos sintomas neurológicos.
"""
    engine = NutriCheckNLP()
    res = engine.analyze_text(report)
    by_key = {n["key"]: n for n in res["nutrients"] + res["findings"]}
    labs = res["labs"]

    # 1. Total extracted labs must be 3: Vitamin B12, Hemoglobin, MCV
    assert len(labs) == 3, f"Expected 3 labs, got {len(labs)}: {[l['label'] for l in labs]}"
    labs_by_analyte = {l["analyte"]: l for l in labs}

    # 2. Vitamin B12 extraction
    assert "vitamin_b12" in labs_by_analyte
    b12_lab = labs_by_analyte["vitamin_b12"]
    assert b12_lab["value"] == 100.0
    assert b12_lab["unit"] == "pg/ml"
    assert by_key["vitamin_b12"]["status"] == "deficient"

    # 3. Hemoglobin extraction
    assert "hemoglobin" in labs_by_analyte
    hb_lab = labs_by_analyte["hemoglobin"]
    assert hb_lab["value"] == 10.8, f"Expected 10.8, got {hb_lab['value']}"
    assert hb_lab["unit"] == "g/dl"
    # Sex is Masculino -> cutoff 13.0 -> 10.8 is deficient
    assert res["patient"]["sex"] == "M", f"Expected sex M, got {res['patient']['sex']}"
    assert by_key["anemia"]["status"] == "deficient"

    # 4. MCV (VCM) extraction
    assert "mcv" in labs_by_analyte
    mcv_lab = labs_by_analyte["mcv"]
    assert mcv_lab["value"] == 110.0
    assert mcv_lab["unit"] == "fl"
    assert mcv_lab["status"] == "high", f"MCV 110 fL should be classified as high/elevated, got {mcv_lab['status']}"

    # 5. Missing nutrients must NOT be classified as normal
    for missing_key in ("iron", "vitamin_d", "calcium", "folate"):
        assert by_key[missing_key]["status"] == "not_assessed", f"{missing_key} should be not_assessed, got {by_key[missing_key]['status']}"

    # 6. Summary counts: Deficient/Low: 2, Borderline: 0, Normal: 0
    all_screened = res["nutrients"] + res["findings"]
    def_count = sum(1 for n in all_screened if n.get("status") == "deficient")
    bor_count = sum(1 for n in all_screened if n.get("status") == "borderline")
    norm_count = sum(1 for n in all_screened if n.get("status") in ("normal", "not_deficient"))

    assert def_count == 2, f"Expected 2 deficient, got {def_count}"
    assert bor_count == 0, f"Expected 0 borderline, got {bor_count}"
    assert norm_count == 0, f"Expected 0 normal, got {norm_count}"

    print("✓ Test Pedro Santos Report: ALL ASSERTIONS PASSED!")


def test_20_diverse_portuguese_reports():
    """Test 20+ diverse Portuguese medical reports with varied patients, formats, and biomarkers."""
    engine = NutriCheckNLP()

    test_reports = [
        # 1. Comma decimal in tabular layout
        (
            "LAUDO LABORATORIAL\nPaciente: Maria Oliveira | Sexo: Feminino | Idade: 52 anos\n"
            "Vitamina D (25-OH) ........... 13,5 ng/mL\nFerritina .................... 8,4 ng/mL\n"
            "Hemoglobina .................. 11,2 g/dL",
            {"vitamin_d": "deficient", "iron": "deficient", "anemia": "deficient"}
        ),
        # 2. Key-value list with borderline vit D and normal B12
        (
            "Exames de Rotina - João Pereira\n- Vitamina D: 24,5 ng/mL\n- Vitamina B12: 450 pg/mL\n- Hemoglobina: 14,2 g/dL",
            {"vitamin_d": "borderline", "vitamin_b12": "normal", "anemia": "normal"}
        ),
        # 3. MCV elevated with macrocytic anemia
        (
            "Paciente: Carlos Mendes | Sexo: Masculino\nHemoglobina: 10,2 g/dL\nVCM: 115 fL\nVitamina B12: 120 pg/mL",
            {"anemia": "deficient", "vitamin_b12": "deficient"}
        ),
        # 4. Total calcium and ionized calcium
        (
            "Paciente: Ana Costa\nCalcio total: 7,5 mg/dL\nCalcio ionico: 0,95 mmol/L\nFerritina: 45 ng/mL",
            {"calcium": "deficient", "iron": "normal"}
        ),
        # 5. Folate deficiency with dotted leader
        (
            "DOSAGENS BIOQUÍMICAS\nFolato sérico ............... 2,1 ng/mL\nVitamina B12 ................ 380 pg/mL",
            {"folate": "deficient", "vitamin_b12": "normal"}
        ),
        # 6. Narrative paragraph report
        (
            "Paciente feminina de 28 anos com queixas de fadiga extrema. No hemograma observou-se hemoglobina de 9,8 g/dL "
            "e ferritina sérica de 6,5 ng/mL, compatível com anemia ferropriva grave.",
            {"iron": "deficient", "anemia": "deficient"}
        ),
        # 7. Thousands dot with decimal comma
        (
            "Vitamina B12: 1.250 pg/mL\nVitamina D: 35,0 ng/mL\nHemoglobina: 13,8 g/dL",
            {"vitamin_b12": "normal", "vitamin_d": "normal", "anemia": "normal"}
        ),
        # 8. TSAT and Ferritin panel
        (
            "Paciente: Lucas Silva\nFerritina: 12 ng/mL\nSaturacao de transferrina: 11 %\nHemoglobina: 11,5 g/dL",
            {"iron": "deficient", "anemia": "deficient"}
        ),
        # 9. Reference range column format
        (
            "EXAME | RESULTADO | VR\nVitamina D | 18,2 ng/mL | VR: 30 a 100 ng/mL\n"
            "Vitamina B12 | 190 pg/mL | VR: 200 a 900 pg/mL\n"
            "Hemoglobina | 12,8 g/dL | VR: 12 a 16 g/dL",
            {"vitamin_d": "deficient", "vitamin_b12": "deficient", "anemia": "normal"}
        ),
        # 10. Normal panel with all targets
        (
            "Vitamina D: 42 ng/mL\nVitamina B12: 580 pg/mL\nFerritina: 110 ng/mL\n"
            "Calcio total: 9,6 mg/dL\nAcido folico: 8,5 ng/mL\nHemoglobina: 14,5 g/dL",
            {"vitamin_d": "normal", "vitamin_b12": "normal", "iron": "normal", "calcium": "normal", "folate": "normal", "anemia": "normal"}
        ),
        # 11. Borderline B12 and Borderline Vit D
        (
            "Paciente: Beatriz Rocha\nVitamina B12: 240 pg/mL\nVitamina D: 26,0 ng/mL",
            {"vitamin_b12": "borderline", "vitamin_d": "borderline"}
        ),
        # 12. Microcytic anemia with low MCV
        (
            "Hemoglobina: 9,0 g/dL\nVCM: 68 fL\nFerritina: 5 ng/mL",
            {"anemia": "deficient", "iron": "deficient"}
        ),
        # 13. Male sex detection with 12.5 g/dL Hb (deficient for male, normal for female)
        (
            "Paciente: Roberto Souza | Sexo: M\nHemoglobina: 12,5 g/dL",
            {"anemia": "deficient"}
        ),
        # 14. Female sex detection with 12.5 g/dL Hb (normal for female)
        (
            "Paciente: Juliana Prado | Sexo: Feminino\nHemoglobina: 12,5 g/dL",
            {"anemia": "normal"}
        ),
        # 15. Hypocalcemia report
        (
            "Paciente: Thiago Faria\nCalcio serico: 6,8 mg/dL\nVitamina D: 11 ng/mL",
            {"calcium": "deficient", "vitamin_d": "deficient"}
        ),
        # 16. Multiple CBC indices
        (
            "HEMOGRAMA COMPLETO\nHemoglobina: 10,5 g/dL\nHematocrito: 32 %\nVCM: 108 fL\nHCM: 34 pg",
            {"anemia": "deficient"}
        ),
        # 17. Serum iron alone
        (
            "Ferro serico: 35 ug/dL\nFerritina: 14 ng/mL",
            {"iron": "deficient"}
        ),
        # 18. Negation test: sem anemia e nega deficiencia
        (
            "Paciente sem anemia. Nega deficiencia de vitamina B12.",
            {"anemia": "normal", "vitamin_b12": "normal"}
        ),
        # 19. Corrected calcium with albumin
        (
            "Calcio corrigido: 7,8 mg/dL\nAlbumina: 3,2 g/dL",
            {"calcium": "deficient"}
        ),
        # 20. Cobalamin synonym
        (
            "Cobalamina serica: 145 pg/mL\nCalcidiol: 18 ng/mL",
            {"vitamin_b12": "deficient", "vitamin_d": "deficient"}
        ),
        # 21. User's exact earlier Portuguese case (32 ng/mL, 420 pg/mL, 13,5 g/dL)
        (
            "Vitamina D: 32 ng/mL\nVitamina B12: 420 pg/mL\nHemoglobina: 13,5 g/dL",
            {"vitamin_d": "normal", "vitamin_b12": "normal", "anemia": "normal"}
        ),
    ]

    for idx, (text, expected) in enumerate(test_reports, 1):
        res = engine.analyze_text(text)
        by_key = {n["key"]: n["status"] for n in res["nutrients"] + res["findings"]}
        for target, exp_status in expected.items():
            got = by_key.get(target)
            norm_got = "normal" if got in ("normal", "not_deficient") else got
            norm_exp = "normal" if exp_status in ("normal", "not_deficient") else exp_status
            assert norm_got == norm_exp, f"[Report {idx}] Failed for {target}: expected {norm_exp}, got {norm_got} (raw got: {got})\nText: {text}"

    print(f"✓ Test 20+ Diverse Portuguese Reports: ALL {len(test_reports)} PASSED!")


def test_english_reports_not_broken():
    """Verify that existing English extraction remains 100% functional without regressions."""
    engine = NutriCheckNLP()
    en_report = """
    LABORATORY REPORT
    Patient: John Doe | Sex: Male
    Vitamin B12: 150 pg/mL
    Vitamin D: 16 ng/mL
    Ferritin: 11 ng/mL
    Hemoglobin: 10.5 g/dL
    Calcium: 7.9 mg/dL
    MCV: 112 fL
    """
    res = engine.analyze_text(en_report)
    by_key = {n["key"]: n["status"] for n in res["nutrients"] + res["findings"]}
    labs = {l["analyte"]: l for l in res["labs"]}

    assert by_key["vitamin_b12"]["status"] == "deficient"
    assert by_key["vitamin_d"]["status"] == "deficient"
    assert by_key["iron"]["status"] == "deficient"
    assert by_key["anemia"]["status"] == "deficient"
    assert by_key["calcium"]["status"] == "deficient"
    assert "mcv" in labs
    assert labs["mcv"]["value"] == 112.0
    assert labs["mcv"]["status"] == "high"

    print("✓ Test English Reports Regression: PASSED!")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING PORTUGUESE EXTRACTION PIPELINE VALIDATION SUITE")
    print("=" * 70)
    test_pedro_santos_report()
    test_20_diverse_portuguese_reports()
    test_english_reports_not_broken()
    print("=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY WITH 100% ACCURACY!")

