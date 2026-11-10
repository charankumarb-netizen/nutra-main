"""
Comprehensive Multilingual Validation Tests for NutriCheck NLP.

Validates:
1. The user's exact Portuguese test case (Vitamina D: 32 ng/mL, Vitamina B12: 420 pg/mL, Hemoglobina: 13,5 g/dL)
2. Portuguese deficiency and borderline cases with decimal commas and reference intervals
3. Spanish reports (laboratories and clinical statements)
4. French reports (laboratories and clinical statements)
5. German reports (laboratories and clinical statements)
6. Hindi reports (Devanagari script biomarkers, values, and units)
7. Table formats, dotted line formats, paragraph formats
8. Multilingual NegEx and Assertion understanding (negated, historical, affirmed)
9. Quality Assessment 4-tier category validation
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from nutricheck_nlp import NutriCheckNLP


def test_user_portuguese_report_exact():
    """Verify that the user's exact Portuguese report produces successful extraction and correct statuses."""
    engine = NutriCheckNLP()
    report = """
    Vitamina D: 32 ng/mL
    Vitamina B12: 420 pg/mL
    Hemoglobina: 13,5 g/dL
    """
    res = engine.analyze_text(report)
    by_key = {n["key"]: n for n in res["nutrients"] + res["findings"]}
    labs = res["labs"]

    # Must extract all 3 biomarkers without 'Not detected'
    assert len(labs) == 3, f"Expected 3 extracted labs, got {len(labs)}: {labs}"
    
    # Check values and units
    lab_map = {l["analyte"]: l for l in labs}
    assert "vitamin_d_total" in lab_map or "vitamin_d" in lab_map
    assert "vitamin_b12" in lab_map
    assert "hemoglobin" in lab_map

    hb_lab = lab_map["hemoglobin"]
    assert hb_lab["value"] == 13.5, f"Decimal comma '13,5' was not parsed as 13.5: {hb_lab['value']}"
    assert hb_lab["unit"] == "g/dL"

    # Statuses
    assert by_key["vitamin_d"]["status"] in ("not_deficient", "normal"), f"Vit D status was {by_key['vitamin_d']['status']}"
    assert by_key["vitamin_b12"]["status"] in ("not_deficient", "normal"), f"B12 status was {by_key['vitamin_b12']['status']}"
    assert by_key["anemia"]["status"] in ("not_deficient", "normal"), f"Anemia status was {by_key['anemia']['status']}"

    # Quality check
    assert res["quality"]["category"] == "successful_extraction"
    assert res["quality"]["sub_checks"]["values_extracted"] is True
    assert res["quality"]["sub_checks"]["units_identified"] is True


def test_portuguese_deficiency_and_comma_decimals():
    """Test Portuguese deficiency cases with comma decimals and reference ranges."""
    engine = NutriCheckNLP()
    report = """
    EXAMES LABORATORIAIS
    Vitamina D (25-OH) ........... 14,2 ng/mL  (VR: 30 a 100 ng/mL)
    Vitamina B12 ................. 165 pg/mL   (VR: 200 a 900 pg/mL)
    Ferritina .................... 9,8 ng/mL   (VR: 30 a 400 ng/mL)
    Hemoglobina .................. 10,4 g/dL   (VR: 12,0 a 16,0 g/dL)
    """
    res = engine.analyze_text(report)
    by_key = {n["key"]: n for n in res["nutrients"] + res["findings"]}

    assert by_key["vitamin_d"]["status"] == "deficient"
    assert by_key["vitamin_b12"]["status"] == "deficient"
    assert by_key["iron"]["status"] == "deficient"
    assert by_key["anemia"]["status"] == "deficient"
    assert res["quality"]["category"] == "successful_extraction"


def test_spanish_report_extraction():
    """Test Spanish lab report with comma and dot decimals."""
    engine = NutriCheckNLP()
    report = """
    INFORME DE LABORATORIO
    Vitamina D (25-hidroxivitamina D): 12,5 ng/mL
    Vitamina B12 (Cobalamina): 140 pg/mL
    Ferritina sérica: 11 ng/mL
    Calcio total: 7,5 mg/dL
    Ácido fólico: 2,3 ng/mL
    Hemoglobina: 9,8 g/dL
    """
    res = engine.analyze_text(report)
    by_key = {n["key"]: n for n in res["nutrients"] + res["findings"]}

    assert by_key["vitamin_d"]["status"] == "deficient"
    assert by_key["vitamin_b12"]["status"] == "deficient"
    assert by_key["iron"]["status"] == "deficient"
    assert by_key["calcium"]["status"] == "deficient"
    assert by_key["folate"]["status"] == "deficient"
    assert by_key["anemia"]["status"] == "deficient"


def test_french_report_extraction():
    """Test French medical report."""
    engine = NutriCheckNLP()
    report = """
    RÉSULTATS BIOLOGIQUES
    Vitamine D (25-OH): 11 ng/mL
    Vitamine B12: 155 pg/mL
    Ferritine: 8 ug/L
    Calcium sérique: 7,8 mg/dL
    Hémoglobine: 10,2 g/dL
    """
    res = engine.analyze_text(report)
    by_key = {n["key"]: n for n in res["nutrients"] + res["findings"]}

    assert by_key["vitamin_d"]["status"] == "deficient"
    assert by_key["vitamin_b12"]["status"] == "deficient"
    assert by_key["iron"]["status"] == "deficient"
    assert by_key["calcium"]["status"] == "deficient"
    assert by_key["anemia"]["status"] == "deficient"


def test_german_report_extraction():
    """Test German medical laboratory report."""
    engine = NutriCheckNLP()
    report = """
    LABORBERICHT
    Vitamin D (25-OH): 13 ng/mL
    Vitamin B12: 175 pg/mL
    Ferritin: 12 ng/mL
    Gesamtkalzium: 7,6 mg/dL
    Hämoglobin: 9,5 g/dL
    """
    res = engine.analyze_text(report)
    by_key = {n["key"]: n for n in res["nutrients"] + res["findings"]}

    assert by_key["vitamin_d"]["status"] == "deficient"
    assert by_key["vitamin_b12"]["status"] == "deficient"
    assert by_key["iron"]["status"] == "deficient"
    assert by_key["calcium"]["status"] == "deficient"
    assert by_key["anemia"]["status"] == "deficient"


def test_hindi_report_extraction():
    """Test Hindi medical laboratory report with Devanagari script."""
    engine = NutriCheckNLP()
    report = """
    रक्त परीक्षण रिपोर्ट
    विटामिन डी: 14 ng/mL
    विटामिन बी12: 160 pg/mL
    फेरिटिन: 11 ng/mL
    हीमोग्लोबिन: 9.8 g/dL
    """
    res = engine.analyze_text(report)
    by_key = {n["key"]: n for n in res["nutrients"] + res["findings"]}

    assert by_key["vitamin_d"]["status"] == "deficient"
    assert by_key["vitamin_b12"]["status"] == "deficient"
    assert by_key["iron"]["status"] == "deficient"
    assert by_key["anemia"]["status"] == "deficient"


def test_paragraph_and_unstructured_reports():
    """Test values embedded in natural language paragraphs rather than tables."""
    engine = NutriCheckNLP()
    report = (
        "Paciente do sexo feminino realizou exames de rotina. Notou-se nível sérico de "
        "Vitamina D em 16 ng/mL e Ferritina de 10 ng/mL, com hemoglobina de 11,2 g/dL. "
        "Não foram relatadas outras alterações."
    )
    res = engine.analyze_text(report)
    by_key = {n["key"]: n for n in res["nutrients"] + res["findings"]}

    assert by_key["vitamin_d"]["status"] == "deficient"
    assert by_key["iron"]["status"] == "deficient"
    assert by_key["anemia"]["status"] == "deficient"


def test_multilingual_negation():
    """Test negation in multiple languages."""
    engine = NutriCheckNLP()
    
    # Portuguese negation
    pt_res = engine.analyze_text("Paciente sem anemia e sem deficiência de ferro.")
    pt_key = {n["key"]: n for n in pt_res["nutrients"] + pt_res["findings"]}
    assert pt_key["anemia"]["status"] == "not_deficient"
    assert pt_key["iron"]["status"] == "not_deficient"

    # Spanish negation
    es_res = engine.analyze_text("Sin anemia ni deficiencia de vitamina B12.")
    es_key = {n["key"]: n for n in es_res["nutrients"] + es_res["findings"]}
    assert es_key["anemia"]["status"] == "not_deficient"
    assert es_key["vitamin_b12"]["status"] == "not_deficient"

    # French negation
    fr_res = engine.analyze_text("Absence d'anémie et pas de carence en fer.")
    fr_key = {n["key"]: n for n in fr_res["nutrients"] + fr_res["findings"]}
    assert fr_key["anemia"]["status"] == "not_deficient"
    assert fr_key["iron"]["status"] == "not_deficient"

    # German negation
    de_res = engine.analyze_text("Keine Anämie und kein Eisenmangel festgestellt.")
    de_key = {n["key"]: n for n in de_res["nutrients"] + de_res["findings"]}
    assert de_key["anemia"]["status"] == "not_deficient"
    assert de_key["iron"]["status"] == "not_deficient"


def test_four_tier_quality_categories():
    """Verify all 4 tiers of quality classification."""
    engine = NutriCheckNLP()

    # 1. successful_extraction
    res1 = engine.analyze_text("Vitamina D: 32 ng/mL\nVitamina B12: 420 pg/mL")
    assert res1["quality"]["category"] == "successful_extraction"

    # 2. partial_extraction (missing units)
    res2 = engine.analyze_text("Vitamina D: 15\nFerritina: 8")
    assert res2["quality"]["category"] == "partial_extraction"

    # 3. insufficient_data (valid document, but no target biomarkers)
    res3 = engine.analyze_text("Glucose: 95 mg/dL\nCholesterol: 180 mg/dL\nCreatinine: 0.9 mg/dL")
    assert res3["quality"]["category"] == "insufficient_data"

    # 4. document_processing_failure (empty or corrupted/unsupported script)
    res4 = engine.analyze_text("    ")
    assert res4["quality"]["category"] == "document_processing_failure"


if __name__ == "__main__":
    print("Running Multilingual Comprehensive Tests...")
    test_user_portuguese_report_exact()
    print("✓ User Portuguese Report: PASSED")
    test_portuguese_deficiency_and_comma_decimals()
    print("✓ Portuguese Deficiency + Comma Decimals: PASSED")
    test_spanish_report_extraction()
    print("✓ Spanish Report Extraction: PASSED")
    test_french_report_extraction()
    print("✓ French Report Extraction: PASSED")
    test_german_report_extraction()
    print("✓ German Report Extraction: PASSED")
    test_hindi_report_extraction()
    print("✓ Hindi Report Extraction: PASSED")
    test_paragraph_and_unstructured_reports()
    print("✓ Paragraph & Unstructured Reports: PASSED")
    test_multilingual_negation()
    print("✓ Multilingual Negation: PASSED")
    test_four_tier_quality_categories()
    print("✓ 4-Tier Quality Assessment Categories: PASSED")
    print("\nALL MULTILINGUAL COMPREHENSIVE TESTS PASSED!")

