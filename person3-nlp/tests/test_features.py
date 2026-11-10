"""The eight added features: detective, timeline, explanations, contradictions, abstention, documents, visit prep, vault."""
import json

import pytest
from fastapi.testclient import TestClient

from nutricheck_nlp.api.app import create_app
from nutricheck_nlp.crosscheck import cross_check
from nutricheck_nlp.explain import explain, verify_explanation
from nutricheck_nlp.privacy import mask_pii
from nutricheck_nlp.timeline import build_timeline
from nutricheck_nlp.vault import HealthVault, VaultError
from nutricheck_nlp.visit_prep import build_visit_prep
from doc_factory import make_docx, make_pdf


def nut(res, key):
    return next(n for n in res["nutrients"] + res["findings"] if n["key"] == key)


def reports(engine, items):
    out = []
    for label, text in items:
        r = engine.analyze_text(text)
        r["report_id"] = label
        out.append(r)
    return out


JAN_APR_JUL = [
    ("Jan", "Collected: 2025-01-10\nFerritin: 35 ng/mL\nPatient reports fatigue."),
    ("Apr", "Collected: 2025-04-12\nFerritin: 21 ng/mL\nPatient denies fatigue."),
    ("Jul", "Collected: 2025-07-09\nFerritin: 12 ng/mL"),
]


# ============================================================ 1. Medical Report Detective
def test_detective_exact_evidence_location_context_and_missing_info(engine):
    text = "LABORATORY RESULTS\nCollected: 2025-03-12\nFerritin: 9 ng/mL\nPatient denies taking iron supplements."
    n = nut(engine.analyze_text(text), "iron")
    ev = next(e for e in n["detective"]["evidence"] if e["kind"] == "lab")
    assert "Ferritin value of 9 ng/mL" in ev["found"] and ev["quote"] == "Ferritin: 9 ng/mL"
    assert ev["location"]["line"] == 3 and ev["location"]["section"] == "laboratory results"
    assert "section “laboratory results”" in ev["location_text"] and ev["date"] == "2025-03-12"
    assert text[ev["span"]["start"]:ev["span"]["end"]].startswith("Ferritin")          # the span can be highlighted in the original
    ctx = " ".join(n["detective"]["context"]).lower()
    assert "denied" in ctx and "iron" in ctx
    miss = " ".join(n["detective"]["missing_information"]).lower()
    assert "reference range" in miss


def test_detective_reports_page_for_pdf_values(engine):
    pdf = make_pdf(["Patient notes", "LABORATORY RESULTS\nFerritin: 9 ng/mL"])
    (res,) = engine.analyze_file("r.pdf", pdf, "application/pdf")
    ev = next(e for e in nut(res, "iron")["detective"]["evidence"] if e["kind"] == "lab")
    assert ev["location"]["page"] == 2 and "page 2" in ev["location_text"]


def test_detective_says_when_no_lab_supports_a_statement(engine):
    n = nut(engine.analyze_text("Patient has iron deficiency."), "iron")
    assert any("No laboratory value" in m for m in n["detective"]["missing_information"])


# ============================================================ 2. Health timeline
def test_timeline_flags_downward_trend_with_evidence(engine):
    tl = build_timeline(reports(engine, JAN_APR_JUL))
    s = tl["series"][0]
    assert s["analyte"] == "ferritin" and s["trend"] == "decreasing" and s["flag"]
    assert [p["canonical_value"] for p in s["points"]] == [35, 21, 12] and [p["date"] for p in s["points"]] == ["2025-01-10", "2025-04-12", "2025-07-09"]
    assert s["change_pct"] == pytest.approx(-65.7, abs=0.1) and "not a diagnosis" in s["note"]
    assert all(p["quote"] and p["location"]["line"] for p in s["points"])


def test_timeline_converts_comparable_units_and_says_so(engine):
    reps = reports(engine, [("a", "Collected: 2024-01-01\nVitamin D: 100 nmol/L"), ("b", "Collected: 2024-06-01\nVitamin D: 20 ng/mL")])
    s = build_timeline(reps)["series"][0]
    assert s["trend"] == "decreasing" and any("converted" in c for c in s["caveats"])


def test_timeline_never_uses_undated_ambiguous_or_unitless_values(engine):
    reps = reports(engine, [("a", "Collected: 2024-01-01\nFerritin: 30 ng/mL"), ("b", "Ferritin: 8 ng/mL"),
                            ("c", "Collected: 03/04/2024\nFerritin: 20 ng/mL"), ("d", "Collected: 2024-09-01\nFerritin: 8")])
    tl = build_timeline(reps)
    assert tl["series"][0]["trend"] == "insufficient_data"
    reasons = " ".join(e["reason"] for e in tl["excluded"])
    assert "no usable date" in reasons and "ambiguous" in reasons and "unit is missing" in reasons


def test_timeline_ambiguous_date_resolved_only_when_told(engine):
    r = engine.analyze_text("Collected: 03/04/2024\nFerritin: 20 ng/mL", date_order="DMY")
    assert r["labs"][0]["date"]["iso"] == "2024-04-03"
    r = engine.analyze_text("Collected: 03/04/2024\nFerritin: 20 ng/mL", date_order="MDY")
    assert r["labs"][0]["date"]["iso"] == "2024-03-04"
    r = engine.analyze_text("Collected: 25/04/2024\nFerritin: 20 ng/mL")           # 25 cannot be a month
    assert r["labs"][0]["date"]["iso"] == "2024-04-25"


def test_timeline_stable_and_increasing(engine):
    up = build_timeline(reports(engine, [("a", "Collected: 2024-01-01\nFerritin: 20 ng/mL"), ("b", "Collected: 2024-03-01\nFerritin: 40 ng/mL")]))
    assert up["series"][0]["trend"] == "increasing" and not up["series"][0]["flag"]
    flat = build_timeline(reports(engine, [("a", "Collected: 2024-01-01\nFerritin: 50 ng/mL"), ("b", "Collected: 2024-03-01\nFerritin: 51 ng/mL")]))
    assert flat["series"][0]["trend"] == "stable"


# ============================================================ 3. Explain at my level
LEVELS = ["simple_en", "detailed_en", "medical_en", "te", "hi"]


@pytest.mark.parametrize("level", LEVELS)
def test_explanations_are_traceable_at_every_level(engine, level):
    r = engine.analyze_text("Collected: 2025-03-12\nFerritin: 9 ng/mL\nVitamin D: 50 nmol/L\nHemoglobin: 11 g/dL\nNo vitamin B12 deficiency.")
    ex = explain(r, level)
    assert ex["traceability"]["ok"], ex["traceability"]["problems"]
    text = json.dumps(ex, ensure_ascii=False)
    assert "9 ng/mL" in text and "“Ferritin: 9 ng/mL”" in text                      # value + quote, never translated
    assert all(p["evidence_refs"] or p["kind"] == "background" for s in ex["sections"] for p in s["paragraphs"])
    assert len(ex["questions"]) == 5 and ex["disclaimer"]


def test_telugu_and_hindi_are_scripts_with_notice(engine):
    r = engine.analyze_text("Ferritin: 9 ng/mL")
    te, hi = explain(r, "te"), explain(r, "hi")
    assert any("\u0c00" <= c <= "\u0c7f" for c in json.dumps(te, ensure_ascii=False)) and "not clinically reviewed" in te["notice"]
    assert any("\u0900" <= c <= "\u097f" for c in json.dumps(hi, ensure_ascii=False)) and "not clinically reviewed" in hi["notice"]
    assert "template" in te["notice"]


def test_tampering_with_a_translation_is_detected(engine):
    r = engine.analyze_text("Ferritin: 9 ng/mL")
    for level in ("simple_en", "hi"):
        ex = explain(r, level)
        lab_par = next(p for s in ex["sections"] for p in s["paragraphs"] if p["kind"] == "lab")
        lab_par["text"] = lab_par["text"].replace("9 ng/mL", "19 ng/mL")                  # a "translation" that changes the value
        assert not verify_explanation(ex, r)["ok"]
    ex = explain(r, "simple_en")
    ex["sections"][0]["status"] = "not_deficient"                                       # a changed conclusion
    assert not verify_explanation(ex, r)["ok"]
    ex = explain(r, "simple_en")
    q = next(p for s in ex["sections"] for p in s["paragraphs"] if p["kind"] == "quote")
    q["text"] = "The report says: “Ferritin: 90 ng/mL”"
    assert not verify_explanation(ex, r)["ok"]


def test_medical_level_has_glossary(engine):
    ex = explain(engine.analyze_text("Ferritin: 9 ng/mL"), "medical_en")
    assert {g["term"] for g in ex["glossary"]} >= {"ferritin", "deficiency"}


def test_explain_when_nothing_to_explain(engine):
    ex = explain(engine.analyze_text("Patient in good condition."), "simple_en")
    assert "not contain enough information" in ex["sections"][0]["paragraphs"][0]["text"] and ex["traceability"]["ok"]


def test_explain_unreliable_value_asks_for_verification(engine):
    ex = explain(engine.analyze_text("Vitamin D: l8 ng/mL"), "simple_en")
    assert ex["traceability"]["ok"]
    assert "not give enough reliable information" in json.dumps(ex)


# ============================================================ 4. Contradictions and missing info
def test_cross_report_symptom_contradiction_and_missing_unit(engine):
    reps = reports(engine, [("A", "Patient reports fatigue."), ("B", "Patient denies fatigue.\nFerritin: 18")])
    cc = cross_check(reps)
    types = {i["type"] for i in cc["issues"]}
    assert {"symptom_conflict", "missing_or_unclear"} <= types
    sym = next(i for i in cc["issues"] if i["type"] == "symptom_conflict")
    assert {e["report_id"] for e in sym["evidence"]} == {"A", "B"}
    assert "does not decide which record is correct" in cc["principle"]
    assert any("no unit" in i["title"] for i in cc["issues"])


def test_cross_report_statement_and_lab_conflicts(engine):
    reps = reports(engine, [("A", "Patient has iron deficiency."), ("B", "No iron deficiency."), ("C", "Collected: 2025-01-01\nFerritin: 8 ng/mL"),
                            ("D", "Collected: 2025-01-01\nFerritin: 80 ng/mL")])
    types = [i["type"] for i in cross_check(reps)["issues"]]
    assert "statement_conflict" in types and "value_conflict" in types and "lab_vs_statement" in types


def test_cross_report_different_dates_downgrades_severity(engine):
    reps = reports(engine, [("A", "Collected: 2024-01-01\nPatient has iron deficiency."), ("B", "Collected: 2025-01-01\nNo iron deficiency.")])
    iss = next(i for i in cross_check(reps)["issues"] if i["type"] == "statement_conflict")
    assert iss["severity"] == "medium" and "change over time" in iss["detail"]


def test_clean_reports_have_no_issues(engine):
    assert cross_check(reports(engine, JAN_APR_JUL[:1]))["count"] <= 1       # only the "no reference range" style info is allowed


def test_unit_differences_noted(engine):
    cc = cross_check(reports(engine, [("A", "Vitamin D: 50 nmol/L"), ("B", "Vitamin D: 20 ng/mL")]))
    assert any("different units" in i["title"] for i in cc["issues"])


# ============================================================ 5. AI that knows when not to predict
def test_abstains_when_unit_missing_ocr_ambiguous_or_implausible(engine):
    for text, reason in [("Vitamin D: 18", "unit_missing"), ("Vitamin D: l8 ng/mL", "ambiguous_ocr_characters"), ("Calcium: 85 mg/dL", "implausible_value"),
                         ("Ferritin: 1,5 ng/mL", "ambiguous_decimal_format")]:
        r = engine.analyze_text(text)
        n = next(x for x in r["nutrients"] if reason in x["review_reasons"])
        assert n["status"] in ("not_assessed", "suspected") and n["reliability"] in ("abstain", "verify") and r["needs_human_review"]
        assert r["quality"]["verdict"] == "verify" and r["quality"]["actions"], text


def test_low_ocr_line_confidence_blocks_a_confident_result(engine):
    lines = [{"text": "Ferritin: 18 ng/mL", "confidence": 0.62, "box": [0, 0, 1, 1]}, {"text": "Vitamin D: 40 ng/mL", "confidence": 0.99, "box": [0, 0, 1, 1]}]
    r = engine.analyze_ocr_result({"success": True, "text": "Ferritin: 18 ng/mL\nVitamin D: 40 ng/mL", "average_confidence": 0.95, "lines": lines, "line_count": 2, "image": "x.png"})
    iron, vd = nut(r, "iron"), nut(r, "vitamin_d")
    assert iron["status"] == "suspected" and "low_ocr_confidence" in iron["review_reasons"] and iron["confidence"] <= 0.7
    assert vd["status"] == "not_deficient" and vd["reliability"] == "ok"
    assert any("Ferritin" in a and "OCR confidence" in a for a in r["quality"]["actions"])


def test_whole_document_low_ocr_confidence_cannot_determine(engine):
    r = engine.analyze_ocr_result({"success": True, "text": "Ferritin: 8 ng/mL", "average_confidence": 0.5, "line_count": 1, "image": "x.png"})
    assert r["quality"]["verdict"] == "cannot_determine" and any("clearer image" in a for a in r["quality"]["actions"])


def test_garbage_text_is_not_readable(engine):
    r = engine.analyze_text("§§##@@ ¤¤ ¦¦ ~~ ^^ `` §§ ¬¬ @@@ ###" * 5)
    assert not next(c for c in r["quality"]["checks"] if c["id"] == "text_readable")["ok"]


def test_insufficient_information_is_a_safe_outcome(engine):
    r = engine.analyze_text("Patient in good general condition.")
    assert r["quality"]["verdict"] == "insufficient_information" and "safe outcome" in r["quality"]["message"]


def test_quality_checks_cover_the_five_questions(engine):
    ids = [c["id"] for c in engine.analyze_text("Ferritin: 8 ng/mL")["quality"]["checks"]]
    assert ids == ["text_readable", "values_extracted", "units_clear", "language_supported", "enough_facts"]


# ============================================================ 6. One-click document understanding
def test_processing_report_for_multi_page_pdf(engine):
    pdf = make_pdf(["LABORATORY RESULTS\nFerritin: 8 ng/mL", "Notes\nPatient in good condition.", "Vitamin D: l8 ng/mL"])
    (res,) = engine.analyze_file("r.pdf", pdf, "application/pdf")
    pr = res["processing_report"]
    assert pr["file_format"] == "pdf" and pr["document_type"] and pr["language"] == "English" and pr["page_count"] == 3
    by = {p["page"]: p for p in pr["pages"]}
    assert by[1]["status"] == "ok" and by[1]["values_found"] == 1
    assert by[3]["status"] == "needs_verification" and by[3]["values_needing_verification"] == 1
    assert pr["pages_processed_ok"] == 2
    v = pr["values_to_verify"][0]
    assert v["page"] == 3 and "ambiguous_ocr_characters" in v["reasons"] and v["quote"] == "Vitamin D: l8 ng/mL"
    assert res["quality"]["pages_needing_attention"] == [3]


def test_blank_scanned_page_is_reported_not_hidden(engine):
    pdf = make_pdf(["Ferritin: 8 ng/mL", ""])
    (res,) = engine.analyze_file("r.pdf", pdf, "application/pdf")
    assert [p["status"] for p in res["processing_report"]["pages"]] == ["ok", "failed"] and res["needs_human_review"]


def test_docx_txt_and_csv_report_format_and_type(engine):
    docx = make_docx(["Ferritin: 8 ng/mL"])
    assert engine.analyze_file("a.docx", docx)[0]["processing_report"]["file_format"] == "docx"
    (res,) = engine.analyze_file("a.txt", b"Patient has iron deficiency and fatigue.")
    assert res["processing_report"]["file_format"] == "txt" and res["document_profile"]["type"] == "clinical note"
    (res,) = engine.analyze_file("a.csv", b"Test,Result,Unit\nFerritin,8,ng/mL\nVitamin D,50,ng/mL\n")
    assert res["document_profile"]["type"] == "laboratory report"


def test_language_is_identified_for_uploads(engine):
    (res,) = engine.analyze_file("a.txt", "Paciente com deficiência de ferro. Ferritina: 8 ng/mL.".encode())
    assert res["language_supported"]
    assert res["translation"]["status"] == "translated"
    assert res["translation"]["source_language"] == "pt"
    assert res["language_info"]["code"] == "en"

# ============================================================ 7. Doctor-visit preparation
def test_visit_prep_contains_results_changes_and_questions(engine):
    reps = reports(engine, JAN_APR_JUL)
    tl, cc = build_timeline(reps), cross_check(reps)
    vp = build_visit_prep(reps, tl, cc, generated_on="2025-08-01")
    md = vp["markdown"]
    assert "| Ferritin | 35 ng/mL | 2025-01-10 |" in md and "Downward trend for review" in md and "2025-08-01" in md
    assert "Is this result outside the reference range for me?" in md and "Should this measurement be compared with my previous report?" in md
    assert any("ferritin" in q.lower() for q in vp["questions"]) and "not a diagnosis" in md
    low = md.lower()
    assert "you should take" not in low and "recommend" not in low.replace("does not recommend", "")


def test_visit_prep_questions_follow_the_findings(engine):
    r = engine.analyze_text("Calcium: 7.5 mg/dL\nHemoglobin: 11 g/dL")
    r["report_id"] = "x"
    qs = " ".join(build_visit_prep([r])["questions"])
    assert "albumin" in qs and "hemoglobin reference range" in qs


@pytest.mark.parametrize("lang", ["hi", "te"])
def test_visit_prep_other_languages_for_standard_questions(engine, lang):
    r = engine.analyze_text("Ferritin: 8 ng/mL")
    r["report_id"] = "x"
    vp = build_visit_prep([r], language=lang)
    assert vp["language"] == lang and "Ferritin" in vp["markdown"]
    with pytest.raises(ValueError):
        build_visit_prep([r], language="fr")


# ============================================================ 8. Privacy vault
def test_mask_pii_masks_identifiers_not_lab_data():
    text = "Patient: John Smith\nPhone: +91 98765 43210\nMRN: AB123456\nDOB: 03/04/1980\nEmail: j.smith@example.com\nCollected: 2025-03-12\nFerritin: 8 ng/mL\nAadhaar 1234 5678 9012"
    m = mask_pii(text)["masked"]
    for secret in ("John Smith", "98765", "AB123456", "03/04/1980", "j.smith@example.com", "1234 5678 9012"):
        assert secret not in m, secret
    assert "Ferritin: 8 ng/mL" in m and "2025-03-12" in m and "[NAME]" in m and "[PHONE]" in m and "[ID]" in m and "[DOB]" in m and "[EMAIL]" in m


def test_mask_pii_does_not_eat_lab_numbers():
    t = "Ferritin: 8 ng/mL\nVitamin D: 12.5 ng/mL\nPlatelets: 250,000 /mm3\nCollected: 12/03/2025\nHemoglobin: 10.2 g/dL"
    assert mask_pii(t)["masked"] == t


KEY = "a-long-secret-key-123"


def test_vault_requires_consent_and_stores_only_masked_summaries(engine, tmp_path):
    v = HealthVault(tmp_path)
    res = engine.analyze_text("Patient: John Smith denies iron deficiency.\nFerritin: 8 ng/mL")
    with pytest.raises(VaultError) as e:
        v.store(KEY, res, consent=False)
    assert e.value.status == 403 and not list(tmp_path.rglob("*.json"))             # nothing is written without permission
    item = v.store(KEY, res, consent=True, label="Jan")
    raw = next(tmp_path.rglob("*.json")).read_text(encoding="utf-8")
    assert "John Smith" not in raw and "[NAME]" in raw and item["original_stored"] is False
    assert not (tmp_path / "x").exists() and not list(tmp_path.rglob("*.bin"))


def test_vault_original_is_separate_and_optional(engine, tmp_path):
    v = HealthVault(tmp_path)
    res = engine.analyze_text("Ferritin: 8 ng/mL")
    item = v.store(KEY, res, consent=True, original=b"ORIGINAL BYTES", store_original=True, filename="r.txt")
    bins, jsons = list(tmp_path.rglob("*.bin")), list(tmp_path.rglob("*.json"))
    assert len(bins) == 1 and len(jsons) == 1 and bins[0].parent != jsons[0].parent
    assert v.get_original(KEY, item["id"]) == b"ORIGINAL BYTES"
    assert b"ORIGINAL BYTES" not in jsons[0].read_bytes()
    with pytest.raises(VaultError):
        v.store(KEY, res, consent=True, store_original=True)                             # no bytes given


def test_vault_delete_export_timeline_and_isolation(engine, tmp_path):
    v = HealthVault(tmp_path)
    for label, t in JAN_APR_JUL:
        v.store(KEY, engine.analyze_text(t), consent=True, label=label)
    assert [r["label"] for r in v.list(KEY)] == ["Jan", "Apr", "Jul"]
    assert v.timeline(KEY)["series"][0]["trend"] == "decreasing"
    exp = v.export(KEY)
    assert len(exp["reports"]) == 3 and exp["timeline"]["series"] and exp["privacy"]["retention"]
    assert v.list("another-secret-key-456") == []                                        # another owner sees nothing
    rid = v.list(KEY)[0]["id"]
    v.delete(KEY, rid)
    with pytest.raises(VaultError) as e:
        v.get(KEY, rid)
    assert e.value.status == 404 and len(v.list(KEY)) == 2
    assert v.delete_all(KEY) == {"deleted_reports": 2} and v.list(KEY) == [] and not list(tmp_path.rglob("*.json"))


def test_vault_rejects_weak_keys_and_bad_ids(engine, tmp_path):
    v = HealthVault(tmp_path)
    for bad in ("", "short", None):
        with pytest.raises(VaultError):
            v.list(bad)  # type: ignore[arg-type]
    v.store(KEY, engine.analyze_text("Ferritin: 8 ng/mL"), consent=True)
    for bad_id in ("../../etc/passwd", "x", "A" * 32, ""):
        with pytest.raises(VaultError):
            v.get(KEY, bad_id)


def test_vault_retention_purge(engine, tmp_path):
    from datetime import datetime, timedelta, timezone
    v = HealthVault(tmp_path, retention_days=30)
    old = datetime.now(timezone.utc) - timedelta(days=45)
    v.store(KEY, engine.analyze_text("Ferritin: 8 ng/mL"), consent=True, original=b"x", store_original=True, now=old)
    v.store(KEY, engine.analyze_text("Ferritin: 9 ng/mL"), consent=True)
    assert v.purge_expired() == 1 and len(v.list(KEY)) == 1 and not list(tmp_path.rglob("*.bin"))


# ============================================================ API surface of the new features
@pytest.fixture()
def client(tmp_path):
    with TestClient(create_app(vault_dir=str(tmp_path / "vault"))) as c:
        yield c


def test_api_records_analyze(client):
    body = {"reports": [{"label": l, "text": t} for l, t in JAN_APR_JUL]}
    j = client.post("/v1/records/analyze", json=body).json()
    assert j["timeline"]["flagged"] == ["ferritin"] and j["crosscheck"]["count"] >= 1 and len(j["analyses"]) == 3
    assert client.post("/v1/records/analyze", json={"reports": []}).status_code == 422
    assert client.post("/v1/records/analyze", json={"reports": [{"text": "x", "sex": "Z"}]}).status_code == 422


def test_api_records_analyze_files(client):
    files = [("files", ("a.txt", b"Collected: 2025-01-01\nFerritin: 30 ng/mL", "text/plain")),
             ("files", ("b.txt", b"Collected: 2025-03-01\nFerritin: 12 ng/mL", "text/plain"))]
    j = client.post("/v1/records/analyze/files", files=files).json()
    assert j["timeline"]["series"][0]["trend"] == "decreasing"
    bad = client.post("/v1/records/analyze/files", files=[("files", ("x.pdf", b"nope", "application/pdf"))])
    assert bad.status_code == 415 or bad.status_code == 422


def test_api_explain_every_level_and_invalid_level(client):
    for level in LEVELS:
        j = client.post("/v1/explain", json={"text": "Ferritin: 9 ng/mL", "level": level}).json()
        assert j["explanation"]["traceability"]["ok"] and j["explanation"]["level"] == level
    assert client.post("/v1/explain", json={"text": "x", "level": "klingon"}).status_code == 422


def test_api_visit_prep_and_mask(client):
    j = client.post("/v1/visit-prep", json={"reports": [{"label": "A", "text": "Ferritin: 8 ng/mL"}], "language": "hi"}).json()
    assert j["language"] == "hi" and "<html" in j["html"]
    m = client.post("/v1/privacy/mask", json={"text": "Patient: Jane Roe, MRN: 12345678"}).json()
    assert "Jane Roe" not in m["masked"] and m["counts"]
    assert "retention" in client.get("/v1/privacy/notice").json()


def test_api_vault_flow(client):
    h = {"X-Vault-Key": KEY}
    no_consent = client.post("/v1/vault/reports", json={"text": "Ferritin: 8 ng/mL"}, headers=h)
    assert no_consent.status_code == 403 and client.get("/v1/vault/reports", headers=h).json()["reports"] == []
    ok = client.post("/v1/vault/reports", json={"text": "Collected: 2025-01-01\nFerritin: 8 ng/mL", "consent": True, "label": "Jan"}, headers=h)
    assert ok.status_code == 200
    rid = ok.json()["id"]
    assert client.get(f"/v1/vault/reports/{rid}", headers=h).json()["label"] == "Jan"
    assert client.get("/v1/vault/export", headers=h).json()["reports"]
    assert client.get("/v1/vault/reports", headers={"X-Vault-Key": "different-secret-key-9"}).json()["reports"] == []
    assert client.get("/v1/vault/reports").status_code == 400                           # missing key
    assert client.delete(f"/v1/vault/reports/{rid}", headers=h).json() == {"deleted": rid}
    assert client.get(f"/v1/vault/reports/{rid}", headers=h).status_code == 404
    assert client.delete("/v1/vault/reports", headers=h).json() == {"deleted_reports": 0}


def test_api_response_exposes_new_fields(client):
    j = client.post("/v1/analyze/text", json={"text": "Ferritin: 9 ng/mL"}).json()
    n = j["nutrients"][0]
    assert n["detective"]["evidence"] and n["reliability"] == "ok" and j["quality"]["verdict"] == "ok" and j["language_supported"]
