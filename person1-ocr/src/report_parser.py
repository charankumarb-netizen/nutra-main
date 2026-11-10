
import re


def parse_medical_report(text):
    """Extract basic fields from OCR text."""

    report = {
        "patient_name": None,
        "age": None,
        "collection_date": None,
        "result_date": None,
        "tests": []
    }

    # Patient name
    match = re.search(r"Paciente:\s*(.+)", text, re.IGNORECASE)
    if match:
        report["patient_name"] = match.group(1).strip()

    # Age
    match = re.search(r"Idade:\s*(\d+)\s*anos", text, re.IGNORECASE)
    if match:
        report["age"] = int(match.group(1))

    # Dates
    match = re.search(
        r"Data da Coleta:\s*([\d/]+)", text, re.IGNORECASE
    )
    if match:
        report["collection_date"] = match.group(1)

    match = re.search(
        r"Data do Resultado:\s*([\d/]+)", text, re.IGNORECASE
    )
    if match:
        report["result_date"] = match.group(1)

    # Laboratory test rows
    test_pattern = re.compile(
        r"^(Hemoglobina|Vitamina D \(25-OH\)|Vitamina B12|"
        r"Ferro Sérico|Ferritina)\s*$",
        re.IGNORECASE | re.MULTILINE
    )

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for i, line in enumerate(lines):
        if test_pattern.match(line) and i + 2 < len(lines):
            report["tests"].append({
                "test_name": line,
                "result": lines[i + 1],
                "reference_range": lines[i + 2]
            })

    return report
