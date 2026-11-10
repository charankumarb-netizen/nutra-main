"""
NutriCheck AI Chatbot - Medical Report Aware Clinical Assistant.

This chatbot is grounded strictly in the active patient's report, parsed lab values,
reference ranges, and evidence detective findings. It provides plain-English,
empathetic, clinically sound explanations without hallucinating or prescribing medications.
"""

from __future__ import annotations

import re
from typing import Any

from .explain import explain as explain_report
from .i18n import FUNCTION_DETAILED, FUNCTION_SIMPLE
from .lexicon import ANALYTES, TARGETS


NUTRIENT_ALIASES = {
    "vitamin_b12": ["b12", "vitamin b12", "vitamin b-12", "vit b12", "cobalamin", "cyanocobalamin"],
    "vitamin_d": ["vitamin d", "vit d", "vitamin d3", "25-hydroxy", "25-oh", "calcidiol", "d3"],
    "iron": ["iron", "ferritin", "serum iron", "transferrin", "tibc", "iron deficiency", "iron saturation"],
    "calcium": ["calcium", "serum calcium", "total calcium", "ca"],
    "folate": ["folate", "folic acid", "vitamin b9", "b9"],
    "anemia": ["anemia", "haemoglobin", "hemoglobin", "hb", "hgb", "red blood cell", "rbc"],
}

NUTRIENT_DISPLAY_NAMES = {
    "vitamin_b12": "Vitamin B12",
    "vitamin_d": "Vitamin D",
    "iron": "Iron (Ferritin)",
    "calcium": "Calcium",
    "folate": "Folate (Folic Acid)",
    "anemia": "Hemoglobin / Anemia",
}

STATUS_FRIENDLY = {
    "deficient": "Low (Deficiency Detected)",
    "borderline": "Borderline / Insufficient",
    "normal": "Normal",
    "not_deficient": "Normal (Not Deficient)",
    "high": "High (Above Range)",
    "not_assessed": "Not Detected / Not Assessed",
    "suspected": "Suspected (Needs Confirmation)",
    "conflicting": "Conflicting Information in Report",
}

FOOD_SOURCES = {
    "vitamin_b12": (
        "• **Animal-based sources**: Eggs, dairy (milk, cheese, yogurt), fish (salmon, tuna), poultry, and lean meats.\n"
        "• **Plant-based & fortified sources**: Fortified breakfast cereals, fortified plant milks (soy, almond, oat), nutritional yeast.\n"
        "• *Absorption note*: Vitamin B12 requires gastric acid and intrinsic factor. If absorption is impaired (e.g. from metformin, acid blockers, or gut conditions), oral supplements or injections may be recommended by your doctor."
    ),
    "vitamin_d": (
        "• **Sunlight**: 15–20 minutes of mid-day sun exposure on arms and legs (when appropriate).\n"
        "• **Dietary sources**: Fatty fish (salmon, mackerel, sardines), egg yolks, fortified milk, fortified orange juice, and UV-exposed mushrooms.\n"
        "• *Absorption note*: Vitamin D is fat-soluble; taking supplements with a meal containing healthy fats improves absorption."
    ),
    "iron": (
        "• **Heme iron (highly absorbed)**: Lean red meats, poultry, and fish.\n"
        "• **Non-heme iron (plant-based)**: Lentils, chickpeas, black beans, spinach, fortified cereals, tofu, and pumpkin seeds.\n"
        "• *Absorption tip*: Pair plant iron with **Vitamin C** (lemon juice, oranges, bell peppers) to dramatically increase absorption. Avoid drinking coffee or tea within 1 hour of iron-rich meals, as tannins block iron uptake."
    ),
    "calcium": (
        "• **Dairy sources**: Milk, yogurt, kefir, and cheeses.\n"
        "• **Non-dairy sources**: Fortified plant milks, calcium-set tofu, dark leafy greens (kale, bok choy, collard greens), canned sardines with soft bones, almonds, and chia seeds.\n"
        "• *Synergy*: Adequate Vitamin D is essential for your intestines to absorb calcium efficiently."
    ),
    "folate": (
        "• **Rich sources**: Dark leafy greens (spinach, romaine, kale), asparagus, broccoli, Brussels sprouts, beans, lentils, peas, citrus fruits, avocados, and fortified grain products.\n"
        "• *Cooking note*: Folate is heat-sensitive; lightly steaming vegetables preserves significantly more folate than boiling."
    ),
    "anemia": (
        "• Support red blood cell production with balanced intake of **Iron, Vitamin B12, and Folate**.\n"
        "• If blood loss (e.g. heavy menstrual cycles or digestive loss) is present, discuss treatment with your physician."
    ),
}

DISCLAIMER_TEXT = (
    "\n\n---\n*Note: NutriCheck provides AI-powered screening support for educational purposes only. "
    "This is not a medical diagnosis or treatment prescription. Please discuss these findings with your qualified healthcare provider.*"
)


class NutriCheckChatbot:
    """
    Intelligent healthcare chatbot grounded in the specific report analysis.
    """

    def __init__(self, result: dict[str, Any] | None = None, report_text: str = "") -> None:
        self.result = result or {}
        self.report_text = report_text
        self._index_nutrients()

    def set_context(self, result: dict[str, Any], report_text: str = "") -> None:
        self.result = result
        self.report_text = report_text
        self._index_nutrients()

    def _index_nutrients(self) -> None:
        self.nutrient_map: dict[str, dict[str, Any]] = {}
        self.labs_map: dict[str, list[dict[str, Any]]] = {}

        if not self.result:
            return

        all_items = self.result.get("nutrients", []) + self.result.get("findings", [])
        for item in all_items:
            self.nutrient_map[item["key"]] = item

        for lab in self.result.get("labs", []):
            target = lab.get("target")
            if target:
                self.labs_map.setdefault(target, []).append(lab)

    def _detect_nutrient_in_query(self, query: str) -> str | None:
        q = query.lower()
        for key, aliases in NUTRIENT_ALIASES.items():
            for alias in aliases:
                # word boundary match
                pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
                if re.search(pattern, q):
                    return key
        return None

    def answer(self, query: str) -> str:
        """
        Generate an accurate, context-aware answer for the user's question.
        """
        q = (query or "").strip().lower()
        if not q:
            return "Please ask a question about your medical report or nutrient screening results."

        if not self.result:
            return (
                "Please upload or paste a medical report first and click **Analyse Report**. "
                "Once your report is processed, I can explain your exact nutrient levels, reference ranges, and findings!"
            )

        # 1. Borderline vs Deficient explanation
        if "difference" in q and ("borderline" in q or "deficient" in q or "deficiency" in q):
            return self._explain_borderline_vs_deficient() + DISCLAIMER_TEXT

        # 2. Which nutrients need attention / What is low / abnormal?
        if any(w in q for w in ["which nutrient", "what is low", "what is abnormal", "need attention", "needs attention", "concerning", "flagged", "what's wrong", "any deficiency"]):
            return self._list_attention_nutrients() + DISCLAIMER_TEXT

        # 3. Explain report in simple English / overall summary
        if any(w in q for w in ["explain my report", "explain report", "simple english", "in simple words", "summary", "summarize", "break down"]):
            return self._explain_report_simple() + DISCLAIMER_TEXT

        # 4. What foods should I eat / Dietary questions
        if any(w in q for w in ["food", "foods", "eat", "diet", "nutrition", "meal", "recipe", "dietary"]):
            target = self._detect_nutrient_in_query(q)
            return self._explain_diet(target) + DISCLAIMER_TEXT

        # 5. Questions to ask the doctor
        if any(w in q for w in ["ask my doctor", "ask doctor", "visit", "questions for doctor", "physician", "appointment"]):
            return self._suggest_doctor_questions() + DISCLAIMER_TEXT

        # 6. Specific Nutrient query (e.g. "Why is my Vitamin B12 low?", "What does Vitamin D mean?")
        target = self._detect_nutrient_in_query(q)
        if target:
            if any(w in q for w in ["why", "cause", "reason", "low", "deficient"]):
                return self._explain_why_nutrient_low(target) + DISCLAIMER_TEXT
            if any(w in q for w in ["what does", "meaning", "mean", "result", "level", "value", "status"]):
                return self._explain_nutrient_meaning(target) + DISCLAIMER_TEXT
            # Generic inquiry about this nutrient
            return self._explain_nutrient_overview(target) + DISCLAIMER_TEXT

        # 7. Quality / Reliability / Confidence questions
        if any(w in q for w in ["confidence", "reliable", "accuracy", "accurate", "mistake", "trust"]):
            return self._explain_confidence_and_quality() + DISCLAIMER_TEXT

        # 8. Fallback: Contextual general answer
        return self._general_contextual_response(query) + DISCLAIMER_TEXT

    # -------------------------------------------------------------------------
    # Response Generators
    # -------------------------------------------------------------------------

    def _explain_borderline_vs_deficient(self) -> str:
        return (
            "### ⚖️ The Difference Between Borderline and Deficient\n\n"
            "In clinical laboratory testing, health markers exist along a continuous spectrum rather than a simple on/off switch:\n\n"
            "1. **Deficient (Low)**:\n"
            "   • Your measured level has fallen strictly **below the clinical cut-off threshold**.\n"
            "   • At this level, bodily reserves are depleted and physiological symptoms (such as fatigue, weakness, cognitive changes, or anemia) are likely occurring or imminent.\n"
            "   • **Action typically needed**: Prompt evaluation by your doctor, often requiring therapeutic supplementation or targeted treatment to identify and fix the underlying cause.\n\n"
            "2. **Borderline (Insufficient)**:\n"
            "   • Your level sits in the **transitional 'grey zone'** between optimal health and overt deficiency.\n"
            "   • While you may not yet have severe clinical symptoms, your body's nutrient reserves are suboptimal or trending down.\n"
            "   • **Action typically needed**: Early nutritional adjustments, lifestyle improvements, and re-testing to prevent progressing into full deficiency.\n\n"
            "NutriCheck clearly labels borderline results so you and your doctor can take proactive action before a deficiency becomes severe."
        )

    def _list_attention_nutrients(self) -> str:
        deficient = []
        borderline = []
        normal = []

        all_items = self.result.get("nutrients", []) + self.result.get("findings", [])
        for item in all_items:
            key = item["key"]
            name = NUTRIENT_DISPLAY_NAMES.get(key, item["name"])
            status = item.get("status")
            status_lbl = item.get("status_label", "")

            # Lab values
            labs = self.labs_map.get(key, [])
            lab_details = ""
            if labs:
                usable_labs = [l for l in labs if l.get("usable")]
                if usable_labs:
                    l = usable_labs[0]
                    lab_details = f" (measured: **{l.get('value')} {l.get('unit')}**)"

            if status == "deficient":
                deficient.append(f"• **{name}**: {status_lbl}{lab_details}")
            elif status == "borderline":
                borderline.append(f"• **{name}**: {status_lbl}{lab_details}")
            elif status in ("normal", "not_deficient"):
                normal.append(f"• **{name}**: Normal{lab_details}")

        response = ["### 📋 Summary of Nutrients Needing Attention\n"]

        if deficient:
            response.append("#### 🔴 High Priority (Deficiency Detected):")
            response.extend(deficient)
            response.append("\n*These values are below standard clinical thresholds and should be reviewed promptly with your doctor.*\n")

        if borderline:
            response.append("#### 🟠 Moderate Priority (Borderline / Insufficient):")
            response.extend(borderline)
            response.append("\n*These levels are in the borderline range; early dietary adjustments or low-dose support can prevent further decline.*\n")

        if not deficient and not borderline:
            response.append("✅ **Good news!** No clear nutrient deficiencies were detected in this report.")

        if normal:
            response.append("#### 🟢 Normal / Satisfactory Nutrients:")
            response.extend(normal)

        return "\n".join(response)

    def _explain_report_simple(self) -> str:
        try:
            ex = explain_report(self.result, level="simple_en")
            lines = [f"### 📖 {ex.get('title', 'Your Report in Simple Words')}\n"]
            for sec in ex.get("sections", []):
                lines.append(f"#### {sec.get('title', '')}")
                for p in sec.get("paragraphs", []):
                    txt = p.get("text", "")
                    refs = p.get("evidence_refs", [])
                    if refs:
                        txt += f" *(Source evidence: {', '.join(refs)})*"
                    lines.append(txt)
                lines.append("")
            return "\n".join(lines)
        except Exception:
            # Fallback simple explanation
            return self._list_attention_nutrients()

    def _explain_why_nutrient_low(self, target: str) -> str:
        name = NUTRIENT_DISPLAY_NAMES.get(target, target.capitalize())
        item = self.nutrient_map.get(target, {})
        status = item.get("status", "not_assessed")
        labs = self.labs_map.get(target, [])

        response = [f"### 🔍 Why is your {name} flagged as low?\n"]

        # 1. State the evidence from the report
        if labs:
            usable = [l for l in labs if l.get("usable")]
            if usable:
                l = usable[0]
                val = l.get("value")
                unit = l.get("unit")
                ref = l.get("lab_reference") or "standard clinical cut-off"
                response.append(
                    f"**Your Laboratory Result**: Your report shows a measured level of **{val} {unit}** "
                    f"(reference threshold: {ref}). Because this value is below the clinical deficiency threshold, "
                    f"NutriCheck flagged it as **{STATUS_FRIENDLY.get(status, status)}**.\n"
                )
        elif item.get("evidence"):
            evidence_quotes = item.get("evidence", [])
            response.append(f"**Report Mention**: Your report states: *\"{'; '.join(evidence_quotes)}\"*.\n")
        else:
            response.append(f"Your report does not have a definitive numerical measurement, but clinical context suggests low levels.\n")

        # 2. Medical reasons
        response.append(f"#### Common clinical causes of low {name}:")
        if target == "vitamin_b12":
            response.append(
                "1. **Dietary intake**: Insufficient consumption of animal foods or unfortified vegetarian/vegan diets.\n"
                "2. **Absorption issues**: Lack of intrinsic factor (pernicious anemia), atrophic gastritis, celiac disease, or prior stomach/intestinal surgery.\n"
                "3. **Medications**: Long-term use of metformin (diabetes) or proton pump inhibitors (PPIs/antacids) which reduce B12 absorption in the stomach.\n"
                "4. **Increased demand or age**: Aging naturally decreases stomach acid required to liberate B12 from food proteins."
            )
        elif target == "vitamin_d":
            response.append(
                "1. **Limited sunlight exposure**: Living at higher latitudes, working indoors, or consistent sunscreen use.\n"
                "2. **Skin pigmentation**: Melanin reduces the skin's ability to produce Vitamin D from sunlight.\n"
                "3. **Dietary factors**: Low intake of fatty fish, fortified dairy products, or egg yolks.\n"
                "4. **Underlying conditions**: Malabsorption (inflammatory bowel disease), obesity (Vitamin D gets sequestered in fat tissue), or kidney/liver challenges."
            )
        elif target == "iron":
            response.append(
                "1. **Blood loss**: Heavy menstrual periods, frequent blood donation, or occult gastrointestinal bleeding (e.g. ulcers, hemorrhoids, polyps).\n"
                "2. **Inadequate dietary iron**: Low consumption of bioavailable heme iron or lack of Vitamin C to aid plant iron absorption.\n"
                "3. **Poor intestinal absorption**: Celiac disease, Crohn's disease, or H. pylori infection.\n"
                "4. **Increased physical demand**: Pregnancy, lactation, or vigorous endurance athletic training."
            )
        elif target == "calcium":
            response.append(
                "1. **Vitamin D deficiency**: Vitamin D is mandatory for intestinal calcium absorption; without it, calcium drops.\n"
                "2. **Parathyroid gland issues**: Low parathyroid hormone (hypoparathyroidism).\n"
                "3. **Kidney disease or medications**: Certain diuretics or renal filtration issues.\n"
                "4. **Low dietary intake**: Inadequate dairy or fortified calcium-rich food intake over extended periods."
            )
        elif target == "folate":
            response.append(
                "1. **Inadequate fresh produce**: Folate is destroyed by prolonged cooking; low intake of leafy green vegetables and legumes.\n"
                "2. **Alcohol consumption**: Alcohol interferes with folate absorption and accelerates its breakdown.\n"
                "3. **Malabsorption or genetics**: Celiac disease, sprue, or MTHFR gene variants.\n"
                "4. **Increased requirement**: Pregnancy and breastfeeding significantly increase maternal folate demand."
            )
        else:
            response.append("• Multiple factors including low iron, B12, folate, chronic disease, or blood loss can cause this.")

        # 3. Next steps
        rec = item.get("recommendation")
        if rec:
            response.append(f"\n**Recommended Action**: {rec}")

        return "\n".join(response)

    def _explain_nutrient_meaning(self, target: str) -> str:
        name = NUTRIENT_DISPLAY_NAMES.get(target, target.capitalize())
        item = self.nutrient_map.get(target, {})
        status = item.get("status", "not_assessed")
        labs = self.labs_map.get(target, [])

        response = [f"### 📊 What does your {name} result mean?\n"]

        # Biological role
        role = FUNCTION_DETAILED.get(target, FUNCTION_SIMPLE.get(target, ""))
        if role:
            response.append(f"**Role in your body**: {role}\n")

        # Current status
        response.append(f"**Current Status**: **{STATUS_FRIENDLY.get(status, status)}**")

        if labs:
            usable = [l for l in labs if l.get("usable")]
            if usable:
                l = usable[0]
                val = l.get("value")
                unit = l.get("unit")
                ref = l.get("lab_reference") or "standard clinical range"
                response.append(f"• **Measured value**: {val} {unit}")
                response.append(f"• **Reference range**: {ref}")
        elif item.get("evidence"):
            response.append(f"• **Evidence from report**: {', '.join(item.get('evidence', []))}")

        # What to do
        response.append("\n**Clinical Interpretation**:")
        if status == "deficient":
            response.append(
                f"Your level is significantly below optimal. Left untreated, deficiency can lead to persistent fatigue, "
                f"impaired immunity, or tissue-specific complications. Your doctor will likely recommend supplementation."
            )
        elif status == "borderline":
            response.append(
                f"Your level is borderline. You have not dropped into severe deficiency, but your reserves are depleted. "
                f"Proactive dietary improvements can often restore healthy levels before medical symptoms occur."
            )
        elif status in ("normal", "not_deficient"):
            response.append(
                f"Your level is in the healthy, normal range. Continue maintaining a balanced, nutrient-rich diet."
            )
        else:
            response.append(
                f"This report does not contain conclusive laboratory data for {name}. If you suspect an issue, "
                f"ask your physician to order a targeted blood panel."
            )

        return "\n".join(response)

    def _explain_nutrient_overview(self, target: str) -> str:
        return self._explain_nutrient_meaning(target)

    def _explain_diet(self, target: str | None) -> str:
        if target and target in FOOD_SOURCES:
            name = NUTRIENT_DISPLAY_NAMES.get(target, target.capitalize())
            return (
                f"### 🥗 Dietary Sources for {name}\n\n"
                f"Here are the best evidence-based dietary sources to support your {name} levels:\n\n"
                f"{FOOD_SOURCES[target]}\n\n"
                f"*Always check with your physician or a registered dietitian before starting high-dose supplements, "
                f"especially if you take other prescription medications.*"
            )

        # Multi-nutrient dietary summary based on what is low or borderline
        low_or_borderline = [
            k for k, item in self.nutrient_map.items()
            if item.get("status") in ("deficient", "borderline")
        ]

        if not low_or_borderline:
            low_or_borderline = ["vitamin_b12", "vitamin_d", "iron"]

        response = ["### 🥗 Personalized Food Recommendations for Your Report\n"]
        for k in low_or_borderline:
            if k in FOOD_SOURCES:
                name = NUTRIENT_DISPLAY_NAMES.get(k, k.capitalize())
                status = self.nutrient_map.get(k, {}).get("status", "deficient")
                response.append(f"#### For {name} ({STATUS_FRIENDLY.get(status, status)}):")
                response.append(FOOD_SOURCES[k])
                response.append("")

        return "\n".join(response)

    def _suggest_doctor_questions(self) -> str:
        response = ["### 🩺 Smart Questions to Ask Your Doctor at Your Next Visit\n"]
        response.append("Take these personalized questions with you to have a productive discussion with your healthcare provider:\n")

        all_items = self.result.get("nutrients", []) + self.result.get("findings", [])
        deficient = [i for i in all_items if i.get("status") == "deficient"]
        borderline = [i for i in all_items if i.get("status") == "borderline"]

        q_count = 1
        if deficient:
            names = ", ".join(NUTRIENT_DISPLAY_NAMES.get(i["key"], i["name"]) for i in deficient)
            response.append(f"{q_count}. **Deficiency Confirmation**: \"My report shows low levels of **{names}**. What do you suspect is the primary underlying cause (diet, absorption, or medication interactions)?\"")
            q_count += 1
            response.append(f"{q_count}. **Supplementation**: \"Do I need prescription-strength therapeutic supplements, or can this be managed with over-the-counter vitamins and diet? What exact dosage and form (e.g. methylcobalamin vs cyanocobalamin, D3 vs D2) do you recommend?\"")
            q_count += 1

        if borderline:
            names = ", ".join(NUTRIENT_DISPLAY_NAMES.get(i["key"], i["name"]) for i in borderline)
            response.append(f"{q_count}. **Borderline Levels**: \"My **{names}** is in the borderline range. Should we begin proactive low-dose supplementation now, or should we monitor it with repeat blood work in 3 to 6 months?\"")
            q_count += 1

        response.append(f"{q_count}. **Follow-Up Testing**: \"When should I get repeat blood tests done to confirm that my levels are returning to the optimal target range?\"")
        q_count += 1
        response.append(f"{q_count}. **Symptom Review**: \"Could my current symptoms (like fatigue, sleep changes, muscle aches, or mood shifts) be related to these test results?\"")

        return "\n".join(response)

    def _explain_confidence_and_quality(self) -> str:
        quality = self.result.get("quality", {})
        verdict = quality.get("verdict", "ok")
        msg = quality.get("message", "All quality checks passed.")
        needs_review = self.result.get("needs_human_review", False)

        return (
            f"### 🛡️ Analysis Reliability & Confidence\n\n"
            f"• **Quality Status**: {verdict.upper()} — {msg}\n"
            f"• **Needs Human Review**: {'Yes (some values or formats require clinician confirmation)' if needs_review else 'No (high confidence extraction)'}\n\n"
            f"NutriCheck uses evidence-based rule extraction paired with a trained clinical classifier. "
            f"Every single finding is tied to an exact line, quote, or laboratory table in your report. "
            f"If an OCR scan is blurry or an unusual unit is used, NutriCheck abstains or flags it for review rather than guessing."
        )

    def _general_contextual_response(self, query: str) -> str:
        # Check summary
        summary = self.result.get("summary", "")
        nutrients = self.result.get("nutrients", [])
        def_count = sum(1 for n in nutrients if n.get("status") == "deficient")
        bor_count = sum(1 for n in nutrients if n.get("status") == "borderline")

        return (
            f"I have reviewed your medical report. The analysis identified **{def_count} deficiency(ies)** "
            f"and **{bor_count} borderline result(s)**.\n\n"
            f"You can ask me specific questions such as:\n"
            f"• *\"Why is my Vitamin B12 low?\"*\n"
            f"• *\"What does my Vitamin D result mean?\"*\n"
            f"• *\"Which nutrients need attention?\"*\n"
            f"• *\"Explain my report in simple English\"*\n"
            f"• *\"What foods should I eat to improve these levels?\"*\n"
            f"• *\"What questions should I ask my doctor?\"*\n"
            f"• *\"What is the difference between borderline and deficient?\"*"
        )

