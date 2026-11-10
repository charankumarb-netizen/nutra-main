"""
Fixed, pre-written explanation text for "Explain my report at my level".

IMPORTANT (scope):
  * Report ANALYSIS is English-only.  Nothing here makes NutriCheck read Telugu or Hindi reports.
  * Telugu and Hindi are OUTPUT templates for the findings the English engine already produced.  Numbers, units, dates and the
    quoted report text are inserted programmatically and are never translated, so the medical meaning cannot drift
    (see explain.verify_explanation).
  * The Telugu / Hindi wording has NOT been reviewed by a clinician or a professional medical translator.  English is authoritative.
"""

from __future__ import annotations

LEVELS = {
    "simple_en": {"language": "English", "label": "Very simple English"},
    "detailed_en": {"language": "English", "label": "Detailed English"},
    "medical_en": {"language": "English", "label": "Medical terminology with definitions"},
    "te": {"language": "Telugu", "label": "Telugu explanation"},
    "hi": {"language": "Hindi", "label": "Hindi explanation"},
}

TRANSLATION_NOTICE = {
    "te": "ఈ తెలుగు వివరణ ముందే సిద్ధం చేసిన టెంప్లేట్ల ఆధారంగా ఉంది; వైద్య నిపుణులు సమీక్షించలేదు. ఆంగ్ల వెర్షన్ ప్రామాణికం. (This Telugu text is template-based and not clinically reviewed; English is authoritative.)",
    "hi": "यह हिंदी व्याख्या पहले से तैयार टेम्पलेट पर आधारित है और किसी चिकित्सक ने इसकी समीक्षा नहीं की है। अंग्रेज़ी संस्करण ही प्रामाणिक है। (This Hindi text is template-based and not clinically reviewed; English is authoritative.)",
}

# ---- what each nutrient / test does ---------------------------------------------------------------------------------
FUNCTION_SIMPLE = {
    "iron": "Iron helps your blood carry oxygen around your body.",
    "vitamin_d": "Vitamin D helps your body absorb calcium and keeps your bones strong.",
    "vitamin_b12": "Vitamin B12 helps your nerves and red blood cells work normally.",
    "calcium": "Calcium is needed for strong bones, muscles and nerves.",
    "folate": "Folate (folic acid) helps your body make new cells, including red blood cells.",
    "anemia": "Hemoglobin is the protein in red blood cells that carries oxygen.",
}
FUNCTION_DETAILED = {
    "iron": "Iron is a component of hemoglobin and of many enzymes. Ferritin reflects the body's stored iron; transferrin saturation shows how much of the iron-carrying protein is filled. Ferritin can also rise with inflammation, so it is read together with other tests.",
    "vitamin_d": "Vitamin D (measured as 25-hydroxyvitamin D) regulates calcium and phosphate absorption and bone health. Levels depend on sun exposure, diet, skin type, kidney and liver function.",
    "vitamin_b12": "Vitamin B12 (cobalamin) is needed for DNA synthesis, red blood cell formation and nerve function. Low levels can follow poor diet, reduced absorption (for example atrophic gastritis) or some medicines such as metformin.",
    "calcium": "Calcium in the blood is tightly controlled by parathyroid hormone and vitamin D. Total calcium depends on albumin, so a low or high value is often interpreted together with albumin or with ionized calcium.",
    "folate": "Folate (vitamin B9) is needed to make DNA and red blood cells. Low levels can follow poor diet, alcohol use, pregnancy or malabsorption, and it is usually interpreted together with vitamin B12.",
    "anemia": "Hemoglobin concentration defines anemia (below about 13 g/dL in men and 12 g/dL in non-pregnant women). Anemia has many causes, including low iron, B12 or folate, chronic disease or blood loss.",
}
FUNCTION_HI = {
    "iron": "आयरन आपके खून को शरीर में ऑक्सीजन ले जाने में मदद करता है।",
    "vitamin_d": "विटामिन डी कैल्शियम के अवशोषण में मदद करता है और हड्डियों को मज़बूत रखता है।",
    "vitamin_b12": "विटामिन बी12 नसों और लाल रक्त कोशिकाओं के सही ढंग से काम करने के लिए ज़रूरी है।",
    "calcium": "कैल्शियम हड्डियों, मांसपेशियों और नसों के लिए ज़रूरी है।",
    "folate": "फोलेट (फोलिक एसिड) नई कोशिकाएँ और लाल रक्त कोशिकाएँ बनाने में मदद करता है।",
    "anemia": "हीमोग्लोबिन लाल रक्त कोशिकाओं में मौजूद वह प्रोटीन है जो ऑक्सीजन ले जाता है।",
}
FUNCTION_TE = {
    "iron": "ఐరన్ మీ రక్తం శరీరంలో ఆక్సిజన్‌ను తీసుకెళ్లడానికి సహాయపడుతుంది.",
    "vitamin_d": "విటమిన్ D కాల్షియం శోషణకు సహాయపడి ఎముకలను బలంగా ఉంచుతుంది.",
    "vitamin_b12": "విటమిన్ B12 నరాలు మరియు ఎర్ర రక్త కణాలు సరిగా పనిచేయడానికి అవసరం.",
    "calcium": "కాల్షియం ఎముకలు, కండరాలు మరియు నరాలకు అవసరం.",
    "folate": "ఫోలేట్ (ఫోలిక్ యాసిడ్) కొత్త కణాలు మరియు ఎర్ర రక్త కణాలు తయారుకావడానికి సహాయపడుతుంది.",
    "anemia": "హిమోగ్లోబిన్ ఎర్ర రక్త కణాలలో ఉండే ప్రోటీన్; ఇది ఆక్సిజన్‌ను తీసుకెళ్తుంది.",
}
NAME_HI = {"iron": "आयरन", "vitamin_d": "विटामिन डी", "vitamin_b12": "विटामिन बी12", "calcium": "कैल्शियम",
           "folate": "फोलेट", "anemia": "हीमोग्लोबिन"}
NAME_TE = {"iron": "ఐరన్", "vitamin_d": "విటమిన్ D", "vitamin_b12": "విటమిన్ B12", "calcium": "కాల్షియం",
           "folate": "ఫోలేట్", "anemia": "హిమోగ్లోబిన్"}
NAME_EN = {"iron": "iron", "vitamin_d": "vitamin D", "vitamin_b12": "vitamin B12", "calcium": "calcium",
           "folate": "folate", "anemia": "hemoglobin"}

# ---- status sentences.  Slots: {name} {test} {value} {unit} {cutoff} {lo} {hi} {date} ---------------------------------
SENTENCES = {
    "simple_en": {
        "lab_deficient": "Your {name} result in the report ({test}) is {value} {unit}. That is below the cut-off this tool uses ({cutoff} {unit}), so it is counted as low.",
        "lab_borderline": "Your {name} result in the report ({test}) is {value} {unit}. That is in the borderline range ({cutoff_lo} to {cutoff_hi} {unit}).",
        "lab_normal": "Your {name} result in the report ({test}) is {value} {unit}. That is not low by the cut-off this tool uses.",
        "lab_high": "Your {name} result in the report ({test}) is {value} {unit}. That is above the usual range; this tool does not treat a high value as a deficiency.",
        "deficient": "The report says you have low {name}.",
        "suspected": "The report hints that {name} may be low, but this needs to be confirmed.",
        "not_deficient": "The report says {name} is not low.",
        "conflicting": "The report gives conflicting information about {name}. Please ask your doctor or the lab to clarify.",
        "not_assessed": "This report does not give enough reliable information to say anything about {name}.",
        "quote": "The report says: “{quote}”",
        "date": "This result is dated {date}.",
        "history": "The report also mentions an earlier low result or deficiency. That is kept separate from the current result.",
        "verify": "Please check this value against the original report before relying on it.",
        "disclaimer": "This is not a diagnosis. Please talk to your doctor about your results.",
        "questions_title": "Questions you could ask your doctor",
        "title": "Your report in simple words",
    },
    "hi": {
        "lab_deficient": "रिपोर्ट में आपका {name} ({test}) {value} {unit} है। यह इस टूल की सीमा ({cutoff} {unit}) से कम है, इसलिए इसे कम माना जाता है।",
        "lab_borderline": "रिपोर्ट में आपका {name} ({test}) {value} {unit} है। यह सीमारेखा (borderline) की श्रेणी ({cutoff_lo} से {cutoff_hi} {unit}) में आता है।",
        "lab_normal": "रिपोर्ट में आपका {name} ({test}) {value} {unit} है। इस टूल की सीमा के अनुसार यह कम नहीं है।",
        "lab_high": "रिपोर्ट में आपका {name} ({test}) {value} {unit} है। यह सामान्य सीमा से ऊपर है; यह टूल ऊँचे मान को कमी नहीं मानता।",
        "deficient": "रिपोर्ट के अनुसार आपका {name} कम है।",
        "suspected": "रिपोर्ट से संकेत मिलता है कि {name} कम हो सकता है, लेकिन इसकी पुष्टि ज़रूरी है।",
        "not_deficient": "रिपोर्ट के अनुसार {name} कम नहीं है।",
        "conflicting": "रिपोर्ट में {name} के बारे में परस्पर विरोधी जानकारी है। कृपया अपने डॉक्टर या लैब से स्पष्टीकरण लें।",
        "not_assessed": "इस रिपोर्ट में {name} के बारे में कुछ कहने के लिए पर्याप्त भरोसेमंद जानकारी नहीं है।",
        "quote": "रिपोर्ट में लिखा है: “{quote}”",
        "date": "यह परिणाम {date} की तारीख का है।",
        "history": "रिपोर्ट में पहले की किसी कम जाँच या कमी का भी ज़िक्र है। उसे मौजूदा परिणाम से अलग रखा गया है।",
        "verify": "इस मान पर भरोसा करने से पहले कृपया मूल रिपोर्ट से इसकी जाँच करें।",
        "disclaimer": "यह निदान नहीं है। कृपया अपने परिणामों के बारे में अपने डॉक्टर से बात करें।",
        "questions_title": "आप अपने डॉक्टर से ये सवाल पूछ सकते हैं",
        "title": "आपकी रिपोर्ट आसान शब्दों में",
    },
    "te": {
        "lab_deficient": "రిపోర్ట్‌లో మీ {name} ({test}) {value} {unit}. ఇది ఈ టూల్ ఉపయోగించే పరిమితి ({cutoff} {unit}) కంటే తక్కువ, కాబట్టి దీన్ని తక్కువగా పరిగణిస్తారు.",
        "lab_borderline": "రిపోర్ట్‌లో మీ {name} ({test}) {value} {unit}. ఇది సరిహద్దు (borderline) స్థాయిలో ({cutoff_lo} నుండి {cutoff_hi} {unit}) ఉంది.",
        "lab_normal": "రిపోర్ట్‌లో మీ {name} ({test}) {value} {unit}. ఈ టూల్ పరిమితి ప్రకారం ఇది తక్కువ కాదు.",
        "lab_high": "రిపోర్ట్‌లో మీ {name} ({test}) {value} {unit}. ఇది సాధారణ పరిధి కంటే ఎక్కువ; ఈ టూల్ ఎక్కువ విలువను లోపంగా పరిగణించదు.",
        "deficient": "రిపోర్ట్ ప్రకారం మీ {name} తక్కువగా ఉంది.",
        "suspected": "రిపోర్ట్ ప్రకారం {name} తక్కువగా ఉండే అవకాశం ఉంది, కానీ నిర్ధారణ అవసరం.",
        "not_deficient": "రిపోర్ట్ ప్రకారం {name} తక్కువ కాదు.",
        "conflicting": "రిపోర్ట్‌లో {name} గురించి పరస్పర విరుద్ధమైన సమాచారం ఉంది. దయచేసి మీ డాక్టర్ లేదా ల్యాబ్‌ను స్పష్టత అడగండి.",
        "not_assessed": "ఈ రిపోర్ట్‌లో {name} గురించి చెప్పడానికి తగినంత నమ్మదగిన సమాచారం లేదు.",
        "quote": "రిపోర్ట్‌లో ఇలా ఉంది: “{quote}”",
        "date": "ఈ ఫలితం {date} తేదీకి చెందినది.",
        "history": "రిపోర్ట్‌లో గతంలో తక్కువ ఫలితం లేదా లోపం ఉన్నట్లు కూడా ఉంది. దాన్ని ప్రస్తుత ఫలితం నుండి వేరుగా ఉంచారు.",
        "verify": "ఈ విలువపై ఆధారపడే ముందు దయచేసి అసలు రిపోర్ట్‌తో సరిచూడండి.",
        "disclaimer": "ఇది వైద్య నిర్ధారణ కాదు. దయచేసి మీ ఫలితాల గురించి మీ డాక్టర్‌తో మాట్లాడండి.",
        "questions_title": "మీరు మీ డాక్టర్‌ను అడగగల ప్రశ్నలు",
        "title": "మీ రిపోర్ట్ సులభమైన మాటల్లో",
    },
}
SENTENCES["detailed_en"] = {
    **SENTENCES["simple_en"],
    "lab_deficient": "{test}: {value} {unit}. This is below the cut-off used by this tool ({cutoff} {unit}) and is classed as low for {name}.",
    "lab_borderline": "{test}: {value} {unit}. This falls in the borderline (insufficiency) band of {cutoff_lo} to {cutoff_hi} {unit} used by this tool.",
    "lab_normal": "{test}: {value} {unit}. This is at or above the cut-off used by this tool, so it is not classed as low.",
    "lab_high": "{test}: {value} {unit}. This is above the upper limit used by this tool. A high value is not a deficiency, but a clinician may still want to look at it.",
    "title": "Your report in detail",
}
SENTENCES["medical_en"] = {
    **SENTENCES["detailed_en"],
    "deficient": "Assessment: {name} deficiency documented in the text.",
    "suspected": "Assessment: possible {name} deficiency; confirmation required.",
    "not_deficient": "Assessment: no {name} deficiency documented.",
    "conflicting": "Assessment: discordant documentation regarding {name} (text versus laboratory, or conflicting measurements); clinical clarification required.",
    "not_assessed": "Assessment: insufficient reliable data to assess {name}.",
    "title": "Report summary with medical terminology",
}

GLOSSARY = {
    "ferritin": "A protein that stores iron; the blood level reflects the body's iron stores.",
    "transferrin saturation": "The percentage of the iron-carrying protein (transferrin) that is loaded with iron.",
    "serum": "The clear liquid part of blood that is tested in the laboratory.",
    "deficiency": "A level below what the body needs, judged against a cut-off.",
    "borderline": "Close to the lower limit; neither clearly low nor clearly adequate.",
    "reference range": "The interval of values the laboratory regards as typical for healthy people.",
    "hemoglobin": "The oxygen-carrying protein in red blood cells.",
    "anemia": "A hemoglobin level below the cut-off for age and sex.",
    "25-hydroxyvitamin D": "The blood form of vitamin D that is measured to judge vitamin D status.",
    "cobalamin": "Another name for vitamin B12.",
    "ionized calcium": "The active, unbound fraction of calcium in the blood.",
    "hypocalcemia": "A calcium level below the reference range.",
    "hypoferritinemia": "A ferritin level below the cut-off (low iron stores).",
    "folate": "Vitamin B9, needed to make DNA and red blood cells.",
}
TERMS_BY_STATUS_AND_KEY = {
    "iron": ["ferritin", "serum", "deficiency", "reference range"],
    "vitamin_d": ["25-hydroxyvitamin D", "deficiency", "borderline", "reference range"],
    "vitamin_b12": ["cobalamin", "deficiency", "borderline", "reference range"],
    "calcium": ["serum", "hypocalcemia", "ionized calcium", "reference range"],
    "folate": ["folate", "deficiency", "borderline", "reference range"],
    "anemia": ["hemoglobin", "anemia", "reference range"],
}

# ---- generic doctor-visit questions -----------------------------------------------------------------------------------
QUESTIONS = {
    "q_range": {
        "en": "Is this result outside the reference range for me?",
        "hi": "क्या यह परिणाम मेरे लिए सामान्य सीमा से बाहर है?",
        "te": "ఈ ఫలితం నాకు సాధారణ పరిధికి వెలుపల ఉందా?",
    },
    "q_compare": {
        "en": "Should this measurement be compared with my previous report?",
        "hi": "क्या इस माप की तुलना मेरी पिछली रिपोर्ट से की जानी चाहिए?",
        "te": "ఈ కొలతను నా మునుపటి రిపోర్ట్‌తో పోల్చాలా?",
    },
    "q_context": {
        "en": "Could other tests or my medical history change how this result is interpreted?",
        "hi": "क्या अन्य जाँचें या मेरा चिकित्सा इतिहास इस परिणाम की व्याख्या बदल सकते हैं?",
        "te": "ఇతర పరీక్షలు లేదా నా వైద్య చరిత్ర ఈ ఫలితం యొక్క అర్థాన్ని మార్చగలవా?",
    },
    "q_repeat": {
        "en": "Do I need a repeat test or confirmation of this value?",
        "hi": "क्या मुझे यह जाँच दोबारा करानी चाहिए या इस मान की पुष्टि करानी चाहिए?",
        "te": "ఈ పరీక్షను మళ్లీ చేయించుకోవాలా లేదా ఈ విలువను నిర్ధారించుకోవాలా?",
    },
    "q_medicines": {
        "en": "Do my medicines or my diet affect this result?",
        "hi": "क्या मेरी दवाएँ या खान-पान इस परिणाम को प्रभावित करते हैं?",
        "te": "నా మందులు లేదా ఆహారం ఈ ఫలితాన్ని ప్రభావితం చేస్తాయా?",
    },
}
