"""
Tests for NutriCheck AI Chatbot.
Verifies context-awareness, accurate nutrient explanation,
borderline vs deficient distinction, and doctor visit questions.
"""

from __future__ import annotations

import pytest

from nutricheck_nlp import NutriCheckNLP, NutriCheckChatbot


@pytest.fixture()
def sample_analysis():
    engine = NutriCheckNLP()
    report = (
        "Patient: Test Subject\n"
        "Date: 2025-03-12\n"
        "Vitamin B12: 150 pg/mL\n"
        "Vitamin D: 24 ng/mL\n"
        "Ferritin: 85 ng/mL\n"
        "Calcium: 9.6 mg/dL\n"
        "Hemoglobin: 10.5 g/dL\n"
    )
    res = engine.analyze_text(report)
    return res, report


def test_chatbot_initialization(sample_analysis):
    res, report = sample_analysis
    bot = NutriCheckChatbot(result=res, report_text=report)
    assert bot.result is not None
    assert "vitamin_b12" in bot.nutrient_map


def test_chatbot_answers_b12_low(sample_analysis):
    res, report = sample_analysis
    bot = NutriCheckChatbot(result=res, report_text=report)
    ans = bot.answer("Why is my Vitamin B12 low?")
    assert "Vitamin B12" in ans
    assert "150" in ans or "Deficiency" in ans or "cut-off" in ans
    assert "Dietary" in ans or "absorption" in ans or "cause" in ans.lower()


def test_chatbot_answers_vitamin_d_meaning(sample_analysis):
    res, report = sample_analysis
    bot = NutriCheckChatbot(result=res, report_text=report)
    ans = bot.answer("What does my Vitamin D result mean?")
    assert "Vitamin D" in ans
    assert "24" in ans or "Borderline" in ans or "range" in ans.lower()


def test_chatbot_which_nutrients_need_attention(sample_analysis):
    res, report = sample_analysis
    bot = NutriCheckChatbot(result=res, report_text=report)
    ans = bot.answer("Which nutrients need attention?")
    assert "Vitamin B12" in ans
    assert "Vitamin D" in ans or "Hemoglobin" in ans


def test_chatbot_borderline_vs_deficient(sample_analysis):
    res, report = sample_analysis
    bot = NutriCheckChatbot(result=res, report_text=report)
    ans = bot.answer("What is the difference between borderline and deficient?")
    assert "Deficient" in ans
    assert "Borderline" in ans
    assert "cut-off" in ans or "threshold" in ans or "grey zone" in ans


def test_chatbot_explain_simple_english(sample_analysis):
    res, report = sample_analysis
    bot = NutriCheckChatbot(result=res, report_text=report)
    ans = bot.answer("Explain my report in simple English")
    assert len(ans) > 50
    assert "doctor" in ans.lower() or "report" in ans.lower()


def test_chatbot_food_recommendations(sample_analysis):
    res, report = sample_analysis
    bot = NutriCheckChatbot(result=res, report_text=report)
    ans = bot.answer("What foods should I eat to improve these levels?")
    assert "Food" in ans or "Dietary" in ans or "B12" in ans
    assert "salmon" in ans.lower() or "eggs" in ans.lower() or "dairy" in ans.lower() or "sources" in ans.lower()


def test_chatbot_doctor_questions(sample_analysis):
    res, report = sample_analysis
    bot = NutriCheckChatbot(result=res, report_text=report)
    ans = bot.answer("What questions should I ask my doctor?")
    assert "Doctor" in ans or "Visit" in ans
    assert "?" in ans

