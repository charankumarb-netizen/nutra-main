"""NutriCheck NLP - nutrient deficiency detection from English medical text."""

__version__ = "1.0.0"

from .pipeline import NutriCheckNLP  # noqa: E402,F401
from .chatbot import NutriCheckChatbot  # noqa: E402,F401

__all__ = ["NutriCheckNLP", "NutriCheckChatbot", "__version__"]
