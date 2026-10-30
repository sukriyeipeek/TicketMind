"""Stable import path for English preprocessing in saved model artifacts."""
import re
import unicodedata

def normalize_text(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()
