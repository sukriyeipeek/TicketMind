"""Sınıflandırıcı için hafif normalleştirme; orijinal metin değiştirilmez."""

import re
import unicodedata


def normalize_text(text: str) -> str:
    # Python'un varsayılan lower() davranışı Türkçe I/İ için yeterli değildir.
    normalized = unicodedata.normalize("NFC", text).translate(str.maketrans({"I": "ı", "İ": "i"})).lower()
    return re.sub(r"\s+", " ", normalized).strip()

