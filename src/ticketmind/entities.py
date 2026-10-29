"""Biçimi bilinen alanlar için kurallar; ürün sözlüğü öğrenilmiş NER değildir."""

import re

PATTERNS = {
    "order_id": re.compile(r"(?<![\w-])TM-\d{3,10}(?![\w-])", re.IGNORECASE),
    "error_code": re.compile(r"(?<!\w)E\d{2,5}(?!\w)", re.IGNORECASE),
    "product": re.compile(
        r"(?<!\w)(?:kulaklı[kğ]|telefon|tablet|bilgisayar|klavye|fare|monitör|hoparlör|şarj cihazı|akıllı saat)\w*(?!\w)",
        re.IGNORECASE,
    ),
}


def extract_entities(text: str) -> list[dict]:
    entities = []
    for field, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            entities.append({
                "field": field,
                "value": match.group(),
                "start": match.start(),
                "end": match.end(),
                "method": "rule" if field != "product" else "demo_dictionary",
            })
    return sorted(entities, key=lambda entity: (entity["start"], entity["field"]))

