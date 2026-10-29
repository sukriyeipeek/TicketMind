"""Arayüz olmadan tek bir mesajı incele."""

import argparse
import json
from pathlib import Path

from ticketmind.data import DEFAULT_MODEL
from ticketmind.service import TicketService


def main() -> None:
    parser = argparse.ArgumentParser(description="Türkçe destek mesajını analiz et.")
    parser.add_argument("text")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    try:
        result = TicketService(args.model).analyze(args.text)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"{error}\nÖnce python -m ticketmind.train komutunu çalıştırın.\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

