"""Smoke tests for German / Spanish / Russian cardinal digit expansion."""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_german_expand_glued_vs_spaced() -> None:
    from german_numbers import expand_digit_tokens_in_text

    line = "Im Jahr 1891 und Kapitel3."
    out = expand_digit_tokens_in_text(line)
    assert "eintausend" in out or "hundert" in out
    assert "Kapitel3" in out or "Kapitel 3" not in out.replace("Kapitel3", "")


def test_spanish_expand_matches_katakana_glued() -> None:
    from spanish_numbers import expand_digit_tokens_in_text

    s = "戦場のヴァルキュリア3 ,"
    assert "3" in expand_digit_tokens_in_text(s)
    assert "mil" in expand_digit_tokens_in_text("En 1891")


def test_russian_thousand_suffix() -> None:
    from russian_numbers import expand_cardinal_digits_to_russian_words

    w = expand_cardinal_digits_to_russian_words("1891")
    assert "тысяча" in w
    assert "один" in w or "девяносто" in w
