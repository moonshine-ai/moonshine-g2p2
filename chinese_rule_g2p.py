#!/usr/bin/env python3
"""
Rule- and lexicon-based **Simplified Chinese** grapheme-to-phoneme (Mandarin IPA).

* **Preprocessing** uses the exported CTB9 ONNX pipeline (same as
  ``scripts/chinese_hanlp_ws_pos_onnx.py``): word segmentation + Penn Chinese
  Treebank–style POS tags. No HanLP or PyTorch at inference.
* **In-vocabulary** multi-character words use ``data/zh_hans/dict.tsv``
  (``word<TAB>ipa`` from `ipa-dict` / open-dict-data). When a surface form has
  several IPAs, **POS-guided heuristics** pick among common readings (e.g. 行
  xíng/háng, 了 le/liǎo, 没 méi/mò). Otherwise the **first** lexicon line wins.
* **Out-of-vocabulary** words fall back to **per-character** lookup in the same
  TSV (syllables joined with spaces). Characters still missing are skipped
  (punctuation-only tokens yield empty IPA).
* **Arabic numerals** (ASCII or fullwidth ``０-９``): optional sign, ``.`` or ``,``
  as decimal point; optional thousands separators ``,`` / ``_``. Runs with more
  than one leading zero (e.g. ``007``) are read **digit-by-digit**. Integers use
  standard **Mandarin cardinal** Han characters (万 / 亿 grouping), then IPA via
  the same lexicon. Decimals use ``点`` + digits one-by-one after the point.

Requires: ``onnxruntime``, ``numpy``, ``tokenizers`` (default tokenizer backend).

Example::

    python chinese_rule_g2p.py --model-dir models/zh_hans/hanlp_ctb9_electra_small \\
        "上海是一座城市。"

Compare to **pypinyin + dragonmapper** and **espeak-ng** syllable counts::

    pip install pypinyin dragonmapper phonemizer
    python scripts/eval_chinese_g2p_compare.py --lines 200 --seed 0
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_DICT_PATH = _REPO_ROOT / "data" / "zh_hans" / "dict.tsv"
_DEFAULT_MODEL_DIR = _REPO_ROOT / "models" / "zh_hans" / "hanlp_ctb9_electra_small"
_SCRIPTS_DIR = _REPO_ROOT / "scripts"

# POS tags where the token is usually punctuation / filler with no dictionary IPA.
_SKIP_PHONETIC_POS = frozenset({"PU", "SP", "URL", "EM", "NOI"})

_CN_DIGITS = "零一二三四五六七八九"
_MAX_CARDINAL_MAGNITUDE = 10**16  # beyond this → digit-by-digit readout


def _ascii_digits(s: str) -> str:
    """Map fullwidth digits to ASCII ``0-9``."""
    return s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))


def _strip_thousands_separators(s: str) -> str:
    return s.replace(",", "").replace("_", "").replace(" ", "")


def _section_under_10000(n: int) -> str:
    """
    Write ``1 <= n <= 9999`` as Mandarin Han numerals (no 万/亿).

    Handles 一十/十, 一百一十, 一千零一十, 一千零一十一, etc.
    """
    if n <= 0 or n >= 10000:
        raise ValueError(n)
    CN = _CN_DIGITS
    q, r = divmod(n, 1000)
    parts: list[str] = []
    if q:
        parts.append(CN[q] + "千")
        if r > 0 and r < 100:
            parts.append("零")
    b, r2 = divmod(r, 100)
    if b:
        parts.append(CN[b] + "百")
        if r2 > 0 and r2 < 10:
            parts.append("零")
    s, t = divmod(r2, 10)
    after_ling = bool(parts and parts[-1] == "零")
    tens_prefix = bool(q or b or after_ling)
    if s == 0:
        if t:
            parts.append(CN[t])
    elif s == 1:
        if t == 0:
            parts.append("一十" if tens_prefix else "十")
        else:
            parts.append(("一" if after_ling else "") + "十" + CN[t])
    else:
        parts.append(CN[s] + "十" + (CN[t] if t else ""))
    return "".join(parts)


def int_to_mandarin_cardinal_han(n: int) -> str:
    """Non-negative integer → simplified cardinal Han string (``零`` … ``亿`` …)."""
    if n < 0:
        raise ValueError(n)
    if n == 0:
        return "零"
    if n >= _MAX_CARDINAL_MAGNITUDE:
        return "".join(_CN_DIGITS[int(c)] for c in str(n))
    low_first: list[int] = []
    x = n
    while x > 0:
        low_first.append(x % 10000)
        x //= 10000
    gs = list(reversed(low_first))  # high → low 4-digit blocks
    units = ["", "万", "亿", "兆", "京", "垓"]
    parts: list[str] = []
    zero_pending = False
    for i, g in enumerate(gs):
        if g == 0:
            if parts:
                zero_pending = True
            continue
        if zero_pending:
            parts.append("零")
            zero_pending = False
        if i > 0 and g < 1000 and parts:
            parts.append("零")
        u = units[len(gs) - 1 - i] if len(gs) - 1 - i < len(units) else units[-1]
        parts.append(_section_under_10000(g) + u)
    return "".join(parts)


def arabic_numeral_token_to_han(s: str) -> str | None:
    """
    Normalize an Arabic-numeral token (ASCII / fullwidth) to a Mandarin reading in Han characters.

    Returns ``None`` if *s* is not a supported numeral token.
    """
    raw = normalize_zh_key(s)
    if not raw:
        return None
    t = _ascii_digits(raw)
    t = _strip_thousands_separators(t)
    if not re.fullmatch(r"[+-]?[0-9]+(?:[.,][0-9]+)?", t):
        return None

    sign_neg = t.startswith("-")
    if t[0] in "+-":
        t = t[1:]

    dec_sep = None
    for sep in (".", ","):
        if sep in t:
            if dec_sep is not None:
                return None
            dec_sep = sep
    if dec_sep:
        whole, frac = t.split(dec_sep, 1)
        if not whole.isdigit() or not frac.isdigit():
            return None
    else:
        whole, frac = t, ""

    if not whole.isdigit():
        return None

    if whole != "0" and whole.startswith("0"):
        if frac:
            return None
        han = "".join(_CN_DIGITS[int(c)] for c in whole)
        return ("负" if sign_neg else "") + han

    w = int(whole) if whole else 0
    body: list[str] = []
    if sign_neg:
        body.append("负")
    if frac:
        body.append(int_to_mandarin_cardinal_han(w) if whole else "零")
        body.append("点")
        body.extend(_CN_DIGITS[int(c)] for c in frac)
    else:
        body.append(int_to_mandarin_cardinal_han(w))
    return "".join(body)


def han_reading_to_ipa(han: str, lex: Mapping[str, list[str]]) -> str | None:
    """Han-only Mandarin string → space-separated IPA syllables; ``None`` if any char missing."""
    if not han:
        return None
    syllables: list[str] = []
    for ch in han:
        rows = lex.get(ch)
        if not rows:
            return None
        syllables.append(disambiguate_heteronym(ch, None, rows))
    return " ".join(syllables)


def arabic_numeral_token_to_ipa(s: str, lex: Mapping[str, list[str]]) -> str | None:
    """Arabic / fullwidth numeral token → IPA using cardinal rules + lexicon."""
    han = arabic_numeral_token_to_han(s)
    if han is None:
        return None
    return han_reading_to_ipa(han, lex)

# Verb-like tags (xíng “walk/OK”, méi “not have”, …).
_VERB_LIKE_POS = frozenset(
    {"VV", "VA", "VE", "VC", "LB", "BA", "SB", "MSP", "AS", "DER", "DEV", "DEC"}
)
# Noun-like tags (háng “row/industry”, …).
_NOUN_LIKE_POS = frozenset({"NN", "NR", "NT", "LC", "OD", "M", "CD", "DT", "PN"})


def _ensure_scripts_path() -> None:
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))


def normalize_zh_key(word: str) -> str:
    """NFC-stripped surface form for lexicon keys (Simplified Chinese)."""
    return unicodedata.normalize("NFC", word.strip())


def load_zh_hans_lexicon(path: Path | None = None) -> dict[str, list[str]]:
    """
    Load ``word\\tIPA`` TSV; duplicate words become ordered IPA lists (file order).
    """
    p = path or _DEFAULT_DICT_PATH
    if not p.is_file():
        raise FileNotFoundError(f"Chinese lexicon not found: {p}")
    m: dict[str, list[str]] = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            surf, ipa = parts[0].strip(), parts[1].strip()
            k = normalize_zh_key(surf)
            if not k:
                continue
            m.setdefault(k, []).append(ipa)
    return m


_LEXICON_CACHE: dict[str, list[str]] | None = None
_LEXICON_PATH: Path | None = None


def _get_lexicon(dict_path: Path | None = None) -> dict[str, list[str]]:
    global _LEXICON_CACHE, _LEXICON_PATH
    p = dict_path or _DEFAULT_DICT_PATH
    if _LEXICON_CACHE is not None and _LEXICON_PATH == p:
        return _LEXICON_CACHE
    _LEXICON_CACHE = load_zh_hans_lexicon(p)
    _LEXICON_PATH = p
    return _LEXICON_CACHE


def _is_cjk_char(ch: str) -> bool:
    if len(ch) != 1:
        return False
    cp = ord(ch)
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0x2B740 <= cp <= 0x2B81F)
        or (0x2B820 <= cp <= 0x2CEAF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)
    )


def _reading_contains_any(ipa: str, needles: tuple[str, ...]) -> bool:
    return any(n in ipa for n in needles)


def disambiguate_heteronym(word: str, pos: str | None, readings: Sequence[str]) -> str:
    """
    Pick one IPA for *word* given CTB-style *pos* and parallel *readings* from TSV order.

    Heuristics are intentionally small; unknown cases use ``readings[0]``.
    """
    if not readings:
        return ""
    if len(readings) == 1:
        return readings[0]
    pos = pos or ""
    w = normalize_zh_key(word)

    if w == "行":
        hang = [r for r in readings if _reading_contains_any(r, ("xɑŋ", "xɤŋ"))]
        xing = [r for r in readings if "ɕɪŋ" in r]
        if pos in _NOUN_LIKE_POS and hang:
            return hang[0]
        if pos in _VERB_LIKE_POS and xing:
            return xing[0]
        return readings[0]

    if w == "了":
        le = [r for r in readings if "lɤ" in r and "ljɑʊ" not in r]
        liao = [r for r in readings if "ljɑʊ" in r]
        if pos in ("AS", "SP", "ETC") and le:
            return le[0]
        if pos == "VV" and liao:
            return liao[0]
        return readings[0]

    if w == "没":
        mei = [r for r in readings if "meɪ" in r or r.startswith("mɤ")]
        mo = [r for r in readings if "mɔ" in r]
        if pos in _VERB_LIKE_POS and mei:
            return mei[0]
        if pos in _NOUN_LIKE_POS and mo:
            return mo[0]
        return readings[0]

    if w == "着":
        zhao = [r for r in readings if "ʈʂɑʊ" in r]
        zhe_zhuo = [r for r in readings if "ʈʂɤ" in r or "ʈʂuɔ" in r]
        if pos in ("AS", "MSP", "ETC") and zhe_zhuo:
            return zhe_zhuo[0]
        if pos in _VERB_LIKE_POS and zhao:
            return zhao[0]
        return readings[0]

    if w == "地":
        de_particle = [r for r in readings if "tɤ" in r]
        di_noun = [r for r in readings if "ti˥˩" in r]
        if pos == "DEV" and de_particle:
            return de_particle[0]
        if pos in _NOUN_LIKE_POS and di_noun:
            return di_noun[0]
        return readings[0]

    if w == "得":
        de_particle = [r for r in readings if "tɤ" in r or "tə" in r]
        dei = [r for r in readings if "teɪ" in r or "tɛɪ" in r]
        if pos in ("DER", "DEV", "AS") and de_particle:
            return de_particle[0]
        if pos in _VERB_LIKE_POS and dei:
            return dei[0]
        return readings[0]

    if w == "长":
        zhang = [r for r in readings if "ʈʂɑŋ" in r or "tʂɑŋ" in r]
        chang = [r for r in readings if "ʈʂʰɑŋ" in r or "tʂʰɑŋ" in r]
        if pos in _VERB_LIKE_POS and chang:
            return chang[0]
        if pos in _NOUN_LIKE_POS and zhang:
            return zhang[0]
        return readings[0]

    if w == "数":
        shu3_verb = [r for r in readings if "ʂu˨˩˦" in r]
        shu4_noun = [r for r in readings if r.startswith("ʂu˥˩") and "ʂuɔ" not in r and "tsʰ" not in r]
        if pos in _VERB_LIKE_POS and shu3_verb:
            return shu3_verb[0]
        if pos in _NOUN_LIKE_POS and shu4_noun:
            return shu4_noun[0]
        return readings[0]

    return readings[0]


def char_fallback_ipa(word: str, lex: Mapping[str, list[str]]) -> str | None:
    """Concatenate single-character lookups with spaces; None if any CJK char missing."""
    k = normalize_zh_key(word)
    if not k:
        return None
    syllables: list[str] = []
    for ch in k:
        if not _is_cjk_char(ch):
            return None
        rows = lex.get(ch)
        if not rows:
            return None
        syllables.append(disambiguate_heteronym(ch, None, rows))
    return " ".join(syllables)


class ChineseOnnxLexiconG2p:
    """
    Segment + POS with ONNX, then IPA via ``data/zh_hans/dict.tsv`` (+ char fallback).
    """

    def __init__(
        self,
        *,
        dict_path: Path | str | None = None,
        model_dir: Path | str | None = None,
        lexicon: Mapping[str, list[str]] | None = None,
        tokenizer_backend: str = "json",
        onnx_providers: list[str] | None = None,
    ) -> None:
        _ensure_scripts_path()
        from chinese_hanlp_ws_pos_onnx import HanlpCtb9OnnxWsPos

        self._dict_path = Path(dict_path) if dict_path else _DEFAULT_DICT_PATH
        self._lex: Mapping[str, list[str]] = lexicon if lexicon is not None else _get_lexicon(self._dict_path)
        md = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        self._ws_pos = HanlpCtb9OnnxWsPos(md, providers=onnx_providers, tokenizer_backend=tokenizer_backend)

    def reload_lexicon(self) -> None:
        global _LEXICON_CACHE, _LEXICON_PATH
        _LEXICON_CACHE = None
        _LEXICON_PATH = None
        self._lex = _get_lexicon(self._dict_path)

    def g2p_word(self, word: str, pos: str | None) -> str:
        """Single segmented token → IPA string (may contain spaces for char fallback)."""
        w = normalize_zh_key(word)
        if not w:
            return ""
        if pos and pos in _SKIP_PHONETIC_POS:
            if all(not _is_cjk_char(c) for c in w):
                return ""

        rows = self._lex.get(w)
        if rows:
            return disambiguate_heteronym(w, pos, rows)

        fb = char_fallback_ipa(w, self._lex)
        if fb is not None:
            return fb

        num_ipa = arabic_numeral_token_to_ipa(w, self._lex)
        if num_ipa is not None:
            return num_ipa

        if re.fullmatch(r"[A-Za-z]+", w):
            return w.lower()

        return ""

    def sentence_to_ipa(self, sentence: str, *, joiner: str = " ") -> str:
        """Full sentence: ONNX segment + POS, then lexicon G2P; non-empty syllables joined."""
        ann = self._ws_pos.segment_and_tag(sentence)
        parts: list[str] = []
        for t in ann.tokens:
            ipa = self.g2p_word(t.word, t.pos)
            if ipa:
                parts.append(ipa)
        return joiner.join(parts)

    def sentence_to_token_ipas(
        self, sentence: str
    ) -> list[tuple[str, str, str]]:
        """``(word, pos, ipa)`` per token; *ipa* may be empty for punctuation."""
        ann = self._ws_pos.segment_and_tag(sentence)
        return [(t.word, t.pos, self.g2p_word(t.word, t.pos)) for t in ann.tokens]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("text", nargs="*", help="Sentence(s); if empty, read stdin lines")
    p.add_argument("--dict-path", type=Path, default=_DEFAULT_DICT_PATH)
    p.add_argument("--model-dir", type=Path, default=_DEFAULT_MODEL_DIR)
    p.add_argument(
        "--tokenizer",
        choices=("json", "vocab_txt"),
        default="json",
        help="Tokenizer backend for the ONNX preprocessor.",
    )
    p.add_argument(
        "--tokens",
        action="store_true",
        help="Print word, POS, and IPA per line (TSV) instead of one IPA line.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    g2p = ChineseOnnxLexiconG2p(
        dict_path=args.dict_path,
        model_dir=args.model_dir,
        tokenizer_backend=args.tokenizer,
    )
    lines: list[str]
    if args.text:
        lines = [" ".join(args.text)]
    else:
        lines = [ln.rstrip("\n\r") for ln in sys.stdin if ln.strip()]

    for line in lines:
        if args.tokens:
            for w, pos, ipa in g2p.sentence_to_token_ipas(line):
                print(f"{w}\t{pos}\t{ipa}")
            print()
        else:
            print(g2p.sentence_to_ipa(line))


if __name__ == "__main__":
    main()
