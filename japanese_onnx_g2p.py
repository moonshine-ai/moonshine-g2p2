#!/usr/bin/env python3
"""
Japanese G2P for vocoder-oriented **IPA** strings using **ONNX Runtime only** (plus numpy).

Pipeline:
1. NFC text → ``KoichiYasuoka/roberta-small-japanese-char-luw-upos`` ONNX → surface LUWs.
2. Each LUW: exact match in ``data/ja/dict.tsv`` (first column → first IPA column) if present.
3. Else if the LUW is **kana-only**, :func:`japanese_kana_to_ipa.katakana_hiragana_to_ipa`.
4. Else **greedy longest-prefix match** against the lexicon (covers many OOV compounds).
5. Skip tokens with no Japanese script (Latin-only symbols, etc.).

This intentionally mirrors the C++ layer for parity tests.
"""

from __future__ import annotations

import argparse
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from japanese_kana_to_ipa import (
    has_japanese_script,
    is_kana_only,
    katakana_hiragana_to_ipa,
)
from japanese_luw_merge import merge_for_lexicon_lookup
from japanese_tok_pos import JapaneseTokPosOnnx

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_DICT = _REPO_ROOT / "data" / "ja" / "dict.tsv"
_DEFAULT_MODEL_DIR = _REPO_ROOT / "data" / "ja" / "roberta_japanese_char_luw_upos_onnx"

_TRAILING_PARTICLES = (
    "について",
    "によって",
    "に対して",
    "では",
    "には",
    "から",
    "まで",
    "へは",
    "は",
    "を",
    "に",
    "で",
    "と",
    "が",
    "も",
    "か",
    "や",
    "へ",
)


def load_ja_lexicon_first_ipa(path: Path | str) -> Dict[str, str]:
    """First IPA field per orthography (tab-separated)."""
    p = Path(path)
    out: Dict[str, str] = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line or line.startswith("#") or "\t" not in line:
                continue
            w, ipa = line.split("\t", 1)
            w = w.strip()
            if not w or w in out:
                continue
            ipa0 = ipa.strip().split()[0] if ipa.strip() else ""
            if ipa0:
                out[w] = ipa0
    return out


def _build_prefix_index(lex: Mapping[str, str]) -> Dict[str, List[str]]:
    by_first: Dict[str, List[str]] = defaultdict(list)
    for w in lex:
        if not w:
            continue
        by_first[w[0]].append(w)
    for k in list(by_first.keys()):
        by_first[k].sort(key=len, reverse=True)
    return by_first


class JapaneseOnnxG2p:
    def __init__(
        self,
        *,
        dict_path: Path | None = None,
        model_dir: Path | None = None,
        onnx: JapaneseTokPosOnnx | None = None,
    ) -> None:
        self._dict_path = Path(dict_path) if dict_path is not None else _DEFAULT_DICT
        self._lex = load_ja_lexicon_first_ipa(self._dict_path)
        self._by_first = _build_prefix_index(self._lex)
        self._tok = onnx or JapaneseTokPosOnnx(model_dir=model_dir)

    def g2p_word(self, surface: str) -> str:
        w = unicodedata.normalize("NFC", surface.strip())
        if not w or not has_japanese_script(w):
            return ""
        if w in self._lex:
            return self._lex[w]
        for suf in sorted(_TRAILING_PARTICLES, key=len, reverse=True):
            if len(w) > len(suf) and w.endswith(suf):
                base = w[: -len(suf)]
                ipa_b = self.g2p_word(base)
                ipa_s = self.g2p_word(suf)
                if ipa_b and ipa_s:
                    return ipa_b + ipa_s
                if ipa_b:
                    return ipa_b + (ipa_s or katakana_hiragana_to_ipa(suf))
                break
        if is_kana_only(w):
            return katakana_hiragana_to_ipa(w)
        i = 0
        parts: List[str] = []
        while i < len(w):
            c = w[i]
            found = False
            for cand in self._by_first.get(c, []):
                if w.startswith(cand, i):
                    parts.append(self._lex[cand])
                    i += len(cand)
                    found = True
                    break
            if found:
                continue
            if is_kana_only(w[i : i + 1]):
                parts.append(katakana_hiragana_to_ipa(w[i]))
            i += 1
        return "".join(parts)

    def text_to_ipa(self, text: str) -> str:
        raw = unicodedata.normalize("NFC", text.strip())
        if not raw:
            return ""
        pairs = merge_for_lexicon_lookup(self._tok.annotate(raw))
        ipa_words: List[str] = []
        for surf, _upos in pairs:
            ipa = self.g2p_word(surf)
            if ipa:
                ipa_words.append(ipa)
        return " ".join(ipa_words)


def text_to_ipa(
    text: str,
    *,
    dict_path: Path | None = None,
    model_dir: Path | None = None,
) -> str:
    return JapaneseOnnxG2p(dict_path=dict_path, model_dir=model_dir).text_to_ipa(text)


def main() -> None:
    ap = argparse.ArgumentParser(description="Japanese ONNX + lexicon G2P (vocoder IPA).")
    ap.add_argument("text", nargs="?", default="東京に行きます。")
    ap.add_argument("--dict", type=Path, default=None)
    ap.add_argument("--model-dir", type=Path, default=None)
    args = ap.parse_args()
    g = JapaneseOnnxG2p(dict_path=args.dict, model_dir=args.model_dir)
    print(g.text_to_ipa(args.text))


if __name__ == "__main__":
    main()
