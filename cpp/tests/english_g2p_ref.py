#!/usr/bin/env python3
"""Reference IPA for C++ English rule-G2P parity (invoked with a UTF-8 file path as argv[1]).

Mirrors :cpp:class:`EnglishRuleG2p` with **no** heteronym/OOV ONNX: per-token
:meth:`~english_rule_g2p.EnglishLexiconRuleG2p.g2p` (first merged lexicon IPA,
then numbers, then hand OOV rules). Must stay aligned with
``EnglishRuleG2p::text_to_ipa`` when both ONNX models are absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cmudict_ipa import normalize_word_for_lookup  # noqa: E402

from english_rule_g2p import EnglishLexiconRuleG2p  # noqa: E402

_DICT = ROOT / "models" / "en_us" / "dict_filtered_heteronyms.tsv"
if not _DICT.is_file():
    _DICT = ROOT / "data" / "en_us" / "dict_filtered_heteronyms.tsv"
_HOMOGRAPH = ROOT / "models" / "en_us" / "heteronym" / "homograph_index.json"


def split_text_to_words_cpp_compat(text: str) -> list[str]:
    """
    Match ``moonshine_g2p::split_text_to_words``: split only on ASCII whitespace
    bytes, not full Unicode whitespace (``str.split()`` also splits U+00A0, thin
    space, etc., which would desync wiki parity with C++).
    """
    out: list[str] = []
    current = bytearray()
    in_token = False
    for b in text.encode("utf-8"):
        if b < 128 and chr(b).isspace():
            if in_token:
                out.append(current.decode("utf-8"))
                current.clear()
                in_token = False
        else:
            current.append(b)
            in_token = True
    if in_token:
        out.append(current.decode("utf-8"))
    return out


def text_to_ipa_line(g2p: EnglishLexiconRuleG2p, line: str) -> str:
    parts: list[str] = []
    for token in split_text_to_words_cpp_compat(line):
        if not normalize_word_for_lookup(token):
            continue
        parts.append(g2p.g2p(token))
    return " ".join(parts)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(2)
    path = Path(sys.argv[1])
    raw = path.read_text(encoding="utf-8")
    g2p = EnglishLexiconRuleG2p(
        dict_path=_DICT,
        heteronym_index_path=_HOMOGRAPH,
        use_onnx_oov=False,
    )
    if len(sys.argv) >= 3 and sys.argv[2] == "--first-lines":
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        for line in raw.splitlines()[:n]:
            print(text_to_ipa_line(g2p, line))
        return
    print(text_to_ipa_line(g2p, raw.rstrip("\n")), end="")


if __name__ == "__main__":
    main()
