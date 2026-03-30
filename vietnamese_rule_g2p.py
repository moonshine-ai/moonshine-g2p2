#!/usr/bin/env python3
"""
Vietnamese grapheme-to-phoneme (G2P) — Northern Vietnamese IPA aligned with ``data/vi/dict.tsv``
(ipa-dict style: six tones as Chao digits, labialized codas ``k͡p`` / ``ŋ͡m`` where appropriate).

**Lexicon + longest match:** whitespace splits the input into tokens. Multi-word keys in the
lexicon (e.g. ``tổ chức``) are matched with a **greedy longest span** over the token sequence.
Tokens may carry leading/trailing punctuation; only the core substring is looked up.

**OOV syllables:** rule-based syllable G2P mirrors the conventions used in the bundled lexicon
(``ch``/``tr`` → /c/, ``kh`` → /x/, ``nh`` onset → /ɲ/, ``nh`` coda → /ŋ/, ``gi``+vowel → /z/,
except ``gì``-style nuclei → /ɣi/, etc.). This is **best-effort**; rare spellings may differ.

**Dependencies:** stdlib only (no ``transformers``). ``onnxruntime`` is **not** required here;
Vietnamese is syllable-local and does not use ONNX in this pipeline.

CLI mirrors other ``*_rule_g2p.py`` modules.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_VI_DICT = _REPO_ROOT / "data" / "vi" / "dict.tsv"

# ---------------------------------------------------------------------------
# Tone (Unicode combining marks, NFD)
# ---------------------------------------------------------------------------

_TONE_COMBINING = {
    "\u0300": 2,  # grave — huyền
    "\u0301": 5,  # acute — sắc
    "\u0303": 4,  # tilde — ngã
    "\u0309": 3,  # hook — hỏi
    "\u0323": 6,  # dot below — nặng
}

_TONE_SUFFIX = {
    1: "\u02e7\u02e7",  # ˧˧ ngang
    2: "\u02e7\u02e8",  # ˧˨ huyền
    3: "\u02e7\u02e9\u02e8",  # ˧˩˨ hỏi
    4: "\u02e7\u02c0\u02e5",  # ˧ˀ˥ ngã (ipa-dict style)
    5: "\u02e6\u02e5",  # ˦˥ sắc (obstruent-coda variant; see ``tone_suffix_sắc``)
    6: "\u02e8\u02c0\u02e9",  # ˨ˀ˩ nặng
}

# Sắc on open syllables / sonorant codas uses ˨˦ in ``dict.tsv``.
_SẮC_OPEN = "\u02e8\u02e6"

# Tie bar (U+0361) between velar and labial in codas, as in the lexicon.
_TIE = "\u0361"


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


def split_tone(s: str) -> tuple[str, int]:
    """Return (NFC base letters with tone marks removed, tone id 1..6)."""
    d = _nfd(s)
    out: list[str] = []
    tone = 1
    for ch in d:
        if ch in _TONE_COMBINING:
            tone = _TONE_COMBINING[ch]
        else:
            out.append(ch)
    return _nfc("".join(out)), tone


def _is_vn_vowel_char(ch: str) -> bool:
    if not ch:
        return False
    cp = ord(ch[0])
    if ch in "yY":
        return True
    # Latin ext Vietnamese blocks + combining handled separately
    if cp in (0x0069, 0x0049):  # i I
        return True
    n = _nfc(ch)
    if len(n) != 1:
        return False
    o = ord(n)
    return (
        0x00C0 <= o <= 0x024F
        or 0x1E00 <= o <= 0x1EFF
        or ch.lower() in "aăâeêioôơuưy"
    )


def _letters_only_lower(s: str) -> str:
    """Lowercase base letters; keep Vietnamese diacritics on vowels."""
    return _nfc(s.lower())


def _rime_is_only_i(remainder: str) -> bool:
    """After onset ``gi``, /ɣi/ if the rime is just ``i`` (+ tones already stripped)."""
    r = remainder.strip()
    if not r:
        return False
    if len(r) == 1 and r.lower() == "i":
        return True
    # Single NFC codepoint for dotted i etc.
    if len(r) <= 3 and r.lower().startswith("i") and all(not c.isalpha() or c == "i" for c in r.lower()):
        # e.g. weird encodings — require only 'i' letters
        alpha = "".join(c for c in r if c.isalpha())
        return alpha == "i"
    return False


def _front_vowel_letter(ch: str) -> bool:
    cl = ch.lower()
    if not cl:
        return False
    base = _nfc(cl)[0]
    return base in "eêéèẻẽẹếềểễệiíìỉĩị"


def parse_onset(body: str) -> tuple[str, str]:
    """
    Parse toneless syllable body into (onset_ipa, rime_orth).
    *body* is NFC lowercase.
    """
    if not body:
        return "", ""

    b = body
    # --- Multi-graph onsets (longest first) ---
    if b.startswith("ngh") and len(b) > 3 and _front_vowel_letter(b[3]):
        return "ŋ", b[3:]
    if b.startswith("ng") and len(b) > 2 and _is_vn_vowel_char(b[2]):
        return "ŋ", b[2:]
    if b.startswith("ch") and len(b) > 2:
        return "c", b[2:]
    if b.startswith("gh") and len(b) > 2 and _front_vowel_letter(b[2]):
        return "\u0263", b[2:]  # ɣ
    if b.startswith("gi"):
        if len(b) == 2:
            # Tone was on ``i``; syllable is /ɣi/, not /zi/.
            return "\u0263", "i"
        rest = b[2:]
        if _rime_is_only_i(rest):
            return "\u0263", rest  # ɣ + i
        return "z", rest
    if b.startswith("qu") and len(b) > 2:
        return "kw", b[2:]
    if b.startswith("tr") and len(b) > 2:
        return "c", b[2:]
    if b.startswith("th") and len(b) > 2:
        return "t\u02b0", b[2:]  # tʰ
    if b.startswith("ph") and len(b) > 2:
        return "f", b[2:]
    if b.startswith("kh") and len(b) > 2:
        return "x", b[2:]
    if b.startswith("nh") and len(b) > 2:
        return "\u0272", b[2:]  # ɲ
    if b.startswith("\u0111") and len(b) > 1:  # đ
        return "d", b[1:]
    if b.startswith("g") and not b.startswith("gh") and not b.startswith("gi"):
        if len(b) > 1:
            return "\u0263", b[1:]
        return "\u0263", ""
    # Single-letter onsets
    if len(b) >= 2 and b[0] in "bcdhklmnpqrstvx":
        ch0 = b[0]
        if ch0 in "ck":
            return "k", b[1:]
        if ch0 == "b":
            return "b", b[1:]
        if ch0 == "d":
            return "z", b[1:]
        if ch0 == "h":
            return "h", b[1:]
        if ch0 == "l":
            return "l", b[1:]
        if ch0 == "m":
            return "m", b[1:]
        if ch0 == "n":
            return "n", b[1:]
        if ch0 == "p":
            return "p", b[1:]
        if ch0 == "r":
            return "z", b[1:]
        if ch0 == "s":
            return "s", b[1:]
        if ch0 == "t":
            return "t", b[1:]
        if ch0 == "v":
            return "v", b[1:]
        if ch0 == "x":
            return "s", b[1:]
    # Vowel-initial
    if _is_vn_vowel_char(b[0]):
        return "", b
    return "", b


_CODAS = ("ch", "nh", "ng", "c", "k", "m", "n", "p", "t")


def parse_rime(rime: str) -> tuple[str, str]:
    """Split rime into (nucleus_orth, coda_orth). *rime* is toneless NFC lowercase."""
    if not rime:
        return "", ""
    for cd in _CODAS:
        if rime.endswith(cd) and len(rime) > len(cd):
            return rime[: -len(cd)], cd
    return rime, ""


# Nucleus orthography → IPA (toneless). Longest multi-letter matches first.
_NUCLEUS_PREFIX = [
    ("iêu", "iəw"),
    ("ươi", "ɯəj"),
    ("ươu", "ɯəw"),
    ("ươ", "ɯə"),
    ("iê", "iə"),
    ("yê", "iə"),
    ("uô", "uo"),
    ("oa", "wa"),
    ("oe", "wɛ"),
    ("uy", "wj"),  # before another vowel handled elsewhere; "uyên" etc. in lexicon
    ("ai", "aj"),
    ("ay", "aj"),
    ("ao", "aw"),
    ("au", "aw"),
    ("âu", "əw"),
    ("ây", "əj"),
    ("ơi", "ɤj"),
    ("ơu", "ɤw"),
    ("ưa", "ɯə"),
    ("ưi", "ɯj"),
    ("ưu", "ɯw"),
    ("ia", "iə"),
    ("iu", "iw"),
    ("êu", "ew"),
    ("ơ", "ɤ"),
    ("ư", "ɯ"),
    ("ô", "o"),
    ("â", "ɤ̆"),
    ("ă", "ɐ"),
    ("ê", "e"),
    ("e", "ɛ"),
    ("o", "ɔ"),
    ("a", "a"),
    ("i", "i"),
    ("u", "u"),
    ("y", "i"),
]


def nucleus_to_ipa(nucleus: str) -> str:
    if not nucleus:
        return ""
    n = nucleus
    for pref, ipa in _NUCLEUS_PREFIX:
        if n.startswith(pref):
            return ipa + nucleus_to_ipa(n[len(pref) :])
    if n:
        # Unknown grapheme — skip
        return nucleus_to_ipa(n[1:])
    return ""


def _nucleus_wants_labial_coda(nucleus_ipa: str) -> bool:
    """Heuristic: /k͡p/ and /ŋ͡m/ after rounded nuclei (lexicon pattern). Excludes /ɯ/."""
    s = nucleus_ipa.rstrip("ː")
    if not s:
        return False
    if s.endswith("ɯ") or "ɯə" in s or s.startswith("ɯ"):
        return False
    if s.endswith(("o", "ɔ", "u", "w")) or s.endswith("əw") or s.endswith("ow"):
        return True
    if "ɔ" in s and not s.endswith("ɯ"):
        return True
    return False


def combine_nucleus_coda(nucleus_orth: str, nucleus_ipa: str, coda: str) -> str:
    """Merge nucleus + coda with a few orthographic specials (``anh``, ``ách``, ``ênh``)."""
    n_letters = "".join(c for c in nucleus_orth if c.isalpha())
    if coda == "nh":
        if n_letters in ("a", "á", "à", "ả", "ã", "ạ"):
            return "ɛŋ"
        if n_letters.startswith("ê") or n_letters in ("ế", "ề", "ể", "ễ", "ệ"):
            return "eŋ"
        return nucleus_ipa + coda_to_ipa_simple("nh", nucleus_ipa)
    if coda == "ch":
        if n_letters in ("a", "á", "à", "ả", "ã", "ạ"):
            return "ɛk"
        return nucleus_ipa + coda_to_ipa_simple("ch", nucleus_ipa)
    return nucleus_ipa + coda_to_ipa_simple(coda, nucleus_ipa)


def coda_to_ipa_simple(coda: str, nucleus_ipa: str) -> str:
    if not coda:
        return ""
    lab = _nucleus_wants_labial_coda(nucleus_ipa)
    if coda == "nh":
        return "ŋ"
    if coda == "ch":
        return "k"
    if coda == "ng":
        return "ŋ" + _TIE + "m" if lab else "ŋ"
    if coda in ("c", "k"):
        return "k" + _TIE + "p" if lab else "k"
    if coda == "n":
        return "n"
    if coda == "m":
        return "m"
    if coda == "p":
        return "p"
    if coda == "t":
        return "t"
    return ""


def _coda_is_obstruent_for_sắc(coda_orth: str) -> bool:
    return coda_orth in ("ch", "c", "k", "p", "t")


def tone_suffix(tone: int, coda_orth: str) -> str:
    if tone == 5 and not _coda_is_obstruent_for_sắc(coda_orth):
        return _SẮC_OPEN
    return _TONE_SUFFIX.get(tone, _TONE_SUFFIX[1])


def apply_tone(
    ipa_base: str,
    tone: int,
    has_coda: bool,
    coda_orth: str = "",
    *,
    nucleus_ipa_for_labial: str = "",
) -> str:
    suf = tone_suffix(tone, coda_orth)
    if tone == 6 and not has_coda and ipa_base:
        return ipa_base + suf + "ʔ"
    if (
        tone == 6
        and has_coda
        and coda_orth == "ng"
        and _nucleus_wants_labial_coda(nucleus_ipa_for_labial)
    ):
        return ipa_base + suf + "ʔ"
    return ipa_base + suf


def vietnamese_syllable_to_ipa(syllable: str) -> str:
    """Single orthographic syllable → IPA (Northern, ipa-dict style). Empty if unparseable."""
    if not syllable or not syllable.strip():
        return ""
    raw = _letters_only_lower(syllable.strip())
    if not raw:
        return ""
    body, tone = split_tone(raw)
    onset_ipa, rime = parse_onset(body)
    if not rime:
        return ""
    nuc_orth, coda_orth = parse_rime(rime)
    nuc_ipa = nucleus_to_ipa(nuc_orth)
    if not nuc_ipa and not onset_ipa and not coda_orth:
        return ""
    if coda_orth:
        rime_ipa = combine_nucleus_coda(nuc_orth, nuc_ipa, coda_orth)
    else:
        rime_ipa = nuc_ipa
    base = onset_ipa + rime_ipa
    if not base:
        return ""
    return apply_tone(
        base, tone, bool(coda_orth), coda_orth, nucleus_ipa_for_labial=nuc_ipa
    )


# ---------------------------------------------------------------------------
# Lexicon + longest match
# ---------------------------------------------------------------------------

_LEXICON_CACHE: dict[str, dict[str, str]] = {}


def _load_lexicon(path: Path) -> dict[str, str]:
    ps = str(path.resolve())
    if ps in _LEXICON_CACHE:
        return _LEXICON_CACHE[ps]
    lex: dict[str, str] = {}
    if not path.is_file():
        _LEXICON_CACHE[ps] = lex
        return lex
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tab = line.find("\t")
            if tab < 0:
                continue
            key = _nfc(line[:tab].strip())
            ipa = line[tab + 1 :].strip()
            if key and key not in lex:
                lex[key] = ipa
    _LEXICON_CACHE[ps] = lex
    return lex


def _max_key_len(lex: dict[str, str]) -> int:
    m = 1
    for k in lex:
        m = max(m, len(k.split()))
    return m


_EDGE_PUNCT_RE = re.compile(
    r'^[\s\"“”‘’\'\(\[\{«‹]*(.+?)[\s\"”’\'\)\]\}»›,;:.!?…]*$',
    re.DOTALL,
)


def _strip_edge_punct(token: str) -> str:
    token = token.strip()
    m = _EDGE_PUNCT_RE.match(token)
    if m:
        return m.group(1).strip()
    return token


def _token_core_for_lookup(token: str) -> str:
    return _nfc(_strip_edge_punct(token))


def greedy_longest_phrase_ipa(tokens: list[str], i: int, lex: dict[str, str], max_w: int) -> tuple[str, int]:
    """
    From tokens[i:], consume longest dict key (word-sequence) and return (ipa, n_tokens_consumed).
    """
    best_ipa = ""
    best_span = 0
    for w in range(min(max_w, len(tokens) - i), 0, -1):
        key = " ".join(_token_core_for_lookup(tokens[i + j]) for j in range(w))
        if key in lex:
            best_ipa = lex[key]
            best_span = w
            break
    return best_ipa, best_span


def vietnamese_g2p_line(
    text: str,
    *,
    dict_path: Path | str | None = None,
) -> str:
    path = Path(dict_path) if dict_path else _DEFAULT_VI_DICT
    lex = _load_lexicon(path)
    max_w = _max_key_len(lex) if lex else 1
    raw = _nfc(text.strip())
    if not raw:
        return ""
    tokens = raw.split()
    out: list[str] = []
    pos = 0
    while pos < len(tokens):
        ipa, span = greedy_longest_phrase_ipa(tokens, pos, lex, max_w)
        if span > 0:
            if ipa:
                out.append(ipa)
            pos += span
            continue
        t = _token_core_for_lookup(tokens[pos])
        pos += 1
        if not t:
            continue
        # ASCII letters: keep lowercase for vocoder (foreign names)
        if all(ord(c) < 128 and (c.isalpha() or c in "-'") for c in t) and not any(
            "\u0100" <= c <= "\u1eff" for c in t
        ):
            out.append(t.lower())
            continue
        # Syllable-level: hyphenated compounds (rare in wiki)
        if "-" in t and not t.startswith("-"):
            parts = [p for p in t.split("-") if p]
            sub: list[str] = []
            for p in parts:
                sub.append(vietnamese_word_to_ipa(p, dict_path=path))
            sub = [s for s in sub if s]
            if sub:
                out.append("-".join(sub))
            continue
        wipa = vietnamese_word_to_ipa(t, dict_path=path)
        if wipa:
            out.append(wipa)
    return " ".join(out)


def vietnamese_word_to_ipa(word: str, *, dict_path: Path | str | None = None) -> str:
    """One whitespace-free token (may be one syllable)."""
    path = Path(dict_path) if dict_path else _DEFAULT_VI_DICT
    lex = _load_lexicon(path)
    w = _token_core_for_lookup(word)
    if not w:
        return ""
    if w in lex:
        return lex[w]
    # Per-syllable: Vietnamese wiki tokens are usually single syllables
    return vietnamese_syllable_to_ipa(w)


def dialect_ids() -> list[str]:
    return ["vi", "vi-VN", "vi_vn", "vie", "vietnamese", "Vietnamese"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Vietnamese rule + lexicon G2P (Northern IPA).")
    ap.add_argument("--dict", type=Path, default=None, help=f"default: {_DEFAULT_VI_DICT}")
    ap.add_argument("text", nargs="*", help="Input text (or use --stdin)")
    ap.add_argument("--stdin", action="store_true")
    args = ap.parse_args()
    dict_path = args.dict or _DEFAULT_VI_DICT
    if args.stdin or not args.text:
        import sys

        text = sys.stdin.read()
    else:
        text = " ".join(args.text)
    print(vietnamese_g2p_line(text, dict_path=dict_path))


if __name__ == "__main__":
    main()
