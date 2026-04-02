"""
Map Moonshine G2P IPA to symbols that appear in Piper ``phoneme_id_map`` JSON (eSpeak-ng style).

Used after NFC and before building Piper phoneme id sequences. C++ mirrors this module in
``moonshine_g2p::utf8_nfc_copy``, ``normalize_g2p_ipa_for_piper``, ``coerce_unknown_ipa_chars_to_piper_inventory``,
and ``ipa_to_piper_ready`` (``moonshine-tts/include/moonshine-g2p/ipa-postprocess.h``); Piper TTS applies the same pipeline
in ``moonshine-tts/piper-tts.cpp``. Korean (``ko``) substring rules and post-pass live in ``moonshine-tts/src/ipa-postprocess.cpp`` —
keep them in sync with ``LANG_SPECIFIC_G2P_TO_PIPER_REPLACEMENTS["ko"]`` and ``_korean_post_normalize_ipa``.

Rules are:
1. Shared replacements (all languages), applied in order.
2. Per-``piper_lang_key`` replacements (e.g. ``en_us``, ``de``), applied after shared.
3. Optional ``coerce_unknown_chars_to_inventory``: last-resort single-codepoint substitution
   by nearest Unicode scalar among inventory letters/modifiers (IPA-like subset only).
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Replacement tables (longest / most specific first within each list)
# ---------------------------------------------------------------------------

# Applied to every language before ``LANG_SPECIFIC``.
SHARED_G2P_TO_PIPER_REPLACEMENTS: list[tuple[str, str]] = [
    # R-colored vowel: CMU / English G2P uses ɝ; eSpeak US Piper bundles use ɜ + length.
    ("\u025d", "\u025c\u02d0"),  # ɝ → ɜː
]

# Moonshine Korean rule G2P vs eSpeak-ng ``ko`` (Zeroth Piper training target). Longest-first.
_KO_G2P_TO_ESPEAK_LIKE: list[tuple[str, str]] = [
    ("kamsʰahamnida", "ɡˈɐmsɐhˌɐpnidˌɐ"),
    ("hasʰejo", "hˌɐsejˌo"),
    ("tɛhanminkuk̚", "dɛhˈɐnminqˌuq"),
    ("anɲjʌŋ", "ˈɐnnjʌŋ"),
    ("sʰejo", "sˌejo"),
    ("sʰe", "sˌe"),
    ("sʰ", "s"),
]

# Keys are ``speak._PIPER_LANG`` style tags (e.g. ``en_us``, ``de``, ``ar_msa``).
LANG_SPECIFIC_G2P_TO_PIPER_REPLACEMENTS: dict[str, list[tuple[str, str]]] = {
    # Populated from ``tests/test_wiki_g2p_piper_phoneme_coverage.py`` failures; extend as needed.
    "en_us": [],
    "en_gb": [],
    "ko": _KO_G2P_TO_ESPEAK_LIKE,
}


def load_piper_phoneme_id_map_keys(onnx_json_path: Path) -> frozenset[str]:
    """Return the set of keys from ``phoneme_id_map`` (one Piper voice JSON)."""
    import json

    with open(onnx_json_path, encoding="utf-8") as f:
        cfg: dict[str, Any] = json.load(f)
    m = cfg.get("phoneme_id_map")
    if not isinstance(m, dict):
        raise ValueError(f"Missing phoneme_id_map in {onnx_json_path}")
    return frozenset(m.keys())


def default_piper_onnx_json_path(*, repo_root: Path, piper_data_subdir: str, default_onnx_basename: str) -> Path:
    """``data/<subdir>/piper-voices/<stem>.onnx.json`` (*default_onnx_basename* may already end in ``.onnx``)."""
    stem = default_onnx_basename.removesuffix(".onnx")
    return repo_root / "data" / piper_data_subdir / "piper-voices" / f"{stem}.onnx.json"


def _korean_post_normalize_ipa(s: str) -> str:
    """eSpeak-style primary stress on word-initial ``jʌ`` (e.g. 여보세요)."""
    if s.startswith("jʌ"):
        s = "jˈʌ" + s[2:]
    return s.replace(" jʌ", " jˈʌ")


def normalize_g2p_ipa_for_piper(ipa: str, *, piper_lang_key: str) -> str:
    """Apply shared + language-specific substring replacements (NFC input recommended)."""
    s = unicodedata.normalize("NFC", ipa)
    for old, new in SHARED_G2P_TO_PIPER_REPLACEMENTS:
        s = s.replace(old, new)
    lang = piper_lang_key
    if lang in ("ko_kr", "korean"):
        lang = "ko"
    for old, new in LANG_SPECIFIC_G2P_TO_PIPER_REPLACEMENTS.get(lang, []):
        s = s.replace(old, new)
    if lang == "ko":
        s = _korean_post_normalize_ipa(s)
    return s


def ipa_to_piper_ready(
    ipa: str,
    *,
    piper_lang_key: str,
    phoneme_id_map_keys: frozenset[str],
    apply_coercion: bool = True,
) -> str:
    """
    Full pipeline: NFC → explicit replacements → optional closest-inventory coercion.

    *phoneme_id_map_keys* is typically from :func:`load_piper_phoneme_id_map_keys`.
    """
    s = normalize_g2p_ipa_for_piper(ipa, piper_lang_key=piper_lang_key)
    if apply_coercion:
        s = coerce_unknown_chars_to_inventory(s, phoneme_keys=phoneme_id_map_keys)
    return s


def _is_ipa_like_inventory_char(ch: str) -> bool:
    """Restrict last-resort ``closest codepoint`` pool to phonetic symbols, not digits/punct."""
    if len(ch) != 1:
        return False
    o = ord(ch)
    cat = unicodedata.category(ch)
    if cat.startswith("L"):
        return True
    # IPA Extensions, Phonetic Extensions, Modifier letters, combining marks we keep in pool
    if 0x0250 <= o <= 0x02FF:
        return True
    if 0x0300 <= o <= 0x036F:
        return True
    if 0x1D00 <= o <= 0x1D7F:
        return True
    return False


def _single_char_inventory(keys: Iterable[str]) -> frozenset[str]:
    return frozenset(k for k in keys if len(k) == 1)


def _substitution_pool(singles: frozenset[str]) -> frozenset[str]:
    """Inventory codepoints we allow as targets for closest-scalar fallback (IPA + letters)."""
    pool = frozenset(c for c in singles if _is_ipa_like_inventory_char(c))
    return pool if pool else singles


def coerce_unknown_chars_to_inventory(
    ipa: str,
    *,
    phoneme_keys: frozenset[str],
    use_closest_scalar: bool = True,
) -> str:
    """
    For each NFC codepoint not in ``phoneme_keys`` and not whitespace, either drop it or
    replace with the closest inventory character (Unicode scalar distance) from the substitution
    pool (IPA extensions, modifier letters, and other letter-category symbols in the map).

    Combining marks (Mn/Me) not in the map are dropped. Unknown punctuation and symbols (P*, S*)
    are dropped instead of being mapped to arbitrary phonemes.
    """
    singles = _single_char_inventory(phoneme_keys)
    pool = _substitution_pool(singles)

    out: list[str] = []
    for ch in unicodedata.normalize("NFC", ipa):
        if ch.isspace():
            out.append(ch)
            continue
        if ch in phoneme_keys:
            out.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat in {"Mn", "Me"}:
            continue
        if cat[0] in {"P", "S"}:
            continue
        if use_closest_scalar and pool:
            o = ord(ch)
            # Deterministic tie-break (same as C++): smallest codepoint when distances tie.
            out.append(min(pool, key=lambda a: (abs(ord(a) - o), ord(a))))
    return "".join(out)


def ipa_codepoints_not_in_map(ipa: str, *, phoneme_keys: frozenset[str]) -> list[str]:
    """Unique NFC codepoints (excluding whitespace) missing from ``phoneme_keys``."""
    nfc = unicodedata.normalize("NFC", ipa)
    bad: list[str] = []
    seen: set[str] = set()
    for ch in nfc:
        if ch.isspace() or ch in phoneme_keys:
            continue
        if ch not in seen:
            seen.add(ch)
            bad.append(ch)
    return bad


def merge_lang_specific_from_heuristic(
    lang_key: str,
    missing_to_replacement: Mapping[str, str],
) -> None:
    """Append pairs to ``LANG_SPECIFIC_G2P_TO_PIPER_REPLACEMENTS[lang_key]`` (for generated tables)."""
    cur = list(LANG_SPECIFIC_G2P_TO_PIPER_REPLACEMENTS.get(lang_key, []))
    for old, new in missing_to_replacement.items():
        if (old, new) not in cur:
            cur.append((old, new))
    LANG_SPECIFIC_G2P_TO_PIPER_REPLACEMENTS[lang_key] = cur
