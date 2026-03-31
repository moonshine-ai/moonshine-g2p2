"""
First-N-lines wiki smoke: run Moonshine G2P and ensure IPA fits Piper ``phoneme_id_map`` keys.

Pipeline: NFC → :func:`piper_ipa_normalization.normalize_g2p_ipa_for_piper` →
:func:`piper_ipa_normalization.coerce_unknown_chars_to_inventory`.

Environment:
  WIKI_G2P_FIRST_N   — lines per language (default 12)
  WIKI_G2P_MAX_CHARS — clip each line before G2P (default 280)
  WIKI_G2P_PIPER_STRICT — if ``1``, skip closest-scalar coercion (only shared + per-lang replacements)
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

import speak
from piper_ipa_normalization import (
    coerce_unknown_chars_to_inventory,
    default_piper_onnx_json_path,
    ipa_codepoints_not_in_map,
    load_piper_phoneme_id_map_keys,
    normalize_g2p_ipa_for_piper,
)

_REPO = Path(__file__).resolve().parent.parent

_FIRST_N = int(os.environ.get("WIKI_G2P_FIRST_N", "12"))
_MAX_CHARS = int(os.environ.get("WIKI_G2P_MAX_CHARS", "280"))
_STRICT = os.environ.get("WIKI_G2P_PIPER_STRICT", "").strip() == "1"

# (speak.py --lang key for Piper bundle, wiki folder under data/, optional reason to skip)
_CASES: list[tuple[str, str | None, str | None]] = [
    ("en_us", None, None),
    ("en_gb", "en_us", None),
    ("de", None, None),
    ("fr", None, None),
    ("hi", None, None),
    ("it", None, None),
    ("nl", None, None),
    ("ru", None, None),
    ("tr", None, None),
    ("uk", None, None),
    ("vi", None, None),
    ("es_mx", None, None),
    ("es_es", None, None),
    ("pt_br", None, None),
    ("pt_pt", None, None),
    ("ar", "ar", None),
]

_AR_ONNX = _REPO / "data" / "ar_msa" / "arabertv02_tashkeel_fadel_onnx" / "model.onnx"
_AR_DICT = _REPO / "data" / "ar_msa" / "dict.tsv"

_g2p_singletons: dict[str, Callable[[str], str]] = {}


def _wiki_path(wiki_subdir: str) -> Path:
    return _REPO / "data" / wiki_subdir / "wiki-text.txt"


def _wiki_subdir_for_case(speak_lang: str, override: str | None) -> str:
    if override is not None:
        return override
    return speak._PIPER_LANG[speak_lang][0]


def _g2p_for(speak_lang: str) -> Callable[[str], str]:
    if speak_lang in _g2p_singletons:
        return _g2p_singletons[speak_lang]

    if speak_lang in ("en_us", "en_gb"):
        fn: Callable[[str], str] = lambda t, _s=speak: _s._english_text_to_ipa(t)  # noqa: E731
    elif speak_lang == "de":
        from german_rule_g2p import text_to_ipa as de_tti

        fn = lambda t, f=de_tti: f(t)  # noqa: E731
    elif speak_lang == "fr":
        from french_g2p import text_to_ipa as fr_tti

        fn = lambda t, f=fr_tti: f(t)  # noqa: E731
    elif speak_lang == "hi":
        from hindi_rule_g2p import text_to_ipa as hi_tti

        fn = lambda t, f=hi_tti: f(t)  # noqa: E731
    elif speak_lang == "it":
        from italian_rule_g2p import text_to_ipa as it_tti

        fn = lambda t, f=it_tti: f(t)  # noqa: E731
    elif speak_lang == "nl":
        from dutch_rule_g2p import text_to_ipa as nl_tti

        fn = lambda t, f=nl_tti: f(t)  # noqa: E731
    elif speak_lang == "ru":
        from russian_rule_g2p import text_to_ipa as ru_tti

        fn = lambda t, f=ru_tti: f(t)  # noqa: E731
    elif speak_lang == "tr":
        from turkish_rule_g2p import text_to_ipa as tr_tti

        fn = lambda t, f=tr_tti: f(t)  # noqa: E731
    elif speak_lang == "uk":
        from ukrainian_rule_g2p import text_to_ipa as uk_tti

        fn = lambda t, f=uk_tti: f(t)  # noqa: E731
    elif speak_lang == "vi":
        from vietnamese_rule_g2p import vietnamese_g2p_line as vi_line

        fn = lambda t, f=vi_line: f(t)  # noqa: E731
    elif speak_lang == "es_mx":
        from spanish_rule_g2p import mexican_spanish_dialect, text_to_ipa as es_tti

        d = mexican_spanish_dialect()
        fn = lambda t, f=es_tti, dialect=d: f(t, dialect=dialect)  # noqa: E731
    elif speak_lang == "es_es":
        from spanish_rule_g2p import castilian_spanish_dialect, text_to_ipa as es_tti

        d = castilian_spanish_dialect()
        fn = lambda t, f=es_tti, dialect=d: f(t, dialect=dialect)  # noqa: E731
    elif speak_lang == "pt_br":
        from portuguese_rule_g2p import text_to_ipa as pt_tti

        fn = lambda t, f=pt_tti: f(t, variant="pt_br")  # noqa: E731
    elif speak_lang == "pt_pt":
        from portuguese_rule_g2p import text_to_ipa as pt_tti

        fn = lambda t, f=pt_tti: f(t, variant="pt_pt")  # noqa: E731
    elif speak_lang == "ar":
        from arabic_rule_g2p import ArabicRuleG2p

        if not _AR_ONNX.is_file():
            pytest.skip("Arabic ONNX bundle not present (data/ar_msa/...)")
        g = ArabicRuleG2p(model_dir=_AR_ONNX.parent, dict_path=_AR_DICT)
        fn = lambda t, gg=g: gg.text_to_ipa(t)  # noqa: E731
    else:
        raise AssertionError(f"unsupported speak_lang {speak_lang!r}")

    _g2p_singletons[speak_lang] = fn
    return fn


def _clip_line(line: str) -> str:
    line = line.rstrip("\n\r")
    if len(line) > _MAX_CHARS:
        line = line[:_MAX_CHARS]
    return line


@pytest.mark.parametrize(
    "speak_lang,wiki_override,skip_reason",
    _CASES,
    ids=[f"{a}-{b or 'wiki=default'}" for a, b, _ in _CASES],
)
def test_wiki_lines_g2p_fit_piper_phoneme_map(
    speak_lang: str,
    wiki_override: str | None,
    skip_reason: str | None,
) -> None:
    if skip_reason:
        pytest.skip(skip_reason)
    wiki_sub = _wiki_subdir_for_case(speak_lang, wiki_override)
    wpath = _wiki_path(wiki_sub)
    if not wpath.is_file():
        pytest.skip(f"missing {wpath}")

    data_subdir, onnx_base = speak._PIPER_LANG[speak_lang]
    json_path = default_piper_onnx_json_path(
        repo_root=_REPO,
        piper_data_subdir=data_subdir,
        default_onnx_basename=onnx_base,
    )
    if not json_path.is_file():
        pytest.skip(f"missing Piper config {json_path}")

    keys = load_piper_phoneme_id_map_keys(json_path)
    g2p = _g2p_for(speak_lang)

    bad_samples: list[str] = []
    with open(wpath, encoding="utf-8", errors="replace") as f:
        for i, raw in enumerate(f):
            if i >= _FIRST_N:
                break
            text = _clip_line(raw)
            if not text.strip():
                continue
            ipa_raw = g2p(text)
            ipa_norm = normalize_g2p_ipa_for_piper(ipa_raw, piper_lang_key=speak_lang)
            ipa_final = (
                ipa_norm
                if _STRICT
                else coerce_unknown_chars_to_inventory(
                    ipa_norm, phoneme_keys=keys, use_closest_scalar=True
                )
            )

            missing = ipa_codepoints_not_in_map(ipa_final, phoneme_keys=keys)
            if missing:
                bad_samples.append(
                    f"line[{i}] missing={missing!r} after_norm={ipa_norm!r} final={ipa_final!r} text={text[:80]!r}"
                )

    assert not bad_samples, "Unknown phoneme codepoints after pipeline:\n" + "\n".join(bad_samples[:20])
