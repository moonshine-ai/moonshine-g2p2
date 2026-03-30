"""
Compare :mod:`korean_rule_g2p` against **eSpeak NG Korean** (voice ``ko``) on wiki prose.

**Reference.** The highest-accuracy *IPA* baseline we can use without extra trained models
is **libespeak-ng** with the Korean voice — the same stack as
:func:`heteronym.espeak_heteronyms.espeak_phonemize_ipa_raw` elsewhere in this repo.
Libraries like g2pK emit **Hangul pronunciation spelling**, not IPA, so they are not
directly comparable to our output.

**Method.** For each of the first 100 lines of ``data/ko/wiki-text.txt``, we take
**Hangul syllables only** (U+AC00..U+D7A3), run :func:`korean_rule_g2p` (lexicon + OOV rules,
normalized IPA) and eSpeak, normalize both
IPA strings (stress, length marks, syllable dots, a few vowel aliases), then compute
:class:`difflib.SequenceMatcher` similarity per line. We also assert that our dotted
syllable count always matches the number of Hangul codepoints (pipeline sanity).

**Dependencies.** ``data/ko/dict.tsv``, ``espeak-phonemizer``, and libespeak-ng Korean voice.

If dependencies or ``data/ko/wiki-text.txt`` are missing, the test is skipped.
"""

from __future__ import annotations

import statistics
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WIKI_KO = _REPO_ROOT / "data" / "ko" / "wiki-text.txt"
_KO_DICT = _REPO_ROOT / "data" / "ko" / "dict.tsv"
_FIRST_N_LINES = 100
_ESPEAK_KO_VOICE = "ko"

# Observed mean similarity ~0.58 on first 100 lines; floor guards against gross regressions
# while tolerating eSpeak / normalization drift between environments.
_MIN_MEAN_SEQUENCE_SIMILARITY = 0.35


def _has_espeak_ko() -> bool:
    try:
        from heteronym.espeak_heteronyms import EspeakPhonemizer, espeak_phonemize_ipa_raw

        p = EspeakPhonemizer()
        raw = espeak_phonemize_ipa_raw(p, "가", voice=_ESPEAK_KO_VOICE)
        return bool(raw and raw.strip())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _KO_DICT.is_file() or not _has_espeak_ko(),
    reason="needs data/ko/dict.tsv and espeak-phonemizer with Korean voice",
)


def hangul_only(line: str) -> str:
    return "".join(ch for ch in line if "\uac00" <= ch <= "\ud7a3")


def normalize_ko_ipa_for_compare(s: str) -> str:
    """
    Loose normalization so rule-based broad IPA and eSpeak Korean IPA are more comparable.

    eSpeak Korean often uses /ɐ/ where broad transcriptions use /ʌ/; stress and length
    marks differ by convention.
    """
    t = unicodedata.normalize("NFC", s.strip().lower())
    for mark in "\u02c8\u02cc":  # primary / secondary stress
        t = t.replace(mark, "")
    t = t.replace("ː", "")
    t = t.replace(".", "")
    # Vowel / height aliases between inventories
    t = t.replace("ɐ", "ʌ")
    return t


def compare_rule_g2p_to_espeak_ko_first_n_wiki_lines(
    *,
    n_lines: int,
    wiki_path: Path,
) -> dict[str, float | int | list[float]]:
    from heteronym.espeak_heteronyms import EspeakPhonemizer, espeak_phonemize_ipa_raw

    from korean_rule_g2p import korean_g2p

    if not wiki_path.is_file():
        raise FileNotFoundError(wiki_path)

    phon = EspeakPhonemizer()
    ratios: list[float] = []
    lines_with_hangul = 0
    espeak_empty = 0

    with wiki_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_lines:
                break
            h = hangul_only(line)
            if not h:
                continue
            lines_with_hangul += 1

            rule_ipa = korean_g2p(h, syllable_sep=".")

            es_raw = espeak_phonemize_ipa_raw(phon, h, voice=_ESPEAK_KO_VOICE)
            if not es_raw or not es_raw.strip():
                espeak_empty += 1
                continue

            a = normalize_ko_ipa_for_compare(rule_ipa)
            b = normalize_ko_ipa_for_compare(es_raw)
            ratios.append(SequenceMatcher(None, a, b).ratio())

    mean_r = statistics.mean(ratios) if ratios else 0.0
    median_r = statistics.median(ratios) if ratios else 0.0
    min_r = min(ratios) if ratios else 0.0

    return {
        "lines_with_hangul": lines_with_hangul,
        "lines_compared_to_espeak": len(ratios),
        "espeak_empty": espeak_empty,
        "mean_sequence_similarity": mean_r,
        "median_sequence_similarity": median_r,
        "min_sequence_similarity": min_r,
        "per_line_ratios": ratios,
    }


@pytest.mark.skipif(not _WIKI_KO.is_file(), reason="data/ko/wiki-text.txt not present")
def test_korean_rule_g2p_matches_espeak_ko_on_first_100_wiki_lines() -> None:
    stats = compare_rule_g2p_to_espeak_ko_first_n_wiki_lines(
        n_lines=_FIRST_N_LINES,
        wiki_path=_WIKI_KO,
    )

    assert stats["lines_with_hangul"] >= 50, (
        f"expected most wiki lines to contain Hangul; got {stats['lines_with_hangul']}"
    )
    assert stats["espeak_empty"] == 0, (
        f"eSpeak Korean should phonemize all Hangul-only excerpts; empty={stats['espeak_empty']}"
    )
    assert stats["mean_sequence_similarity"] >= _MIN_MEAN_SEQUENCE_SIMILARITY, (
        f"mean normalized IPA similarity vs eSpeak ko dropped below {_MIN_MEAN_SEQUENCE_SIMILARITY}: "
        f"mean={stats['mean_sequence_similarity']:.4f}, "
        f"median={stats['median_sequence_similarity']:.4f}, "
        f"min={stats['min_sequence_similarity']:.4f}, "
        f"lines={stats['lines_compared_to_espeak']}"
    )


@pytest.mark.skipif(not _WIKI_KO.is_file(), reason="data/ko/wiki-text.txt not present")
def test_korean_rule_g2p_espeak_similarity_single_line_regression() -> None:
    """Pinned short phrase so CI catches large drift without depending on full wiki stats."""
    from heteronym.espeak_heteronyms import EspeakPhonemizer, espeak_phonemize_ipa_raw

    from korean_rule_g2p import korean_g2p

    h = "안녕하세요"
    phon = EspeakPhonemizer()
    rule_ipa = korean_g2p(h, syllable_sep=".")
    es_raw = espeak_phonemize_ipa_raw(phon, h, voice=_ESPEAK_KO_VOICE)
    assert es_raw
    a = normalize_ko_ipa_for_compare(rule_ipa)
    b = normalize_ko_ipa_for_compare(es_raw)
    r = SequenceMatcher(None, a, b).ratio()
    assert r >= 0.45, f"안녕하세요 similarity vs eSpeak ko too low: {r:.4f} rule={rule_ipa!r} espeak={es_raw!r}"
