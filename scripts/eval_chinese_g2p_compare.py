#!/usr/bin/env python3
"""
Compare **Chinese ONNX + lexicon G2P** (``chinese_rule_g2p.py``) against other libraries.

Baselines
---------

1. **pypinyin + dragonmapper** (reference IPA): per-character ``lazy_pinyin(..., style=TONE3)``
   is converted with ``dragonmapper.transcriptions.pinyin_to_ipa``. This is a common
   **default-reading** Mandarin pipeline (no heteronym model).

2. **espeak-ng** (optional): Mandarin voice ``cmn``, ``--ipa`` output. Inventory and
   segmentation differ from ipa-dict; we report **syllable-count agreement** only
   (whitespace-separated chunks from phonemizer / espeak).

Metrics (per line, Han characters only — punctuation stripped for alignment)
---------------------------------------------------------------------------

* **syllable_count_ok**: ``len(our_syllables) == len(Han_chars)`` after restricting the
  utterance to **Han characters only** (Latin in wiki lines is dropped for this benchmark).
* **strict_ipa**: after :func:`strip_ipa_tone_marks`, exact string match syllable-wise
  to dragonmapper reference IPA.
* **loose_ipa**: same but :func:`collapse_ipa_allophones` maps ipa-dict / dragonmapper
  variants (e.g. ``ɚ``/``ɨ``, ``tswɔ``/``tsuɔ``) before comparing.

Limitations
-----------

* pypinyin’s default reading is often wrong on polyphones (行, 了, …); our system can
  differ **by design** and score *lower* on strict agreement with this baseline while
  still matching the lexicon.
* espeak uses a different phone set; syllable counts usually align for plain prose.

Optional dependencies::

    pip install pypinyin dragonmapper phonemizer

``espeak-ng`` should be on ``PATH`` for espeak syllable counts (phonemizer uses it).

Example::

    python scripts/eval_chinese_g2p_compare.py --lines 200 --seed 0
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def _is_cjk(ch: str) -> bool:
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


def han_only(s: str) -> str:
    return "".join(c for c in s if _is_cjk(c))


def strip_ipa_tone_marks(s: str) -> str:
    """Remove IPA tone letters / combining tone marks (Chao-style contours)."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    for ch in "\u02E5\u02E6\u02E7\u02E8\u02E9\u02EA\u02EB":
        s = s.replace(ch, "")
    for ch in "˥˦˧˨˩":
        s = s.replace(ch, "")
    return s


def collapse_ipa_allophones(s: str) -> str:
    """Map common ipa-dict vs dragonmapper / allophone pairs for loose comparison."""
    s = strip_ipa_tone_marks(s)
    s = s.replace("ɚ", "ɨ")
    s = s.replace("ɑ", "a")
    s = s.replace("tswɔ", "tsuɔ")
    s = s.replace("twɔ", "tuɔ")
    return s


def try_import_compare_deps() -> tuple[object, object]:
    try:
        from pypinyin import lazy_pinyin, Style
        from dragonmapper.transcriptions import pinyin_to_ipa
    except ImportError as e:
        raise SystemExit(
            "Install comparison deps: pip install pypinyin dragonmapper\n" f"({e})"
        ) from e
    return lazy_pinyin, pinyin_to_ipa


def espeak_syllable_chunks(text: str) -> list[str] | None:
    """Syllable-ish tokens from espeak-ng Mandarin IPA (whitespace split)."""
    han = han_only(text)
    if not han:
        return []
    try:
        r = subprocess.run(
            ["espeak-ng", "-v", "cmn", "--ipa", "-q", han],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    raw = (r.stdout or "").strip()
    if not raw:
        return None
    return raw.split()


def phonemizer_syllable_chunks(text: str) -> list[str] | None:
    try:
        from phonemizer import phonemize
    except ImportError:
        return None
    han = han_only(text)
    if not han:
        return []
    try:
        raw = phonemize(han, language="cmn", backend="espeak", strip=True, preserve_punctuation=False)
    except Exception:
        return None
    raw = raw.strip()
    if not raw:
        return None
    return raw.split()


@dataclass
class LineStats:
    n_han: int
    n_our: int
    count_ok: bool
    strict_matches: int
    loose_matches: int
    n_compare: int


def _safe_pinyin_to_ipa(pinyin_to_ipa: object, tone3: str) -> str | None:
    try:
        return pinyin_to_ipa(tone3)
    except (ValueError, KeyError):
        return None


def eval_line(
    han: str,
    our_syl: list[str],
    lazy_pinyin: object,
    pinyin_to_ipa: object,
) -> LineStats | None:
    from pypinyin import Style

    py_list = lazy_pinyin(list(han), style=Style.TONE3)
    ref_ipa: list[str] = []
    for x in py_list:
        ipa = _safe_pinyin_to_ipa(pinyin_to_ipa, x)
        if ipa is None:
            return None
        ref_ipa.append(ipa)
    n_han = len(han)
    n_our = len(our_syl)
    count_ok = n_our == n_han == len(ref_ipa)

    strict_m = loose_m = 0
    n_cmp = min(len(ref_ipa), len(our_syl))
    for i in range(n_cmp):
        if strip_ipa_tone_marks(ref_ipa[i]) == strip_ipa_tone_marks(our_syl[i]):
            strict_m += 1
        if collapse_ipa_allophones(ref_ipa[i]) == collapse_ipa_allophones(our_syl[i]):
            loose_m += 1

    return LineStats(
        n_han=n_han,
        n_our=n_our,
        count_ok=count_ok,
        strict_matches=strict_m,
        loose_matches=loose_m,
        n_compare=n_cmp,
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--wiki",
        type=Path,
        default=_REPO_ROOT / "data" / "zh_hans" / "wiki-text.txt",
        help="Corpus lines to sample (non-empty).",
    )
    p.add_argument("--lines", type=int, default=200, help="Number of lines to evaluate.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--model-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "zh_hans" / "roberta_chinese_base_upos_onnx",
    )
    args = p.parse_args(argv)

    lazy_pinyin, pinyin_to_ipa = try_import_compare_deps()

    sys.path.insert(0, str(_REPO_ROOT))
    from chinese_rule_g2p import ChineseOnnxLexiconG2p

    if not args.wiki.is_file():
        raise SystemExit(f"Missing wiki file: {args.wiki}")

    pool: list[str] = []
    with args.wiki.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if len(han_only(s)) >= 4:
                pool.append(s)
    if not pool:
        raise SystemExit("No suitable lines in wiki (need ≥4 Han chars).")

    rng = random.Random(args.seed)
    if len(pool) <= args.lines:
        sample = pool
    else:
        sample = rng.sample(pool, args.lines)

    g2p = ChineseOnnxLexiconG2p(model_dir=args.model_dir)

    tot_han = 0
    lines_count_ok = 0
    tot_strict = 0
    tot_loose = 0
    tot_syl_pairs = 0
    tot_our_syllables = 0
    espeak_lines = 0
    espeak_count_ok = 0
    skipped_dragonmapper = 0
    skipped_onnx = 0

    for line in sample:
        han = han_only(line)
        es_ch = phonemizer_syllable_chunks(han)
        if es_ch is None:
            es_ch = espeak_syllable_chunks(han)
        if es_ch is not None:
            espeak_lines += 1
            if len(es_ch) == len(han):
                espeak_count_ok += 1

        try:
            # Han-only utterance so syllable counts align with per-char pypinyin.
            ipa_line = g2p.sentence_to_ipa(han)
        except ValueError as e:
            err = str(e)
            if "sequence length" in err or "span width" in err or "span_inner_pad" in err:
                skipped_onnx += 1
                continue
            raise
        our_syl = [x for x in ipa_line.split() if x]

        st = eval_line(han, our_syl, lazy_pinyin, pinyin_to_ipa)
        if st is None:
            skipped_dragonmapper += 1
            continue
        tot_han += st.n_han
        tot_our_syllables += st.n_our
        if st.count_ok:
            lines_count_ok += 1
        tot_strict += st.strict_matches
        tot_loose += st.loose_matches
        tot_syl_pairs += st.n_compare

    n = len(sample)
    n_used = n - skipped_dragonmapper - skipped_onnx
    print(f"Lines sampled: {n}; used for IPA compare: {n_used} (seed={args.seed}, wiki={args.wiki.name})")
    if skipped_onnx:
        print(f"Skipped {skipped_onnx} line(s): ONNX sequence / span limit (very long input).")
    if skipped_dragonmapper:
        print(
            f"Skipped {skipped_dragonmapper} line(s): dragonmapper cannot convert some pypinyin syllables (e.g. nü)."
        )
    print(f"Han characters total (in compared lines): {tot_han}")
    if tot_han:
        print(
            f"Our syllables / Han chars (mean): {tot_our_syllables / tot_han:.3f} "
            "(<1.0 gaps in lexicon; >1.0 extra splits / erhua-style mismatch)"
        )
    if n_used:
        print(
            f"Syllable count match (our vs Han count): {lines_count_ok}/{n_used} lines "
            f"({100.0 * lines_count_ok / n_used:.1f}%)"
        )
    if tot_syl_pairs:
        print(
            f"vs pypinyin→dragonmapper IPA — tone-stripped strict syllable match: "
            f"{tot_strict}/{tot_syl_pairs} ({100.0 * tot_strict / tot_syl_pairs:.1f}%)"
        )
        print(
            f"vs pypinyin→dragonmapper IPA — loose allophone syllable match: "
            f"{tot_loose}/{tot_syl_pairs} ({100.0 * tot_loose / tot_syl_pairs:.1f}%)"
        )
    if espeak_lines > 0:
        print(
            f"espeak-ng syllable count vs Han (phonemizer or CLI): "
            f"{espeak_count_ok}/{espeak_lines} lines ({100.0 * espeak_count_ok / espeak_lines:.1f}%)"
        )
    else:
        print("espeak-ng: not available (install espeak-ng and optionally phonemizer).")


if __name__ == "__main__":
    main()
