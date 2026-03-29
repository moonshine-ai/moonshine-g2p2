#!/usr/bin/env python3
"""
Evaluate **English G2P**: lexicon+rules (``english_rule_g2p``) vs **ONNX OOV**
(:class:`moonshine_onnx_g2p.OnnxOovG2p`) vs references.

**References**

* **CMUdict** — first stored IPA per key in ``dict_filtered_heteronyms.tsv``.
* **eSpeak NG** — single-token word IPA when ``espeak-phonemizer`` works.

**Metrics** (after greedy segmentation with ``models/en_us/oov/phoneme_vocab.json``):

* **Exact** — predicted phoneme token list equals reference token list.
* **PER** — mean normalized Levenshtein distance on token sequences
  (``edit / max(len(ref), len(hyp))``).

On the CMUdict reference, ``EnglishLexiconRuleG2p`` should match **exactly** for
in-lexicon words (it returns the same TSV string). The interesting baseline is
**ONNX vs CMU**.

**eSpeak block:** For dictionary hits, ``g2p`` still returns **CMU** IPA, not
eSpeak. So the “lexicon+rules” row vs eSpeak mostly measures **lexicon CMU vs
eSpeak** disagreement, not the quality of the OOV rule pass. Use that row to
compare how close ONNX is to eSpeak relative to the static dictionary line.

Example::

    python scripts/eval_english_g2p_metrics.py --max-words 2000 --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from english_rule_g2p import (
    EnglishLexiconRuleG2p,
    espeak_ng_ipa_line,
    load_english_lexicon,
    load_oov_phoneme_vocab_tokens,
    segment_ipa_with_vocab,
)
from heteronym.ipa_postprocess import levenshtein_distance


def _espeak_single_word_ipa(word: str, *, voice: str) -> str | None:
    raw = espeak_ng_ipa_line(word, voice=voice)
    if not raw:
        return None
    parts = [x for x in raw.split() if x]
    return parts[0] if len(parts) == 1 else None


def _per(ref_toks: list[str], hyp_toks: list[str]) -> float:
    la, lb = len(ref_toks), len(hyp_toks)
    if la == 0 and lb == 0:
        return 0.0
    d = levenshtein_distance(ref_toks, hyp_toks)
    return d / max(la, lb)


def _eval_block(
    label: str,
    word_keys: list[str],
    *,
    ref_ipa_fn,
    hyp_rules_ipa_fn,
    hyp_onnx_ipa_fn,
    vocab: frozenset[str],
    include_onnx: bool,
) -> None:
    print(f"== {label} ==")
    n = 0
    ex_r = ex_o = 0
    per_r: list[float] = []
    per_o: list[float] = []
    for wkey in word_keys:
        ref_ipa = ref_ipa_fn(wkey)
        if not ref_ipa:
            continue
        ref_toks = segment_ipa_with_vocab(ref_ipa, vocab)
        if not ref_toks:
            continue
        pr = hyp_rules_ipa_fn(wkey)
        if not pr:
            continue
        tr = segment_ipa_with_vocab(pr, vocab)
        if include_onnx:
            po = hyp_onnx_ipa_fn(wkey)
            if not po:
                continue
            to = segment_ipa_with_vocab(po, vocab)
        n += 1
        if tr == ref_toks:
            ex_r += 1
        per_r.append(_per(ref_toks, tr))
        if include_onnx:
            if to == ref_toks:
                ex_o += 1
            per_o.append(_per(ref_toks, to))
    print(f"  evaluated: {n}")
    if n:
        print(
            f"  lexicon+rules  exact={ex_r}/{n} ({100.0 * ex_r / n:.2f}%)  "
            f"mean_PER={sum(per_r) / len(per_r):.4f}"
        )
        if include_onnx:
            print(
                f"  onnx_oov       exact={ex_o}/{n} ({100.0 * ex_o / n:.2f}%)  "
                f"mean_PER={sum(per_o) / len(per_o):.4f}"
            )
    print()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dict-tsv",
        type=Path,
        default=_REPO_ROOT / "models" / "en_us" / "dict_filtered_heteronyms.tsv",
    )
    p.add_argument("--max-words", type=int, default=3000, help="0 = all keys")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--espeak-voice", default="en-us")
    p.add_argument(
        "--onnx",
        type=Path,
        default=_REPO_ROOT / "models" / "en_us" / "oov" / "model.onnx",
    )
    args = p.parse_args(argv)

    vocab_path = _REPO_ROOT / "models" / "en_us" / "oov" / "phoneme_vocab.json"
    if not vocab_path.is_file():
        raise SystemExit(f"missing {vocab_path}")
    vocab = load_oov_phoneme_vocab_tokens(vocab_path)

    lex = load_english_lexicon(args.dict_tsv)
    g2p = EnglishLexiconRuleG2p(lexicon=lex)

    all_keys = sorted(lex.keys())
    rng = random.Random(args.seed)
    if args.max_words and args.max_words > 0 and len(all_keys) > args.max_words:
        rng.shuffle(all_keys)
        word_keys = sorted(all_keys[: args.max_words])
    else:
        word_keys = all_keys

    onnx = None
    if args.onnx.is_file():
        try:
            from moonshine_onnx_g2p import OnnxOovG2p

            onnx = OnnxOovG2p(args.onnx)
        except Exception as e:
            print(f"# ONNX skipped: {e}", file=sys.stderr)

    def ref_cmu(w: str) -> str | None:
        return lex.get(w)

    def hyp_rules(w: str) -> str | None:
        ipa = g2p.g2p(w)
        return ipa or None

    def hyp_onnx(w: str) -> str | None:
        if onnx is None:
            return None
        toks = onnx.predict_phonemes(w)
        if not toks:
            return None
        return "".join(toks)

    print(f"Word sample: {len(word_keys)} (seed={args.seed})")
    print()
    include_onnx = onnx is not None
    if not include_onnx:
        print("# ONNX not loaded; CMUdict block shows lexicon+rules sanity only.", file=sys.stderr)

    _eval_block(
        "Reference: CMUdict (first IPA)",
        word_keys,
        ref_ipa_fn=ref_cmu,
        hyp_rules_ipa_fn=hyp_rules,
        hyp_onnx_ipa_fn=hyp_onnx,
        vocab=vocab,
        include_onnx=include_onnx,
    )

    if not include_onnx:
        return

    def ref_espeak(w: str) -> str | None:
        return _espeak_single_word_ipa(w, voice=args.espeak_voice)

    _eval_block(
        "Reference: eSpeak NG (single-token words only); "
        "lexicon+rules uses CMU IPA on hits → expect low exact vs eSpeak",
        word_keys,
        ref_ipa_fn=ref_espeak,
        hyp_rules_ipa_fn=hyp_rules,
        hyp_onnx_ipa_fn=hyp_onnx,
        vocab=vocab,
        include_onnx=True,
    )


if __name__ == "__main__":
    main()
