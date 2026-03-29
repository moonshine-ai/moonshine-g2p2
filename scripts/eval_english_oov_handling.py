#!/usr/bin/env python3
"""
Compare **English OOV** predictors against **CMUdict-style** gold (same metric family
as :mod:`scripts.eval_english_g2p_metrics`).

The bundled ``data/en_us/oov-training/oov_{train,valid}.json`` are **random train/valid
splits of dictionary words** for training the OOV model. Those words usually **still
appear** in ``dict_filtered_heteronyms.tsv``, so this script uses **simulated OOV**:

* **Gold** — first IPA string for each sampled key from the production TSV (same
  inventory as :class:`english_rule_g2p.EnglishLexiconRuleG2p`).
* **Hand rules** — :func:`english_rule_g2p.english_oov_rules_ipa` (ignores the
  lexicon on purpose). This is the **non-neural fallback** when a word is missing
  from the lexicon.
* **ONNX OOV** — :class:`moonshine_onnx_g2p.OnnxOovG2p` phoneme sequence.
* **eSpeak NG** — single-token IPA from ``espeak-phonemizer`` when available
  (calibration: how close eSpeak is to CMU gold on the same sample).

Tokenization uses greedy :func:`english_rule_g2p.segment_ipa_with_vocab` with
``models/en_us/oov/phoneme_vocab.json``. Metrics: **exact** (full token-list match)
and **mean PER** (normalized token edit distance).

Example::

    python scripts/eval_english_oov_handling.py --max-words 5000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cmudict_ipa import normalize_word_for_lookup
from english_rule_g2p import (
    EnglishLexiconRuleG2p,
    _normalize_grapheme_key,
    espeak_ng_ipa_line,
    english_oov_rules_ipa,
    load_oov_phoneme_vocab_tokens,
    segment_ipa_with_vocab,
)
from heteronym.ipa_postprocess import levenshtein_distance


def _per(ref_toks: list[str], hyp_toks: list[str]) -> float:
    la, lb = len(ref_toks), len(hyp_toks)
    if la == 0 and lb == 0:
        return 0.0
    d = levenshtein_distance(ref_toks, hyp_toks)
    return d / max(la, lb)


def _espeak_single_word_ipa(word: str, *, voice: str) -> str | None:
    raw = espeak_ng_ipa_line(word, voice=voice)
    if not raw:
        return None
    parts = [x for x in raw.split() if x]
    return parts[0] if len(parts) == 1 else None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dict-tsv",
        type=Path,
        default=_REPO_ROOT / "models" / "en_us" / "dict_filtered_heteronyms.tsv",
    )
    p.add_argument("--max-words", type=int, default=5000, help="0 = all keys")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--espeak-voice", default="en-us")
    p.add_argument(
        "--onnx",
        type=Path,
        default=_REPO_ROOT / "models" / "en_us" / "oov" / "model.onnx",
    )
    p.add_argument(
        "--skip-espeak",
        action="store_true",
        help="omit eSpeak block (much faster; default runs eSpeak when available)",
    )
    args = p.parse_args(argv)

    vocab_path = _REPO_ROOT / "models" / "en_us" / "oov" / "phoneme_vocab.json"
    if not vocab_path.is_file():
        raise SystemExit(f"missing {vocab_path}")
    vocab = load_oov_phoneme_vocab_tokens(vocab_path)

    g2p = EnglishLexiconRuleG2p(dict_path=args.dict_tsv)
    all_keys = sorted(g2p._lex.keys())
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
            print(f"# ONNX OOV skipped: {e}", file=sys.stderr)
    else:
        print(f"# ONNX OOV skipped: not found {args.onnx}", file=sys.stderr)

    n = ex_lex = ex_rules = ex_onnx = ex_es = 0
    per_lex: list[float] = []
    per_rules: list[float] = []
    per_onnx: list[float] = []
    per_es: list[float] = []

    for wkey in word_keys:
        gold_ipa = g2p._lex.get(wkey)
        if not gold_ipa:
            continue
        ref_toks = segment_ipa_with_vocab(gold_ipa, vocab)
        if not ref_toks:
            continue

        lex_ipa = gold_ipa
        tl = segment_ipa_with_vocab(lex_ipa, vocab)
        n += 1
        if tl == ref_toks:
            ex_lex += 1
        per_lex.append(_per(ref_toks, tl))

        rules_ipa = english_oov_rules_ipa(wkey)
        tr = segment_ipa_with_vocab(rules_ipa, vocab)
        if tr == ref_toks:
            ex_rules += 1
        per_rules.append(_per(ref_toks, tr))

        if onnx is not None:
            po = "".join(onnx.predict_phonemes(wkey))
            to = segment_ipa_with_vocab(po, vocab)
            if to == ref_toks:
                ex_onnx += 1
            per_onnx.append(_per(ref_toks, to))

        es_ipa = (
            None
            if args.skip_espeak
            else _espeak_single_word_ipa(wkey, voice=args.espeak_voice)
        )
        if es_ipa:
            te = segment_ipa_with_vocab(es_ipa, vocab)
            if te == ref_toks:
                ex_es += 1
            per_es.append(_per(ref_toks, te))

    print("Simulated OOV (hypotheses ignore lexicon except lex-oracle row)")
    print(f"  sample: {len(word_keys)} keys, evaluated with segmentable gold: {n}")
    print(f"  seed={args.seed}")
    print()
    if n:
        print(
            f"  lexicon oracle (sanity)     exact={ex_lex}/{n} ({100.0 * ex_lex / n:.2f}%)  "
            f"mean_PER={sum(per_lex) / len(per_lex):.4f}"
        )
        print(
            f"  hand OOV rules (baseline)   exact={ex_rules}/{n} ({100.0 * ex_rules / n:.2f}%)  "
            f"mean_PER={sum(per_rules) / len(per_rules):.4f}"
        )
        if onnx is not None and per_onnx:
            print(
                f"  ONNX OOV                    exact={ex_onnx}/{n} ({100.0 * ex_onnx / n:.2f}%)  "
                f"mean_PER={sum(per_onnx) / len(per_onnx):.4f}"
            )
        if per_es:
            print(
                f"  eSpeak NG vs CMU gold       exact={ex_es}/{len(per_es)} "
                f"({100.0 * ex_es / len(per_es):.2f}% of {len(per_es)} espeak_ok)  "
                f"mean_PER={sum(per_es) / len(per_es):.4f}"
            )
        else:
            print("  eSpeak NG: no usable single-token lines (install espeak / espeak-phonemizer?)")

    # Optional: oov_valid.json coverage vs production lexicon
    valid_path = _REPO_ROOT / "data" / "en_us" / "oov-training" / "oov_valid.json"
    if valid_path.is_file():
        blob = json.loads(valid_path.read_text(encoding="utf-8"))
        lex_k = g2p._lex
        miss = 0
        tot = 0
        for v in blob.values():
            w = (v.get("char") or "").strip()
            if not w:
                continue
            gk = _normalize_grapheme_key(normalize_word_for_lookup(w))
            tot += 1
            if gk not in lex_k:
                miss += 1
        print()
        print(
            f"Note: oov_valid.json rows checked vs production lexicon: "
            f"{miss}/{tot} keys absent (usually 0 — split is for training, not missing words)."
        )


if __name__ == "__main__":
    main()
