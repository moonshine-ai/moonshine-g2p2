#!/usr/bin/env python3
"""
Evaluate heteronym **context heuristics** in ``english_rule_g2p`` / ``english_heteronym_heuristics``.

Uses heteronym training JSON (``homograph_wordid`` = gold IPA for the span).
Reports accuracy for:

* **baseline** — first pronunciation after CMU/homograph merge (same as context-blind
  default, equivalent to :meth:`english_rule_g2p.EnglishLexiconRuleG2p.g2p` for
  in-lexicon keys).
* **heuristic** — :meth:`~english_rule_g2p.EnglishLexiconRuleG2p.g2p_span`.
* **onnx** (optional) — :class:`moonshine_onnx_g2p.OnnxHeteronymG2p` when
  ``--heteronym-onnx`` points to ``model.onnx`` (and merged config beside it).

Example::

    python scripts/eval_english_heteronym_heuristics.py --max-examples 4000 --seed 1
    python scripts/eval_english_heteronym_heuristics.py --max-examples 0 \\
        --heteronym-onnx models/en_us/heteronym/model.onnx
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
from english_rule_g2p import EnglishLexiconRuleG2p


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--train-json",
        type=Path,
        default=_REPO_ROOT / "data" / "en_us" / "heteronym-training" / "homograph_train.json",
    )
    p.add_argument(
        "--dict-tsv",
        type=Path,
        default=_REPO_ROOT / "models" / "en_us" / "dict_filtered_heteronyms.tsv",
    )
    p.add_argument("--max-examples", type=int, default=5000, help="0 = all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--focus-words",
        type=str,
        default="",
        help="comma-separated homograph keys to filter (e.g. live,read,close,use)",
    )
    p.add_argument(
        "--heteronym-onnx",
        type=Path,
        default=None,
        help="path to heteronym model.onnx (default: try models/en_us/heteronym/model.onnx)",
    )
    p.add_argument(
        "--no-onnx",
        action="store_true",
        help="skip heteronym ONNX (faster full-file runs for baseline vs rules only)",
    )
    args = p.parse_args(argv)

    if not args.train_json.is_file():
        raise SystemExit(f"missing {args.train_json}")

    blob = json.loads(args.train_json.read_text(encoding="utf-8"))
    rows = list(blob.values())
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.max_examples and args.max_examples > 0:
        rows = rows[: args.max_examples]

    focus: set[str] | None = None
    if args.focus_words.strip():
        focus = {x.strip().lower() for x in args.focus_words.split(",") if x.strip()}

    g2p = EnglishLexiconRuleG2p(dict_path=args.dict_tsv)

    onnx_path = args.heteronym_onnx
    if onnx_path is None:
        onnx_path = _REPO_ROOT / "models" / "en_us" / "heteronym" / "model.onnx"
    het = None
    if args.no_onnx:
        print("# heteronym ONNX skipped: --no-onnx", file=sys.stderr)
    elif onnx_path.is_file():
        try:
            from moonshine_onnx_g2p import OnnxHeteronymG2p

            het = OnnxHeteronymG2p(onnx_path)
        except Exception as e:
            print(f"# heteronym ONNX skipped: {e}", file=sys.stderr)
    else:
        print(f"# heteronym ONNX skipped: not found {onnx_path}", file=sys.stderr)

    base_ok = heur_ok = onnx_ok = n = 0
    by_word: dict[str, list[tuple[bool, bool, bool | None]]] = {}

    for v in rows:
        hg = (v.get("homograph") or "").strip().lower()
        if not hg:
            continue
        if focus is not None and hg not in focus:
            continue
        gold = (v.get("homograph_wordid") or "").strip()
        if not gold:
            continue
        text = v.get("char") or ""
        try:
            s = int(v["homograph_char_start"])
            e = int(v["homograph_char_end"])
        except (KeyError, TypeError, ValueError):
            continue
        if e > len(text) or s < 0 or s >= e:
            continue
        cands = g2p.pronunciation_candidates(text[s:e])
        if len(cands) < 2:
            continue

        span_surface = text[s:e].strip().lower()
        gk = "".join(c for c in span_surface if c.isalpha())
        if gk != hg:
            continue

        base = cands[0]
        heur = g2p.g2p_span(text, s, e)
        if het is not None:
            lk = normalize_word_for_lookup(text[s:e])
            onnx_ipa = het.disambiguate_ipa(
                text, s, e, lookup_key=lk, cmudict_alternatives=cands
            )
        else:
            onnx_ipa = None
        n += 1
        b_hit = base == gold
        h_hit = heur == gold
        o_hit = onnx_ipa == gold if onnx_ipa is not None else None
        if b_hit:
            base_ok += 1
        if h_hit:
            heur_ok += 1
        if o_hit:
            onnx_ok += 1
        by_word.setdefault(hg, []).append((b_hit, h_hit, o_hit))

    print(f"Examples (multi-pronunciation keys only): {n}")
    if n:
        print(f"  baseline first-candidate exact: {base_ok}/{n} ({100.0 * base_ok / n:.2f}%)")
        print(f"  g2p_span heuristics exact:      {heur_ok}/{n} ({100.0 * heur_ok / n:.2f}%)")
        if het is not None:
            print(f"  heteronym ONNX exact:           {onnx_ok}/{n} ({100.0 * onnx_ok / n:.2f}%)")
        print()
        if het is not None:
            print("Per homograph (baseline% → heuristic% → ONNX%, n):")
        else:
            print("Per homograph (baseline% → heuristic%, n):")
        for w in sorted(by_word.keys(), key=lambda x: (-len(by_word[x]), x)):
            pairs = by_word[w]
            m = len(pairs)
            b = sum(1 for x, _, _ in pairs if x)
            h = sum(1 for _, y, _ in pairs if y)
            if het is not None:
                o_ct = sum(1 for _, _, o in pairs if o)
                print(
                    f"  {w:12s}  {100.0 * b / m:5.1f}% → {100.0 * h / m:5.1f}% → "
                    f"{100.0 * o_ct / m:5.1f}%   (n={m})"
                )
            else:
                print(
                    f"  {w:12s}  {100.0 * b / m:5.1f}% → {100.0 * h / m:5.1f}%   (n={m})"
                )


if __name__ == "__main__":
    main()
