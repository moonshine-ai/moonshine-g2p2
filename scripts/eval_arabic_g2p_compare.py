#!/usr/bin/env python3
"""
Compare ONNX-based :mod:`arabic_rule_g2p` to Camel Tools reference
(``scripts/arabic_g2p_ref_camel_tools.py``) on sample lines or a text file.

Reports line-level equality rate and a simple **phoneme-token** overlap (Jaccard on IPA tokens).

Requires for reference path: ``pip install camel-tools``. If import fails, only the ONNX path runs.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from arabic_rule_g2p import ArabicRuleG2p  # noqa: E402


def _tok_set(ipa: str) -> set[str]:
    return set(re.findall(r"[^\s.]+", ipa))


def jaccard(a: str, b: str) -> float:
    sa, sb = _tok_set(a), _tok_set(b)
    if not sa and not sb:
        return 1.0
    u = sa | sb
    if not u:
        return 0.0
    return len(sa & sb) / len(u)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines-file", type=Path, default=None)
    ap.add_argument("--max-lines", type=int, default=50)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--dict", type=Path, default=None)
    args = ap.parse_args()

    ref_line_to_ipa = None
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "arabic_g2p_ref_camel_tools", _REPO / "scripts" / "arabic_g2p_ref_camel_tools.py"
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        ref_line_to_ipa = mod.ref_line_to_ipa
    except Exception:
        pass

    g = ArabicRuleG2p(model_dir=args.model_dir, dict_path=args.dict)

    lines: list[str] = []
    if args.lines_file and args.lines_file.is_file():
        with args.lines_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n\r")
                if line.strip():
                    lines.append(line)
                if len(lines) >= args.max_lines:
                    break
    else:
        lines = [
            "القاهرة عاصمة مصر.",
            "الكتاب على المكتب.",
            "الشمس مشرقة.",
        ]

    if ref_line_to_ipa is None:
        print("Reference module not loaded; install camel-tools. Printing ONNX path only.", file=sys.stderr)

    j_scores: list[float] = []
    exact = 0
    n = 0
    for line in lines:
        n += 1
        onnx_ipa = g.text_to_ipa(line)
        if ref_line_to_ipa is None:
            print(f"IN:  {line[:120]}")
            print(f"ONX: {onnx_ipa[:200]}")
            continue
        try:
            ref_ipa = ref_line_to_ipa(line)
        except Exception as e:
            print(f"ref failed line {n}: {e}", file=sys.stderr)
            continue
        j = jaccard(onnx_ipa, ref_ipa)
        j_scores.append(j)
        if onnx_ipa == ref_ipa:
            exact += 1
        if n <= 8:
            print(f"--- line {n} ---")
            print("IN: ", line[:160])
            print("REF:", ref_ipa[:220])
            print("ONX:", onnx_ipa[:220])
            print(f"Jaccard IPA tokens: {j:.3f}")

    if j_scores:
        avg = sum(j_scores) / len(j_scores)
        print(f"Summary: {n} lines, exact match {exact}/{len(j_scores)}, mean Jaccard {avg:.3f}")


if __name__ == "__main__":
    main()
