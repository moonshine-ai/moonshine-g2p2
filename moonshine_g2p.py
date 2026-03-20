"""
Running text to IPA via CMUdict word lookups (:class:`CmudictIpa`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cmudict_ipa import CmudictIpa, split_text_to_words

_DEFAULT_DICT_TSV = Path(__file__).resolve().parent / "data" / "en_us" / "dict.tsv"


class MoonshineG2P:
    """
    Split text with :func:`~cmudict_ipa.split_text_to_words`, look up each token
    via :meth:`CmudictIpa.translate_to_ipa` (normalization lives there), and join
    IPA strings with spaces. Unknown tokens are omitted. When CMUdict
    lists several IPA forms for a word, the first entry in the sorted list from
    :meth:`CmudictIpa.translate_to_ipa` is used.
    """

    def __init__(self, cmudict: CmudictIpa) -> None:
        self._cmudict = cmudict

    def text_to_ipa(self, text: str) -> str:
        parts: list[str] = []
        for token in split_text_to_words(text):
            (_, alts), = self._cmudict.translate_to_ipa([token])
            if alts:
                parts.append(alts[0])
        return " ".join(parts)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grapheme-to-IPA for running text using a CMUdict-derived TSV (word, IPA)."
    )
    p.add_argument(
        "-d",
        "--dict",
        type=Path,
        metavar="PATH",
        default=None,
        help=f"dictionary TSV (default: {_DEFAULT_DICT_TSV})",
    )
    p.add_argument(
        "text",
        nargs="*",
        help='words or phrase to translate (joined with spaces; default: "Hello world!")',
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    path = args.dict if args.dict is not None else _DEFAULT_DICT_TSV
    if not path.is_file():
        sys.stderr.write(
            f"error: dictionary file not found: {path}\n"
            "Run: python3 scripts/download_cmudict_to_tsv.py\n"
        )
        sys.exit(1)
    g2p = MoonshineG2P(CmudictIpa(path))
    phrase = "Hello world!" if not args.text else " ".join(args.text)
    print(g2p.text_to_ipa(phrase))


if __name__ == "__main__":
    main()
