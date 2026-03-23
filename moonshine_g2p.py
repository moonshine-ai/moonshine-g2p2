"""
Running text to IPA via CMUdict word lookups (:class:`CmudictIpa`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from cmudict_ipa import CmudictIpa, normalize_word_for_lookup, split_text_to_words

if TYPE_CHECKING:
    from heteronym.infer import HeteronymDisambiguator

_DEFAULT_DICT_TSV = Path(__file__).resolve().parent / "data" / "en_us" / "dict_filtered_heteronyms.txt"
_ESPEAK_VOICE = "en-us"


def _espeak_ng_ipa_line(text: str) -> str | None:
    """
    IPA string from libespeak-ng via ``espeak_phonemizer`` (same family as ``espeak-ng --ipa``).
    Uses the same separator settings as ``heteronym.espeak_heteronyms.espeak_phonemize_ipa_raw``.
    Returns None if the optional dependency or engine is unavailable, or phonemization fails.
    """
    try:
        from heteronym.espeak_heteronyms import EspeakPhonemizer, espeak_phonemize_ipa_raw
    except ImportError:
        return None
    t = text.strip()
    if not t:
        return None
    try:
        phon = EspeakPhonemizer(default_voice=_ESPEAK_VOICE)
        raw = espeak_phonemize_ipa_raw(phon, t, voice=_ESPEAK_VOICE)
    except (AssertionError, OSError, RuntimeError):
        return None
    return raw or None


class MoonshineG2P:
    """
    Split text with :func:`~cmudict_ipa.split_text_to_words`, look up each token
    via :meth:`CmudictIpa.translate_to_ipa` (normalization lives there), and join
    IPA strings with spaces. Unknown tokens are omitted. When CMUdict lists
    several IPA forms for a word, a trained heteronym model (optional checkpoint)
    chooses among them using greedy phoneme decoding and Levenshtein matching on a
    short centered context window; without a checkpoint, the first sorted alternative is used.
    """

    def __init__(
        self,
        cmudict: CmudictIpa,
        *,
        heteronym_checkpoint: Path | str | None = None,
        heteronym_device: str | None = None,
    ) -> None:
        self._cmudict = cmudict
        self._heteronym: HeteronymDisambiguator | None = None
        if heteronym_checkpoint is not None:
            from heteronym.infer import HeteronymDisambiguator as _HD

            self._heteronym = _HD(heteronym_checkpoint, device=heteronym_device)

    def text_to_ipa(self, text: str) -> str:
        parts: list[str] = []
        pos = 0
        for token in split_text_to_words(text):
            idx = text.find(token, pos)
            if idx < 0:
                idx = text.find(token)
            start, end = idx, idx + len(token)
            pos = end

            key = normalize_word_for_lookup(token)
            if not key:
                continue
            (_, alts), = self._cmudict.translate_to_ipa([token])
            if not alts:
                continue
            if len(alts) == 1:
                parts.append(alts[0])
            elif self._heteronym is not None:
                parts.append(
                    self._heteronym.disambiguate_ipa(
                        text,
                        start,
                        end,
                        lookup_key=key,
                        cmudict_alternatives=alts,
                    )
                )
            else:
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
    p.add_argument(
        "--heteronym-checkpoint",
        type=Path,
        default=None,
        metavar="PATH",
        help="trained heteronym checkpoint.pt (char_vocab.json, phoneme_vocab.json, homograph_index.json in same dir)",
    )
    p.add_argument(
        "--heteronym-device",
        type=str,
        default=None,
        metavar="DEVICE",
        help="torch device for heteronym model (default: cuda if available else cpu)",
    )
    p.add_argument(
        "--no-espeak",
        action="store_true",
        help="do not print a second IPA line from eSpeak NG",
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
    ckpt = args.heteronym_checkpoint
    if ckpt is not None and not ckpt.is_file():
        sys.stderr.write(f"error: heteronym checkpoint not found: {ckpt}\n")
        sys.exit(1)
    g2p = MoonshineG2P(
        CmudictIpa(path),
        heteronym_checkpoint=ckpt,
        heteronym_device=args.heteronym_device,
    )
    phrase = "Hello world!" if not args.text else " ".join(args.text)
    print(g2p.text_to_ipa(phrase))
    if not args.no_espeak:
        espeak_line = _espeak_ng_ipa_line(phrase)
        if espeak_line is not None:
            print(f"{espeak_line} (espeak-ng)")


if __name__ == "__main__":
    main()
