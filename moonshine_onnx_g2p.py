"""
Running text to IPA via a CMUdict TSV, with heteronym and OOV paths executed
through ONNX Runtime (no PyTorch / no repo training modules).

Dependencies::

    pip install onnxruntime numpy

Optional reference line (same as ``moonshine_g2p.py``)::

    pip install espeak-phonemizer
    # plus system libespeak-ng

Default artifacts: ``model.onnx`` plus ``config_onnx.json`` (merged vocabs,
``train_config``, heteronym ``homograph_index`` or OOV ``oov_index``, and ONNX
I/O metadata) in each ``models/en_us/{heteronym,oov}/`` directory. If
``config_onnx.json`` is missing, the script falls back to the older layout
(separate ``char_vocab.json``, ``phoneme_vocab.json``, etc.).

**ONNX models:** build with ``python scripts/export_models_to_onnx.py``. The exported
graph fixes the **decoder sequence length** at ``max_phoneme_len`` (legacy PyTorch
ONNX export); this script pads each greedy step to that length and reads logits at
the current prefix index, matching ``moonshine_g2p.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import IO, Any, Iterable, List, Tuple

import numpy as np

try:
    import onnxruntime as ort
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "onnxruntime is required. Install with: pip install onnxruntime"
    ) from e

# --- constants (mirrors training code) -------------------------------------------

SPECIAL_PAD = "<pad>"
SPECIAL_UNK = "<unk>"
SPECIAL_PHON_PAD = "<pad>"
SPECIAL_PHON_UNK = "<unk>"
SPECIAL_PHON_BOS = "<bos>"
SPECIAL_PHON_EOS = "<eos>"

HETERONYM_CONTEXT_MAX_CHARS = 32
_ESPEAK_VOICE = "en-us"

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_DICT_TSV = _REPO_ROOT / "models" / "en_us" / "dict_filtered_heteronyms.tsv"
_DEFAULT_OOV_ONNX = _REPO_ROOT / "models" / "en_us" / "oov" / "model.onnx"
_DEFAULT_HETERONYM_ONNX = _REPO_ROOT / "models" / "en_us" / "heteronym" / "model.onnx"

CONFIG_ONNX_SCHEMA_VERSION = 1
_CONFIG_ONNX_NAME = "config_onnx.json"
# Some trees ship the export bundle as ``onnx-config.json`` (same schema as ``config_onnx.json``).
_LEGACY_MERGED_CONFIG_NAME = "onnx-config.json"


def _load_merged_onnx_config(parent: Path, *, expect_kind: str) -> dict[str, Any]:
    """Load merged ONNX JSON from *parent*; prefer ``config_onnx.json``, then ``onnx-config.json``."""
    for name in (_CONFIG_ONNX_NAME, _LEGACY_MERGED_CONFIG_NAME):
        p = parent / name
        if not p.is_file():
            continue
        cfg = json.loads(p.read_text(encoding="utf-8"))
        _validate_merged_onnx_config(cfg, expect_kind=expect_kind, path=p)
        return cfg
    raise FileNotFoundError(
        f"no merged ONNX config in {parent} (tried {_CONFIG_ONNX_NAME!r}, "
        f"{_LEGACY_MERGED_CONFIG_NAME!r})"
    )


def _validate_merged_onnx_config(cfg: dict[str, Any], *, expect_kind: str, path: Path) -> None:
    try:
        ver_i = int(cfg.get("config_schema_version", -1))
    except (TypeError, ValueError):
        ver_i = -1
    if ver_i != CONFIG_ONNX_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: expected config_schema_version {CONFIG_ONNX_SCHEMA_VERSION}, got "
            f"{cfg.get('config_schema_version')!r}"
        )
    if cfg.get("model_kind") != expect_kind:
        raise ValueError(
            f"{path}: expected model_kind {expect_kind!r}, got {cfg.get('model_kind')!r}"
        )


# --- TSV CMUdict (from cmudict_ipa.CmudictIpa TSV path, inlined) -----------------


def _normalize_grapheme_key(word_token: str) -> str:
    """Lowercase and strip CMU-style alternate suffix ``word(2)`` -> ``word``."""
    s = word_token.lower()
    if s.endswith(")") and "(" in s:
        i = s.rfind("(")
        mid = s[i + 1 : -1]
        if mid.isdigit():
            s = s[:i]
    return s


def split_text_to_words(text: str) -> List[str]:
    return text.split()


def normalize_word_for_lookup(token: str) -> str:
    s = token.lower().strip()
    if not s:
        return ""
    i, j = 0, len(s)
    while i < j and not s[i].isalnum():
        i += 1
    while i < j and not s[j - 1].isalnum():
        j -= 1
    return s[i:j]


class CmudictIpaTsv:
    """``word_lower`` -> sorted unique IPA strings (one TSV pronunciation per line)."""

    def __init__(self, path: str | Path) -> None:
        self._ipa_by_word: dict[str, List[str]] = {}
        with open(path, encoding="utf-8", errors="replace") as f:
            self._load_tsv_lines(f)

    def _load_tsv_lines(self, f: IO[str]) -> None:
        raw: dict[str, set[str]] = {}
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" not in line:
                continue
            word_token, ipa = line.split("\t", 1)
            word_token = word_token.strip()
            ipa = ipa.strip()
            if not word_token or not ipa:
                continue
            key = _normalize_grapheme_key(word_token)
            raw.setdefault(key, set()).add(ipa)
        self._ipa_by_word = {k: sorted(v) for k, v in raw.items()}

    def translate_to_ipa(self, words: Iterable[str]) -> List[Tuple[str, List[str]]]:
        out: List[Tuple[str, List[str]]] = []
        for w in words:
            key = normalize_word_for_lookup(w)
            if not key:
                out.append((w, []))
                continue
            alts = self._ipa_by_word.get(key)
            if alts is None:
                out.append((w, []))
            else:
                out.append((w, list(alts)))
        return out


# --- heteronym context window (from g2p_common.context_window) -------------------


def heteronym_centered_context_window(
    text: str,
    span_s: int,
    span_e: int,
    *,
    max_chars: int = HETERONYM_CONTEXT_MAX_CHARS,
) -> tuple[str, int, int] | None:
    L = len(text)
    s, e = span_s, span_e
    if e > L or s < 0 or s >= e or max_chars < 1:
        return None
    span_w = e - s
    if span_w > max_chars:
        return None

    if L > max_chars:
        w0_lo = max(0, e - max_chars)
        w0_hi = min(s, L - max_chars)
        if w0_lo > w0_hi:
            return None
        ideal = (s + e) / 2.0 - max_chars / 2.0
        w0 = int(round(ideal))
        w0 = max(w0_lo, min(w0, w0_hi))
        text = text[w0 : w0 + max_chars]
        s -= w0
        e -= w0
        L = len(text)

    if L < max_chars:
        total_pad = max_chars - L
        center = (s + e) / 2.0
        left_pad = int(round(max_chars / 2.0 - center))
        left_pad = max(0, min(left_pad, total_pad))
        right_pad = total_pad - left_pad
        text = " " * left_pad + text + " " * right_pad
        s += left_pad
        e += left_pad

    return text, s, e


# --- IPA postprocess (from heteronym.ipa_postprocess) ----------------------------


def ipa_string_to_phoneme_tokens(s: str) -> list[str]:
    t = unicodedata.normalize("NFC", (s or "").strip())
    if not t:
        return []
    if " " in t:
        return [p for p in t.split() if p]
    return list(t)


def levenshtein_distance(a: list[str], b: list[str]) -> int:
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i]
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[lb]


def pick_closest_alternative_index(
    predicted_phoneme_tokens: list[str],
    ipa_alternatives: list[str],
    *,
    n_valid: int,
    extra_phonemes: int,
) -> int:
    n = min(n_valid, len(ipa_alternatives))
    if n <= 0:
        return 0
    best_i, best_d = 0, 10**9
    for i in range(n):
        cand = ipa_string_to_phoneme_tokens(ipa_alternatives[i])
        lim = len(cand) + max(0, int(extra_phonemes))
        prefix = predicted_phoneme_tokens[:lim]
        d = levenshtein_distance(cand, prefix)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def pick_closest_cmudict_ipa(
    predicted_phoneme_tokens: list[str],
    cmudict_alternatives: list[str],
    *,
    extra_phonemes: int,
) -> str:
    if not cmudict_alternatives:
        return ""
    if len(cmudict_alternatives) == 1:
        return cmudict_alternatives[0]
    i = pick_closest_alternative_index(
        predicted_phoneme_tokens,
        cmudict_alternatives,
        n_valid=len(cmudict_alternatives),
        extra_phonemes=extra_phonemes,
    )
    return cmudict_alternatives[i]


def _normalize_ipa_compare(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()


def match_prediction_to_cmudict_ipa(predicted: str, alts: list[str]) -> str | None:
    e0 = _normalize_ipa_compare(predicted)
    for alt in alts:
        if _normalize_ipa_compare(alt) == e0:
            return alt
    e1 = e0.replace("ː", "")
    for alt in alts:
        if _normalize_ipa_compare(alt).replace("ː", "") == e1:
            return alt
    return None


# --- vocab helpers ----------------------------------------------------------------


def _stoi_to_itos(stoi: dict[str, int]) -> list[str]:
    n = max(stoi.values(), default=-1) + 1
    itos = [""] * n
    for s, i in stoi.items():
        if 0 <= i < n:
            itos[i] = s
    return itos


def encode_chars(text: str, char_stoi: dict[str, int]) -> list[int]:
    unk = char_stoi[SPECIAL_UNK]
    return [char_stoi.get(ch, unk) for ch in text]


def _decoder_io_padded(
    cur: list[int],
    *,
    max_phoneme_len: int,
    pad_token_id: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    ONNX export of ``nn.Transformer`` uses traced shapes; decoder self-attention
    reshapes can fail when *decoder_seq_len* changes between greedy steps. Pad
    the decoder sequence to *max_phoneme_len* and mask the tail so ORT matches
    PyTorch variable-length runs (logits are taken at index ``len(cur) - 1``).
    """
    L = len(cur)
    if L > max_phoneme_len:
        raise ValueError(f"decoder length {L} > max_phoneme_len {max_phoneme_len}")
    row = cur + [pad_token_id] * (max_phoneme_len - L)
    mask = [1] * L + [0] * (max_phoneme_len - L)
    return np.array([row], dtype=np.int64), np.array([mask], dtype=np.int64), L


# --- ONNX Runtime runners ----------------------------------------------------------


def _session_providers(*, use_cuda: bool) -> list[str]:
    avail = ort.get_available_providers()
    if use_cuda and "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class OnnxOovG2p:
    def __init__(self, onnx_path: Path, *, use_cuda: bool = False) -> None:
        self._path = Path(onnx_path)
        if not self._path.is_file():
            raise FileNotFoundError(f"OOV ONNX model not found: {self._path}")
        parent = self._path.parent
        merged_path = parent / _CONFIG_ONNX_NAME
        if merged_path.is_file():
            cfg = json.loads(merged_path.read_text(encoding="utf-8"))
            _validate_merged_onnx_config(cfg, expect_kind="oov", path=merged_path)
            self._char_stoi = cfg["char_vocab"]
            self._phon_stoi = cfg["phoneme_vocab"]
            train_cfg = cfg["train_config"]
            oov_meta = cfg["oov_index"]
        else:
            with (parent / "char_vocab.json").open(encoding="utf-8") as f:
                self._char_stoi = json.load(f)
            with (parent / "phoneme_vocab.json").open(encoding="utf-8") as f:
                self._phon_stoi = json.load(f)
            with (parent / "oov_index.json").open(encoding="utf-8") as f:
                oov_meta = json.load(f)
            with (parent / "train_config.json").open(encoding="utf-8") as f:
                train_cfg = json.load(f)

        mpl = oov_meta.get("max_phoneme_len")
        if mpl is None:
            raise KeyError("oov_index must contain max_phoneme_len")

        self._phon_itos = _stoi_to_itos(self._phon_stoi)
        self._max_seq_len = int(train_cfg["max_seq_len"])
        self._max_phoneme_len = int(mpl)
        self._pad_id = self._char_stoi[SPECIAL_PAD]
        self._bos = self._phon_stoi[SPECIAL_PHON_BOS]
        self._eos = self._phon_stoi[SPECIAL_PHON_EOS]
        self._phon_pad = self._phon_stoi[SPECIAL_PHON_PAD]

        so = ort.SessionOptions()
        self._sess = ort.InferenceSession(
            str(self._path),
            sess_options=so,
            providers=_session_providers(use_cuda=use_cuda),
        )

    def predict_phonemes(self, word: str) -> list[str]:
        if not word:
            return []
        ids = encode_chars(word, self._char_stoi)
        if len(ids) > self._max_seq_len:
            ids = ids[: self._max_seq_len]
        pad = self._max_seq_len - len(ids)
        enc_ids = np.array([ids + [self._pad_id] * pad], dtype=np.int64)
        enc_mask = np.array([[1] * len(ids) + [0] * pad], dtype=np.int64)

        cur = [self._bos]
        for _ in range(self._max_phoneme_len):
            dec, dec_mask, L = _decoder_io_padded(
                cur,
                max_phoneme_len=self._max_phoneme_len,
                pad_token_id=self._phon_pad,
            )
            logits = self._sess.run(
                None,
                {
                    "encoder_input_ids": enc_ids,
                    "encoder_attention_mask": enc_mask,
                    "decoder_input_ids": dec,
                    "decoder_attention_mask": dec_mask,
                },
            )[0]
            nxt = int(np.argmax(logits[0, L - 1]))
            if nxt == self._eos or nxt == self._phon_pad:
                break
            cur.append(nxt)
            if len(cur) >= self._max_phoneme_len:
                break

        out: list[str] = []
        for tid in cur[1:]:
            if tid == self._eos:
                break
            if 0 <= tid < len(self._phon_itos):
                tok = self._phon_itos[tid]
                if tok in (SPECIAL_PHON_PAD, SPECIAL_PHON_BOS, SPECIAL_PHON_EOS):
                    continue
                out.append(tok)
        return out


class OnnxHeteronymG2p:
    def __init__(self, onnx_path: Path, *, use_cuda: bool = False) -> None:
        self._path = Path(onnx_path)
        if not self._path.is_file():
            raise FileNotFoundError(f"heteronym ONNX model not found: {self._path}")
        parent = self._path.parent
        cfg = _load_merged_onnx_config(parent, expect_kind="heteronym")
        self._char_stoi = cfg["char_vocab"]
        self._phon_stoi = cfg["phoneme_vocab"]
        homograph = cfg["homograph_index"]
        train_cfg = cfg["train_config"]

        self._ordered: dict[str, Any] = homograph["ordered_candidates"]
        self._max_candidates = int(homograph["max_candidates"])
        self._group_key = str(homograph["group_key"])
        if int(train_cfg["max_candidates"]) != self._max_candidates:
            raise ValueError(
                "train_config.json max_candidates does not match homograph_index.json"
            )
        if str(train_cfg["group_key"]) != self._group_key:
            raise ValueError("train_config.json group_key does not match homograph_index.json")

        self._phon_itos = _stoi_to_itos(self._phon_stoi)
        self._max_seq_len = int(train_cfg["max_seq_len"])
        self._max_phoneme_len = int(
            train_cfg.get("max_phoneme_len", train_cfg.get("max_ipa_len", 64))
        )
        self._lev_extra = int(train_cfg.get("levenshtein_extra_phonemes", 4))
        self._pad_id = self._char_stoi[SPECIAL_PAD]
        self._bos = self._phon_stoi[SPECIAL_PHON_BOS]
        self._eos = self._phon_stoi[SPECIAL_PHON_EOS]
        self._phon_pad = self._phon_stoi[SPECIAL_PHON_PAD]

        so = ort.SessionOptions()
        self._sess = ort.InferenceSession(
            str(self._path),
            sess_options=so,
            providers=_session_providers(use_cuda=use_cuda),
        )

    def disambiguate_ipa(
        self,
        full_text: str,
        span_s: int,
        span_e: int,
        *,
        lookup_key: str,
        cmudict_alternatives: list[str],
    ) -> str:
        if len(cmudict_alternatives) <= 1:
            return cmudict_alternatives[0] if cmudict_alternatives else ""

        gkey = lookup_key if self._group_key == "lower" else full_text[span_s:span_e]
        if gkey not in self._ordered:
            return cmudict_alternatives[0]

        win = heteronym_centered_context_window(full_text, span_s, span_e)
        if win is None:
            return cmudict_alternatives[0]
        window_text, ws, we = win

        ids = encode_chars(window_text, self._char_stoi)
        span = [0.0] * len(ids)
        for j in range(ws, min(we, len(span))):
            span[j] = 1.0

        while len(ids) < self._max_seq_len:
            ids.append(self._pad_id)
            span.append(0.0)
        ids = ids[: self._max_seq_len]
        span = span[: self._max_seq_len]
        attn_1d = [1 if i != self._pad_id else 0 for i in ids]

        if sum(span) < 1.0:
            return cmudict_alternatives[0]

        enc_ids = np.array([ids], dtype=np.int64)
        enc_mask = np.array([attn_1d], dtype=np.int64)
        span_np = np.array([span], dtype=np.float32)

        cur = [self._bos]
        for _ in range(self._max_phoneme_len):
            dec, dec_mask, L = _decoder_io_padded(
                cur,
                max_phoneme_len=self._max_phoneme_len,
                pad_token_id=self._phon_pad,
            )
            logits = self._sess.run(
                None,
                {
                    "encoder_input_ids": enc_ids,
                    "encoder_attention_mask": enc_mask,
                    "span_mask": span_np,
                    "decoder_input_ids": dec,
                    "decoder_attention_mask": dec_mask,
                },
            )[0]
            nxt = int(np.argmax(logits[0, L - 1]))
            if nxt == self._eos or nxt == self._phon_pad:
                break
            cur.append(nxt)
            if len(cur) >= self._max_phoneme_len:
                break

        pred_tokens: list[str] = []
        for tid in cur[1:]:
            if tid == self._eos:
                break
            if 0 <= tid < len(self._phon_itos):
                tok = self._phon_itos[tid]
                if tok in (SPECIAL_PHON_PAD, SPECIAL_PHON_BOS, SPECIAL_PHON_EOS):
                    continue
                pred_tokens.append(tok)

        raw = pick_closest_cmudict_ipa(
            pred_tokens,
            cmudict_alternatives,
            extra_phonemes=self._lev_extra,
        )
        matched = match_prediction_to_cmudict_ipa(raw, cmudict_alternatives)
        return matched if matched is not None else cmudict_alternatives[0]


# --- eSpeak reference (minimal, from heteronym.espeak_heteronyms settings) -------


def _espeak_ng_ipa_line(text: str, *, voice: str = _ESPEAK_VOICE) -> str | None:
    try:
        from espeak_phonemizer import Phonemizer as EspeakPhonemizer
    except ImportError:
        return None
    t = text.strip()
    if not t:
        return None
    phoneme_separator = ""
    word_separator = " "
    try:
        phon = EspeakPhonemizer(default_voice=voice)
        raw = phon.phonemize(
            t,
            voice=voice,
            phoneme_separator=phoneme_separator,
            word_separator=word_separator,
        )
    except (AssertionError, OSError, RuntimeError):
        return None
    return raw.strip() if raw else None


# --- façade -----------------------------------------------------------------------


def _resolve_optional_onnx(explicit: Path | str | None, default_path: Path) -> Path | None:
    if explicit is not None:
        return Path(explicit)
    return default_path if default_path.is_file() else None


class MoonshineOnnxG2P:
    """
    Same pipeline as :class:`MoonshineG2P` in ``moonshine_g2p.py``, but     neural
    steps call ONNX Runtime sessions for ``model.onnx``; configuration is read from
    ``config_onnx.json`` beside it when present (see export script).
    """

    def __init__(
        self,
        cmudict: CmudictIpaTsv,
        *,
        heteronym_onnx: Path | str | None = None,
        oov_onnx: Path | str | None = None,
        use_cuda: bool = False,
    ) -> None:
        self._cmudict = cmudict
        self._heteronym: OnnxHeteronymG2p | None = None
        het = _resolve_optional_onnx(heteronym_onnx, _DEFAULT_HETERONYM_ONNX)
        if het is not None:
            self._heteronym = OnnxHeteronymG2p(het, use_cuda=use_cuda)
        self._oov: OnnxOovG2p | None = None
        oov = _resolve_optional_onnx(oov_onnx, _DEFAULT_OOV_ONNX)
        if oov is not None:
            self._oov = OnnxOovG2p(oov, use_cuda=use_cuda)

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
                if self._oov is not None:
                    phones = self._oov.predict_phonemes(key)
                    if phones:
                        parts.append("".join(phones))
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
        description="Grapheme-to-IPA using CMUdict TSV + optional ONNX heteronym/OOV models."
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
        help='words or phrase (default: "Hello world!")',
    )
    p.add_argument(
        "--heteronym-onnx",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "path to heteronym model.onnx (sibling config_onnx.json from export, or "
            "legacy separate JSON vocabs/index); default if file exists: "
            f"{_DEFAULT_HETERONYM_ONNX}"
        ),
    )
    p.add_argument(
        "--oov-onnx",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "path to OOV model.onnx (sibling config_onnx.json or legacy JSON files); "
            f"default if file exists: {_DEFAULT_OOV_ONNX}"
        ),
    )
    p.add_argument(
        "--cuda",
        action="store_true",
        help="use CUDAExecutionProvider when available",
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
    if args.heteronym_onnx is not None and not args.heteronym_onnx.is_file():
        sys.stderr.write(f"error: heteronym ONNX not found: {args.heteronym_onnx}\n")
        sys.exit(1)
    if args.oov_onnx is not None and not args.oov_onnx.is_file():
        sys.stderr.write(f"error: OOV ONNX not found: {args.oov_onnx}\n")
        sys.exit(1)

    g2p = MoonshineOnnxG2P(
        CmudictIpaTsv(path),
        heteronym_onnx=args.heteronym_onnx,
        oov_onnx=args.oov_onnx,
        use_cuda=args.cuda,
    )
    phrase = "Hello world!" if not args.text else " ".join(args.text)
    print(g2p.text_to_ipa(phrase))
    if not args.no_espeak:
        espeak_line = _espeak_ng_ipa_line(phrase)
        if espeak_line is not None:
            print(f"{espeak_line} (espeak-ng)")


if __name__ == "__main__":
    main()
