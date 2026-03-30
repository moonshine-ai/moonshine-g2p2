# Korean — `ko`

## Contents

- **`dict.tsv`** — Hangul word → IPA (broad Seoul-style inventory after normalization), used by C++ `KoreanRuleG2p` (lexicon + internal Hangul rule pipeline).

## Provenance

| Asset | Source |
|--------|--------|
| `dict.tsv` | [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict) — `data/ko.txt` (MIT). Fetched via `scripts/download_multilingual_ipa_lexicons.py`. |

**Note:** A separate Korean **morph + UPOS** ONNX export exists for tagging tests (`data/ko/roberta_korean_morph_upos_onnx/`, HF [KoichiYasuoka/roberta-base-korean-morph-upos](https://huggingface.co/KoichiYasuoka/roberta-base-korean-morph-upos)), produced by `scripts/export_korean_ud_onnx.py`. It is **not** wired into the main Korean G2P path in C++ or Python `korean_rule_g2p`—only this lexicon is required for pronunciation in the shipped pipeline.

## Recreating

```bash
python scripts/download_multilingual_ipa_lexicons.py --only ko
```

Writes `data/ko/dict.tsv` and `data/ko/source.txt`. Copy `dict.tsv` into `cpp/data/ko/` if you maintain this tree separately.

### Optional: Korean morph ONNX (for `KoreanTokPosOnnx` / tests only)

```bash
python scripts/export_korean_ud_onnx.py
```

Default: `data/ko/roberta_korean_morph_upos_onnx/`.
