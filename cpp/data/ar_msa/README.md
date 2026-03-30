# Arabic (MSA) — `ar_msa`

## Contents

- **`dict.tsv`** — optional lexicon: undiacritized Arabic key → IPA (tab-separated).
- **`arabertv02_tashkeel_fadel_onnx/`** — BERT token-classification ONNX for full vocalization (diacritics), used before IPA rules in C++ `ArabicRuleG2p`.

## Provenance

| Asset | Source |
|--------|--------|
| ONNX bundle | [AbderrahmanSkiredj1/arabertv02_tashkeel_fadel](https://huggingface.co/AbderrahmanSkiredj1/arabertv02_tashkeel_fadel) on Hugging Face (token classification for tashkīl). |
| `vocab.txt`, tokenizer files | Written by the export script from the same HF checkpoint (`save_pretrained` + custom `vocab.txt` line order). |
| `dict.tsv` (if built in-repo) | Pipeline in `scripts/build_ar_msa_lexicon_from_camel_tools.py` using [CAMeL Tools](https://camel-tools.readthedocs.io/) + repo `arabic_ipa` helpers on wiki-like text. |

## Recreating

### ONNX directory (`arabertv02_tashkeel_fadel_onnx/`)

1. Install build deps: `pip install torch transformers onnx onnxruntime` (versions compatible with your PyTorch).
2. From repo root:

   ```bash
   python scripts/export_arabic_msa_diacritizer_onnx.py \
     --out-dir data/ar_msa/arabertv02_tashkeel_fadel_onnx
   ```

3. C++ expects at least: `model.onnx`, `meta.json`, `vocab.txt`, `tokenizer_config.json` (plus any files the tokenizer saved alongside them).

**Compatibility:** Recent `torch` + `transformers` combinations can fail inside BERT’s attention-mask handling when tracing (observed: `ValueError: Wrong shape for input_ids … or attention_mask` with torch 2.10). The checked-in ONNX was exported with an older stack; if the script fails, use a venv with pinned versions from the time of export, or adjust `scripts/export_arabic_msa_diacritizer_onnx.py` for the current `transformers` API.

### Lexicon (`dict.tsv`)

1. Install Camel Tools and MSA data (see `scripts/build_ar_msa_lexicon_from_camel_tools.py` docstring, e.g. `camel_data -i disambig-mle-calima-msa-r13`).
2. Run:

   ```bash
   python scripts/build_ar_msa_lexicon_from_camel_tools.py \
     --wiki data/ar/wiki-text.txt \
     --out data/ar_msa/dict.tsv
   ```

3. Copy or symlink into `cpp/data/ar_msa/` if you maintain this tree separately.
