# Moonshine G2P

Converts text (graphemes) to the International Phonetic Alphabet (phonemes) for multiple languages.

Avoids using rules, instead relies on training small transformer models on large datasets to handle the conversion. The exact model architectures and stages do vary depending on the language's characteristics.

The C++ TTS/G2P tree lives in the [`moonshine-tts`](https://github.com/moonshine-ai/moonshine-tts) git submodule. After cloning, run `git submodule update --init --recursive` (or clone with `git clone --recurse-submodules …`) so `moonshine-tts/` is populated.

```bash
python scripts/export_wikitext_lines.py --only es_mx --out data/es_mx/wiki-text.txt

python scripts/build_oov_espeak_dataset.py --language es_mx

python train_oov.py --language es_mx
```

```
cd moonshine-tts
mkdir -p build
cd build
cmake ..
cmake --build .
./moonshine_g2p --model-root ../models --language en_us "Live, laugh, love"
./moonshine_g2p --model-root ../models --language ja "東京に行きます。"
```

## C++ bundled data (`moonshine-tts/data/`)

The [`moonshine-tts/data/`](moonshine-tts/data/README.md) tree (submodule [moonshine-ai/moonshine-tts](https://github.com/moonshine-ai/moonshine-tts)) holds lexicons, ONNX exports, and Kokoro TTS assets consumed by the C++ targets (`moonshine_g2p`, `moonshine_tts`) when using `builtin_cpp_data_root()`. Each subfolder has its own **README** with provenance, licenses, and rebuild commands. You can follow the links to the original projects to verify that all of these are available under permissive licenses (like MIT, Apache v2, etc) and can be used commercially. Please [contact me](mailto:pete@moonshine.ai) if you spot a mistake. 

| Directory | Main upstream sources | README |
|-----------|----------------------|--------|
| [`kokoro/`](moonshine-tts/data/kokoro/README.md) | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Hugging Face); ONNX export via the [`kokoro`](https://pypi.org/project/kokoro/) Python package | [→](moonshine-tts/data/kokoro/README.md) |
| [`ar_msa/`](moonshine-tts/data/ar_msa/README.md) | [AbderrahmanSkiredj1/arabertv02_tashkeel_fadel](https://huggingface.co/AbderrahmanSkiredj1/arabertv02_tashkeel_fadel); optional lexicon via [CAMeL Tools](https://camel-tools.readthedocs.io/) | [→](moonshine-tts/data/ar_msa/README.md) |
| [`de/`](moonshine-tts/data/de/README.md) | [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict) (`de.txt`, MIT) | [→](moonshine-tts/data/de/README.md) |
| [`en_us/`](moonshine-tts/data/en_us/README.md) | [CMU Pronouncing Dictionary](https://github.com/cmusphinx/cmudict); heteronym / OOV ONNX models **trained in this repo** | [→](moonshine-tts/data/en_us/README.md) |
| [`fr/`](moonshine-tts/data/fr/README.md) | [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict) (`fr_FR.txt`); liaison POS CSVs (see also [hbenbel/French-Dictionary](https://github.com/hbenbel/French-Dictionary)) | [→](moonshine-tts/data/fr/README.md) |
| [`hi/`](moonshine-tts/data/hi/README.md) | [English Wiktionary](https://en.wiktionary.org/) (Hindi entries, CC BY-SA); [hermitdave/FrequencyWords](https://github.com/hermitdave/FrequencyWords) | [→](moonshine-tts/data/hi/README.md) |
| [`it/`](moonshine-tts/data/it/README.md) | [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict) | [→](moonshine-tts/data/it/README.md) |
| [`ja/`](moonshine-tts/data/ja/README.md) | [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict); [KoichiYasuoka/roberta-small-japanese-char-luw-upos](https://huggingface.co/KoichiYasuoka/roberta-small-japanese-char-luw-upos) | [→](moonshine-tts/data/ja/README.md) |
| [`ko/`](moonshine-tts/data/ko/README.md) | [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict); Piper training audio in the style of [MeloTTS](https://github.com/myshell-ai/MeloTTS); optional morph ONNX: [KoichiYasuoka/roberta-base-korean-morph-upos](https://huggingface.co/KoichiYasuoka/roberta-base-korean-morph-upos) | [→](moonshine-tts/data/ko/README.md) |
| [`nl/`](moonshine-tts/data/nl/README.md) | [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict) | [→](moonshine-tts/data/nl/README.md) |
| [`pt_br/`](moonshine-tts/data/pt_br/README.md), [`pt_pt/`](moonshine-tts/data/pt_pt/README.md) | [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict) | [→ pt_br](moonshine-tts/data/pt_br/README.md) · [→ pt_pt](moonshine-tts/data/pt_pt/README.md) |
| [`ru/`](moonshine-tts/data/ru/README.md) | [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict) | [→](moonshine-tts/data/ru/README.md) |
| [`vi/`](moonshine-tts/data/vi/README.md) | [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict) | [→](moonshine-tts/data/vi/README.md) |
| [`zh_hans/`](moonshine-tts/data/zh_hans/README.md) | [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict); [KoichiYasuoka/chinese-roberta-base-upos](https://huggingface.co/KoichiYasuoka/chinese-roberta-base-upos) | [→](moonshine-tts/data/zh_hans/README.md) |

## Acknowledgements

- **CMUdict** — English pronunciations ([cmusphinx/cmudict](https://github.com/cmusphinx/cmudict)); see [`moonshine-tts/data/en_us/README.md`](moonshine-tts/data/en_us/README.md).
- **[hbenbel/French-Dictionary](https://github.com/hbenbel/French-Dictionary)** — related French lexicon work; liaison CSVs are curated in-repo ([`moonshine-tts/data/fr/README.md`](moonshine-tts/data/fr/README.md)).
- **[LuminosoInsight/mecab-ko-dic](https://github.com/LuminosoInsight/mecab-ko-dic)** — Korean morphological resources used elsewhere in Korean NLP ecosystems.
- **[KoichiYasuoka/esupar](https://github.com/KoichiYasuoka/esupar)** — related UD / dependency tooling from the same author family as the RoBERTa UPOS ONNX models above.
- **[KoichiYasuoka/chinese-roberta-base-upos](https://huggingface.co/KoichiYasuoka/chinese-roberta-base-upos)** — Chinese UPOS ONNX bundle ([`moonshine-tts/data/zh_hans/README.md`](moonshine-tts/data/zh_hans/README.md)).
- **[KoichiYasuoka/roberta-base-korean-morph-upos](https://huggingface.co/KoichiYasuoka/roberta-base-korean-morph-upos)** — Korean morph UPOS (tests / optional ONNX; [`moonshine-tts/data/ko/README.md`](moonshine-tts/data/ko/README.md)).
- **[KoichiYasuoka/roberta-small-japanese-char-luw-upos](https://huggingface.co/KoichiYasuoka/roberta-small-japanese-char-luw-upos)** — Japanese char-LUW UPOS ONNX ([`moonshine-tts/data/ja/README.md`](moonshine-tts/data/ja/README.md)).
- **[myshell-ai/MeloTTS](https://github.com/myshell-ai/MeloTTS)** (MIT) — multi-lingual TTS; synthetic Korean from MeloTTS was used as training material for the bundled Piper voice `ko_KR-melotts-medium` ([`moonshine-tts/data/ko/README.md`](moonshine-tts/data/ko/README.md)).