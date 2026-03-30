# C++ runtime data (per-language bundles)

This tree mirrors assets the C++ `moonshine_g2p` stack expects under a single **`--model-root`** (see each `README.md` for paths). Files are normally maintained under the repository root `data/` and `models/`; `cpp/data/` is a curated copy for self-contained C++ builds.

| Folder | Role |
|--------|------|
| [ar_msa](ar_msa/README.md) | Arabic MSA: tashkīl ONNX + optional lexicon |
| [de](de/README.md) | German IPA lexicon |
| [en_us](en_us/README.md) | English CMU-style lexicon + heteronym/OOV ONNX |
| [fr](fr/README.md) | French lexicon + liaison POS CSVs |
| [it](it/README.md) | Italian IPA lexicon |
| [ja](ja/README.md) | Japanese lexicon + char-LUW UPOS ONNX |
| [ko](ko/README.md) | Korean IPA lexicon |
| [nl](nl/README.md) | Dutch IPA lexicon |
| [pt_br](pt_br/README.md) | Brazilian Portuguese IPA lexicon |
| [pt_pt](pt_pt/README.md) | European Portuguese IPA lexicon |
| [ru](ru/README.md) | Russian IPA lexicon |
| [vi](vi/README.md) | Vietnamese IPA lexicon |
| [zh_hans](zh_hans/README.md) | Simplified Chinese lexicon + RoBERTa UPOS ONNX |

All commands below assume the **repository root** as the current working directory unless noted.
