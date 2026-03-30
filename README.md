# Moonshine G2P

Converts text (graphemes) to the International Pronounciation Alphabet (phonemes) for multiple languages.

Avoids using rules, instead relies on training small transformer models on large datasets to handle the conversion. The exact model architectures and stages do vary depending on the language's characteristics.

```bash
python scripts/export_wikitext_lines.py --only es_mx --out data/es_mx/wiki-text.txt

python scripts/build_oov_espeak_dataset.py --language es_mx

python train_oov.py --language es_mx
```

```
cd cpp
mkdir -p build
cd build
cmake ..
cmake --build .
./moonshine_g2p --model-root ../models --language en_us "Live, laugh, love"
./moonshine_g2p --model-root ../models --language ja "東京に行きます。"
```

## Acknowledgements

CMU

https://github.com/hbenbel/French-Dictionary

https://github.com/hankcs/HanLP

https://github.com/LuminosoInsight/mecab-ko-dic

https://github.com/KoichiYasuoka/esupar

https://huggingface.co/KoichiYasuoka/roberta-base-korean-morph-upos

https://huggingface.co/KoichiYasuoka/roberta-small-japanese-char-luw-upos