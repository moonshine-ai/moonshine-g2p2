"""Smoke tests for Japanese ONNX G2P (requires exported assets under ``data/ja/``)."""

from __future__ import annotations

import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


class TestJapaneseOnnxG2p(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _REPO / "data" / "ja" / "roberta_japanese_char_luw_upos_onnx" / "model.onnx"
        self.dict_path = _REPO / "data" / "ja" / "dict.tsv"

    def test_assets(self) -> None:
        if not self.model.is_file():
            self.skipTest("Japanese ONNX model not present (run scripts/export_japanese_ud_onnx.py)")
        from japanese_onnx_g2p import JapaneseOnnxG2p

        g = JapaneseOnnxG2p(model_dir=self.model.parent, dict_path=self.dict_path)
        ipa = g.text_to_ipa("東京に行きます。")
        self.assertIn("toɯkjoɯ", ipa)
        self.assertIn("ikimasɯ", ipa)


if __name__ == "__main__":
    unittest.main()
