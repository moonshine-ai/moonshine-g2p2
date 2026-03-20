"""Tests for ``cmudict_ipa`` (no network; uses StringIO fixture)."""

import unittest
from io import StringIO

from cmudict_ipa import CmudictIpa, normalize_word_for_lookup, split_text_to_words


FIXTURE = """;;; CMUdict-style sample
a AH0
a(2) EY1
hello HH AH0 L OW1
bout B AW1 T
"""

TSV_FIXTURE = """# word\\tipa
hello\thəlˈoʊ
a\tə
a\tˈeɪ
bout\tbˈaʊt
"""


class TestSplitTextToWords(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(split_text_to_words(""), [])

    def test_splits_on_whitespace_runs(self):
        self.assertEqual(split_text_to_words("a  b\tc\nd"), ["a", "b", "c", "d"])

    def test_strips_leading_trailing_whitespace_tokens(self):
        self.assertEqual(split_text_to_words("  hello world  "), ["hello", "world"])


class TestNormalizeWordForLookup(unittest.TestCase):
    def test_lowercase_and_strip_edge_punctuation(self):
        self.assertEqual(normalize_word_for_lookup("Hello,"), "hello")
        self.assertEqual(normalize_word_for_lookup('("Hi")'), "hi")

    def test_empty_when_no_alnum(self):
        self.assertEqual(normalize_word_for_lookup("..."), "")
        self.assertEqual(normalize_word_for_lookup("  "), "")

    def test_preserves_internal_apostrophe(self):
        self.assertEqual(normalize_word_for_lookup("Don't"), "don't")


class TestCmudictIpa(unittest.TestCase):
    def test_load_merge_alternates_and_sort_ipa(self):
        d = CmudictIpa(StringIO(FIXTURE))
        self.assertEqual(d._ipa_by_word["a"], ["ə", "ˈeɪ"])
        self.assertEqual(d._ipa_by_word["hello"], ["həlˈoʊ"])

    def test_translate_preserves_order_and_unknown_is_empty_list(self):
        d = CmudictIpa(StringIO(FIXTURE))
        r = d.translate_to_ipa(["Hello", "nope", "a"])
        self.assertEqual(r[0], ("Hello", ["həlˈoʊ"]))
        self.assertEqual(r[1], ("nope", []))
        self.assertEqual(r[2], ("a", ["ə", "ˈeɪ"]))

    def test_translate_normalizes_surface_punctuation(self):
        d = CmudictIpa(StringIO(FIXTURE))
        r = d.translate_to_ipa(["Hello,"])
        self.assertEqual(r[0], ("Hello,", ["həlˈoʊ"]))

    def test_translate_empty_key_returns_empty_pronunciations(self):
        d = CmudictIpa(StringIO(FIXTURE))
        r = d.translate_to_ipa(["..."])
        self.assertEqual(r[0], ("...", []))


class TestCmudictIpaTsv(unittest.TestCase):
    def test_load_tsv_merges_duplicate_words(self):
        d = CmudictIpa(StringIO(TSV_FIXTURE), format="tsv")
        self.assertEqual(d._ipa_by_word["a"], ["ə", "ˈeɪ"])
        self.assertEqual(d._ipa_by_word["hello"], ["həlˈoʊ"])

    def test_translate_tsv_same_as_cmudict_fixture(self):
        d = CmudictIpa(StringIO(TSV_FIXTURE), format="tsv")
        r = d.translate_to_ipa(["Hello", "a"])
        self.assertEqual(r[0], ("Hello", ["həlˈoʊ"]))
        self.assertEqual(r[1], ("a", ["ə", "ˈeɪ"]))


if __name__ == "__main__":
    unittest.main()
