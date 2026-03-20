"""Tests for ``cmudict_ipa`` (no network; uses StringIO fixture)."""

import unittest
from io import StringIO

from cmudict_ipa import CmudictIpa


FIXTURE = """;;; CMUdict-style sample
a AH0
a(2) EY1
hello HH AH0 L OW1
bout B AW1 T
"""


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


if __name__ == "__main__":
    unittest.main()
