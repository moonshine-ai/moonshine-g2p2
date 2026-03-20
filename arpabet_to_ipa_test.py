"""Tests for CMU ARPAbet → IPA mapping."""

import unittest

from arpabet_to_ipa import arpabet_phone_to_ipa, arpabet_words_to_ipa


class TestArpabetPhoneToIpa(unittest.TestCase):
    def test_empty_token(self):
        self.assertEqual(arpabet_phone_to_ipa(""), "")

    def test_consonants_no_stress_digit(self):
        self.assertEqual(arpabet_phone_to_ipa("T"), "t")
        self.assertEqual(arpabet_phone_to_ipa("CH"), "tʃ")
        self.assertEqual(arpabet_phone_to_ipa("NG"), "ŋ")

    def test_ah_stress(self):
        self.assertEqual(arpabet_phone_to_ipa("AH0"), "ə")
        self.assertEqual(arpabet_phone_to_ipa("AH1"), "\u02C8ʌ")
        self.assertEqual(arpabet_phone_to_ipa("AH2"), "\u02CCʌ")

    def test_er_stress(self):
        self.assertEqual(arpabet_phone_to_ipa("ER0"), "ɚ")
        self.assertEqual(arpabet_phone_to_ipa("ER1"), "\u02C8ɝ")

    def test_primary_secondary_stress_vowel(self):
        self.assertEqual(arpabet_phone_to_ipa("EY1"), "\u02C8eɪ")
        self.assertEqual(arpabet_phone_to_ipa("IY2"), "\u02CCi")

    def test_unstressed_vowel_other_than_ah_er(self):
        self.assertEqual(arpabet_phone_to_ipa("IH0"), "ɪ")

    def test_unknown_token_passthrough(self):
        self.assertEqual(arpabet_phone_to_ipa("XX9"), "XX9")


class TestArpabetWordsToIpa(unittest.TestCase):
    def test_empty_sequence(self):
        self.assertEqual(arpabet_words_to_ipa([]), "")
        self.assertEqual(arpabet_words_to_ipa([""]), "")

    def test_hello(self):
        self.assertEqual(
            arpabet_words_to_ipa(["HH", "AH0", "L", "OW1"]),
            "həl\u02C8oʊ",
        )

    def test_skips_empty_tokens(self):
        self.assertEqual(arpabet_words_to_ipa(["T", "", "AH0"]), "tə")


if __name__ == "__main__":
    unittest.main()
