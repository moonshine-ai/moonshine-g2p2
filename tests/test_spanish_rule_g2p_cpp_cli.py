#!/usr/bin/env python3
"""Compare C++ spanish_rule_g2p CLI output to spanish_rule_g2p.text_to_ipa."""
from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

from spanish_rule_g2p import dialect_from_cli_id, dialect_ids, text_to_ipa

ROOT = Path(__file__).resolve().parents[1]


def _find_cli_binary() -> Path:
    env = os.environ.get("SPANISH_RULE_G2P_CPP")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    for rel in (Path("cpp/build/moonshine_g2p"),):
        p = ROOT / rel
        if p.is_file():
            return p
    raise unittest.SkipTest(
        "moonshine_g2p binary not found; build with "
        "`cmake -S cpp -B cpp/build && cmake --build cpp/build` "
        "(ONNX enabled) or set SPANISH_RULE_G2P_CPP to the executable path."
    )


def _run_cpp(text: str, dialect: str, *, with_stress: bool, narrow: bool) -> str:
    bin_path = _find_cli_binary()
    cmd: list[str] = [str(bin_path), "--dialect", dialect, "--stdin"]
    if not with_stress:
        cmd.append("--no-stress")
    if not narrow:
        cmd.append("--broad-phonemes")
    cmd.append(text)
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        input=text,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"CLI failed ({proc.returncode}): {cmd!r}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout.rstrip("\n")


class TestSpanishRuleG2pCppCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls._bin = _find_cli_binary()
        except unittest.SkipTest:
            cls._bin = None

    def setUp(self) -> None:
        if self._bin is None:
            self.skipTest("no C++ binary")

    def test_phrases_all_dialects(self) -> None:
        phrases = [
            "Hola, México.",
            "El niño juega en Oaxaca.",
            "Yo llamo a mi hermana.",
            "Cerveza y zapato.",
            "Texas y taxi.",
        ]
        for did in dialect_ids():
            d = dialect_from_cli_id(did)
            for phrase in phrases:
                with self.subTest(dialect=did, phrase=phrase):
                    py_out = text_to_ipa(phrase, d, with_stress=True)
                    cpp_out = _run_cpp(phrase, did, with_stress=True, narrow=True)
                    self.assertEqual(py_out, cpp_out)

    def test_broad_phonemes_and_no_stress(self) -> None:
        phrase = "La casa de papel."
        did = "es-MX"
        d = dialect_from_cli_id(did, narrow_intervocalic_obstruents=False)
        py_out = text_to_ipa(phrase, d, with_stress=False)
        cpp_out = _run_cpp(phrase, did, with_stress=False, narrow=False)
        self.assertEqual(py_out, cpp_out)


if __name__ == "__main__":
    unittest.main()
