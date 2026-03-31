"""
Parity: ``piper_ipa_normalization`` (Python) vs ``ipa-postprocess.cpp`` via ``piper_ipa_normalize_cli``.

Set ``MOONSHINE_G2P_PIPER_IPA_CLI`` to the built binary if it is not at ``cpp/build/piper_ipa_normalize_cli``.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from piper_ipa_normalization import (
    coerce_unknown_chars_to_inventory,
    ipa_to_piper_ready,
    load_piper_phoneme_id_map_keys,
    normalize_g2p_ipa_for_piper,
)

_REPO = Path(__file__).resolve().parent.parent


def _cli_exe() -> Path:
    env = os.environ.get("MOONSHINE_G2P_PIPER_IPA_CLI", "").strip()
    if env:
        return Path(env)
    return _REPO / "cpp" / "build" / "piper_ipa_normalize_cli"


def _run_cpp(ipa: str, *, lang: str, json_path: Path, coerce: bool) -> str:
    exe = _cli_exe()
    if not exe.is_file():
        pytest.skip(f"C++ CLI not built: {exe}")
    cmd = [str(exe), "--lang", lang, "--phoneme-json", str(json_path)]
    if not coerce:
        cmd.append("--no-coerce")
    cmd.append("--")
    cmd.append(ipa)
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        pytest.fail(f"cli failed rc={p.returncode} stderr={p.stderr!r}")
    return p.stdout


def _write_phoneme_json(keys: list[str]) -> Path:
    fd, path = tempfile.mkstemp(suffix=".onnx.json", text=True)
    os.close(fd)
    p = Path(path)
    p.write_text(json.dumps({"phoneme_id_map": {k: [0] for k in keys}}), encoding="utf-8")
    return p


def test_parity_toy_inventory_coerce():
    keys = frozenset({"a", "z"})
    ipa = "\u03b1"  # Greek alpha — unchanged by shared ɝ→ɜː replacement; exercises coerce only.
    py = coerce_unknown_chars_to_inventory(
        normalize_g2p_ipa_for_piper(ipa, piper_lang_key="en_us"), phoneme_keys=keys
    )
    jf = _write_phoneme_json(list(keys))
    try:
        cpp = _run_cpp(ipa, lang="en_us", json_path=jf, coerce=True)
    finally:
        jf.unlink(missing_ok=True)
    assert py == cpp == "z"


def test_parity_toy_mn_dropped():
    keys = frozenset({"h"})
    ipa = "h\u0300"
    py = coerce_unknown_chars_to_inventory(
        normalize_g2p_ipa_for_piper(ipa, piper_lang_key="de"), phoneme_keys=keys
    )
    jf = _write_phoneme_json(list(keys))
    try:
        cpp = _run_cpp(ipa, lang="de", json_path=jf, coerce=True)
    finally:
        jf.unlink(missing_ok=True)
    assert py == cpp == "h"


def test_parity_hello_world_rhotic_with_en_json():
    json_path = _REPO / "data" / "en_us" / "piper-voices" / "en_US-lessac-medium.onnx.json"
    if not json_path.is_file():
        pytest.skip(f"missing {json_path}")
    keys = load_piper_phoneme_id_map_keys(json_path)
    ipa_in = "həlˈoʊ wˈɝld"
    py = ipa_to_piper_ready(ipa_in, piper_lang_key="en_us", phoneme_id_map_keys=keys, apply_coercion=True)
    cpp = _run_cpp(ipa_in, lang="en_us", json_path=json_path, coerce=True)
    assert py == cpp


def test_parity_no_coerce_normalize_only():
    keys = frozenset({"h", "ə", "l", "ˈ", "o", "ʊ", " ", "w", "ɜ", "ː", "d"})
    ipa_in = "həlˈoʊ wˈɝld"
    py = ipa_to_piper_ready(ipa_in, piper_lang_key="en_us", phoneme_id_map_keys=keys, apply_coercion=False)
    jf = _write_phoneme_json(list(keys))
    try:
        cpp = _run_cpp(ipa_in, lang="en_us", json_path=jf, coerce=False)
    finally:
        jf.unlink(missing_ok=True)
    assert py == cpp
