#!/usr/bin/env python3
"""
Rule-based Spanish grapheme-to-phoneme (broad IPA) conversion.

Designed around a small :class:`SpanishDialect` with CLI presets for Castilian Spain,
Mexico, neutral Latin American (``es-419``), major South American countries, the
Caribbean (CU/DO/PR), and Central America (GT)—plus custom fields for new regions.

References (high level):
- Wikipedia \"Help:IPA/Spanish\" — consonant/vowel chart, seseo vs distinción, yeísmo, rhotics.
- Standard Spanish stress: acute marks primary stress when it breaks the default pattern;
  otherwise stress is on the penultimate syllable if the word ends in a vowel, n, or s,
  and on the final syllable otherwise.

Limitations (intentionally documented):
- Letter <x> is etymology-sensitive (México, Texas, taxi, xenón, …). A small exception
  map plus a conservative default is used; expand the map or add a lexicon for production.
- Nasal place assimilation (e.g. *tengo* /ˈteŋɡo/) is optional (off by default).
- Semivocalic [j]/[w] allophones of /i//u/ next to vowels are not inserted; vowel letters
  map to full vowels unless *y* is clearly consonantal.

Primary stress is written immediately **before the stressed vowel** (after syllable onsets), similar
to eSpeak NG. Optional *narrow_intervocalic_obstruents* maps intervocalic /b d ɡ/ to **[β ð ɣ]**.

CLI: after the rule-based IPA line, a second line from eSpeak NG is printed by default
(``--no-espeak`` to disable), using the same stack as ``oov/infer.py`` / ``heteronym.espeak_heteronyms``.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Iterable, Mapping

# Default eSpeak NG voice for Spanish (CLI reference line); override with ``--espeak-voice``.
# ``es-mx`` is ideal for Mexico when installed; ``es-419`` is Latin American and more often present.
_DEFAULT_ESPEAK_VOICE = "es-419"


def espeak_ng_ipa_line(text: str, *, voice: str = _DEFAULT_ESPEAK_VOICE) -> str | None:
    """
    IPA string from libespeak-ng via ``espeak_phonemizer`` (same separator policy as
    :func:`heteronym.espeak_heteronyms.espeak_phonemize_ipa_raw`).

    Returns ``None`` if the optional dependency or engine is unavailable, or phonemization fails.
    """
    try:
        from heteronym.espeak_heteronyms import EspeakPhonemizer, espeak_phonemize_ipa_raw
    except ImportError:
        return None
    t = text.strip()
    if not t:
        return None
    try:
        phon = EspeakPhonemizer(default_voice=voice)
        raw = espeak_phonemize_ipa_raw(phon, t, voice=voice)
    except (AssertionError, OSError, RuntimeError):
        return None
    return raw or None


@dataclass(frozen=True)
class SpanishDialect:
    """
    Pronunciation knobs for rule-based G2P.

    *ce_ci_z_ipa*: IPA for ⟨z⟩ and for ⟨c⟩ before ⟨e i⟩ (``\"s\"`` = seseo, ``\"θ\"`` = distinción).
    *yeismo*: if True, ⟨ll⟩ matches ⟨y⟩ (both ``y_consonant_ipa``); if False, ⟨ll⟩ uses *ll_ipa*.
    *y_consonant_ipa*: consonantal realization for ⟨y⟩ / yeísmo ⟨ll⟩ (often ``ʝ`` on Wikipedia).
    *ll_ipa*: only used when *yeismo* is False (traditional [ʎ] in some regions).
    *x_intervocalic_default*: default IPA for ⟨x⟩ between vowels (often ``ks``).
    *x_initial_before_vowel*: default for word-initial ⟨x⟩ + vowel (many Latin-American readings use ``s`` for *xenón*-type words).
    *voiceless_velar_fricative*: IPA for ⟨j⟩, ⟨g⟩ before ⟨e i⟩, and similar (usually ``x`` in this key).
    *trill_ipa*, *tap_ipa*: ⟨rr⟩ vs intervocalic single ⟨r⟩.
    *nasal_assimilation*: if True, rewrite n→ŋ before velars, n→m before labials, n→ɱ before /f/.
    *narrow_intervocalic_obstruents*: if True, map intervocalic /b/ /d/ /ɡ/ to **[β]**, **[ð]**, **[ɣ]**.
    *coda_s_mode*: ``\"keep\"`` (default), ``\"h\"`` (word-final /s/ → **[h]**), or ``\"drop\"`` (omit word-final /s/).
        Internal syllable-coda /s/ is not rewritten yet (would need syllable-aware IPA).
    """

    id: str
    ce_ci_z_ipa: str
    yeismo: bool
    y_consonant_ipa: str
    ll_ipa: str
    x_intervocalic_default: str
    x_initial_before_vowel: str
    voiceless_velar_fricative: str
    trill_ipa: str
    tap_ipa: str
    nasal_assimilation: bool = False
    narrow_intervocalic_obstruents: bool = True
    coda_s_mode: str = "keep"


def _spanish_dialect_common(
    *,
    id: str,
    ce_ci_z_ipa: str,
    yeismo: bool,
    y_consonant_ipa: str,
    ll_ipa: str,
    voiceless_velar_fricative: str,
    coda_s_mode: str,
    narrow_intervocalic_obstruents: bool,
) -> SpanishDialect:
    return SpanishDialect(
        id=id,
        ce_ci_z_ipa=ce_ci_z_ipa,
        yeismo=yeismo,
        y_consonant_ipa=y_consonant_ipa,
        ll_ipa=ll_ipa,
        x_intervocalic_default="ks",
        x_initial_before_vowel="s",
        voiceless_velar_fricative=voiceless_velar_fricative,
        trill_ipa="r",
        tap_ipa="ɾ",
        nasal_assimilation=False,
        narrow_intervocalic_obstruents=narrow_intervocalic_obstruents,
        coda_s_mode=coda_s_mode,
    )


def mexican_spanish_dialect(*, narrow_intervocalic_obstruents: bool = True) -> SpanishDialect:
    """Mexico: seseo, yeísmo /ʝ/, ⟨j⟩/soft ⟨g⟩ as /x/, keep word-final /s/."""
    return _spanish_dialect_common(
        id="es-MX",
        ce_ci_z_ipa="s",
        yeismo=True,
        y_consonant_ipa="ʝ",
        ll_ipa="ʎ",
        voiceless_velar_fricative="x",
        coda_s_mode="keep",
        narrow_intervocalic_obstruents=narrow_intervocalic_obstruents,
    )


def latin_american_neutral_dialect(*, narrow_intervocalic_obstruents: bool = True) -> SpanishDialect:
    """Generic Latin American (seseo, yeísmo, /x/): useful when no country-specific preset fits."""
    return _spanish_dialect_common(
        id="es-419",
        ce_ci_z_ipa="s",
        yeismo=True,
        y_consonant_ipa="ʝ",
        ll_ipa="ʎ",
        voiceless_velar_fricative="x",
        coda_s_mode="keep",
        narrow_intervocalic_obstruents=narrow_intervocalic_obstruents,
    )


def castilian_spanish_dialect(
    *, narrow_intervocalic_obstruents: bool = True, dialect_id: str = "es-ES"
) -> SpanishDialect:
    """
    Castilian (Spain, mainstream urban): **distinción** (⟨z⟩ / ⟨ce⟩ / ⟨ci⟩ → /θ/), yeísmo, /x/ for ⟨j⟩/soft ⟨g⟩.
    Does not model Andalusian ceceo/seseo or Northern uvular /χ/ (use a custom :class:`SpanishDialect`).
    """
    return _spanish_dialect_common(
        id=dialect_id,
        ce_ci_z_ipa="θ",
        yeismo=True,
        y_consonant_ipa="ʝ",
        ll_ipa="ʎ",
        voiceless_velar_fricative="x",
        coda_s_mode="keep",
        narrow_intervocalic_obstruents=narrow_intervocalic_obstruents,
    )


def peninsular_distincion_dialect(*, narrow_intervocalic_obstruents: bool = True) -> SpanishDialect:
    """Backward-compatible alias for :func:`castilian_spanish_dialect` with id ``es-ES-distincion``."""
    return castilian_spanish_dialect(
        narrow_intervocalic_obstruents=narrow_intervocalic_obstruents,
        dialect_id="es-ES-distincion",
    )


def colombian_spanish_dialect(*, narrow_intervocalic_obstruents: bool = True) -> SpanishDialect:
    """Colombia (highland-oriented norm): seseo, yeísmo, /x/ (coastal [h] is common but not default here)."""
    return _spanish_dialect_common(
        id="es-CO",
        ce_ci_z_ipa="s",
        yeismo=True,
        y_consonant_ipa="ʝ",
        ll_ipa="ʎ",
        voiceless_velar_fricative="x",
        coda_s_mode="keep",
        narrow_intervocalic_obstruents=narrow_intervocalic_obstruents,
    )


def venezuelan_spanish_dialect(*, narrow_intervocalic_obstruents: bool = True) -> SpanishDialect:
    """Venezuela: seseo, yeísmo; ⟨j⟩/soft ⟨g⟩ often [h] — approximated as /h/."""
    return _spanish_dialect_common(
        id="es-VE",
        ce_ci_z_ipa="s",
        yeismo=True,
        y_consonant_ipa="ʝ",
        ll_ipa="ʎ",
        voiceless_velar_fricative="h",
        coda_s_mode="keep",
        narrow_intervocalic_obstruents=narrow_intervocalic_obstruents,
    )


def ecuadorian_spanish_dialect(*, narrow_intervocalic_obstruents: bool = True) -> SpanishDialect:
    """Ecuador: seseo, yeísmo, /x/ (Andean-influenced broad transcription)."""
    return _spanish_dialect_common(
        id="es-EC",
        ce_ci_z_ipa="s",
        yeismo=True,
        y_consonant_ipa="ʝ",
        ll_ipa="ʎ",
        voiceless_velar_fricative="x",
        coda_s_mode="keep",
        narrow_intervocalic_obstruents=narrow_intervocalic_obstruents,
    )


def peruvian_spanish_dialect(*, narrow_intervocalic_obstruents: bool = True) -> SpanishDialect:
    """Peru: seseo, yeísmo, /x/."""
    return _spanish_dialect_common(
        id="es-PE",
        ce_ci_z_ipa="s",
        yeismo=True,
        y_consonant_ipa="ʝ",
        ll_ipa="ʎ",
        voiceless_velar_fricative="x",
        coda_s_mode="keep",
        narrow_intervocalic_obstruents=narrow_intervocalic_obstruents,
    )


def chilean_spanish_dialect(*, narrow_intervocalic_obstruents: bool = True) -> SpanishDialect:
    """Chile: seseo, yeísmo; word-final /s/ often weakened — ``coda_s_mode=\"h\"`` (conservative: final only)."""
    return _spanish_dialect_common(
        id="es-CL",
        ce_ci_z_ipa="s",
        yeismo=True,
        y_consonant_ipa="ʝ",
        ll_ipa="ʎ",
        voiceless_velar_fricative="x",
        coda_s_mode="h",
        narrow_intervocalic_obstruents=narrow_intervocalic_obstruents,
    )


def argentine_spanish_dialect(*, narrow_intervocalic_obstruents: bool = True) -> SpanishDialect:
    """Argentina (Río de la Plata): seseo, yeísmo rehilado as /ʒ/ for ⟨ll⟩/⟨y⟩."""
    return _spanish_dialect_common(
        id="es-AR",
        ce_ci_z_ipa="s",
        yeismo=True,
        y_consonant_ipa="ʒ",
        ll_ipa="ʎ",
        voiceless_velar_fricative="x",
        coda_s_mode="keep",
        narrow_intervocalic_obstruents=narrow_intervocalic_obstruents,
    )


def uruguayan_spanish_dialect(*, narrow_intervocalic_obstruents: bool = True) -> SpanishDialect:
    """Uruguay: same broad preset as Argentina (ʒ yeísmo)."""
    return _spanish_dialect_common(
        id="es-UY",
        ce_ci_z_ipa="s",
        yeismo=True,
        y_consonant_ipa="ʒ",
        ll_ipa="ʎ",
        voiceless_velar_fricative="x",
        coda_s_mode="keep",
        narrow_intervocalic_obstruents=narrow_intervocalic_obstruents,
    )


def bolivian_andean_spanish_dialect(*, narrow_intervocalic_obstruents: bool = True) -> SpanishDialect:
    """Bolivia (Andean, no yeísmo): ⟨ll⟩ → /ʎ/, consonantal ⟨y⟩ → /ʝ/."""
    return _spanish_dialect_common(
        id="es-BO",
        ce_ci_z_ipa="s",
        yeismo=False,
        y_consonant_ipa="ʝ",
        ll_ipa="ʎ",
        voiceless_velar_fricative="x",
        coda_s_mode="keep",
        narrow_intervocalic_obstruents=narrow_intervocalic_obstruents,
    )


def paraguayan_spanish_dialect(*, narrow_intervocalic_obstruents: bool = True) -> SpanishDialect:
    """Paraguay: no yeísmo preset (⟨ll⟩ /ʎ/)."""
    return _spanish_dialect_common(
        id="es-PY",
        ce_ci_z_ipa="s",
        yeismo=False,
        y_consonant_ipa="ʝ",
        ll_ipa="ʎ",
        voiceless_velar_fricative="x",
        coda_s_mode="keep",
        narrow_intervocalic_obstruents=narrow_intervocalic_obstruents,
    )


def cuban_spanish_dialect(*, narrow_intervocalic_obstruents: bool = True) -> SpanishDialect:
    """Cuba: seseo, yeísmo, /h/ for ⟨j⟩/soft ⟨g⟩; word-final /s/ → [h]."""
    return _spanish_dialect_common(
        id="es-CU",
        ce_ci_z_ipa="s",
        yeismo=True,
        y_consonant_ipa="ʝ",
        ll_ipa="ʎ",
        voiceless_velar_fricative="h",
        coda_s_mode="h",
        narrow_intervocalic_obstruents=narrow_intervocalic_obstruents,
    )


def dominican_spanish_dialect(*, narrow_intervocalic_obstruents: bool = True) -> SpanishDialect:
    """Dominican Republic: like Cuban preset (seseo, /h/, final /s/ → [h])."""
    return _spanish_dialect_common(
        id="es-DO",
        ce_ci_z_ipa="s",
        yeismo=True,
        y_consonant_ipa="ʝ",
        ll_ipa="ʎ",
        voiceless_velar_fricative="h",
        coda_s_mode="h",
        narrow_intervocalic_obstruents=narrow_intervocalic_obstruents,
    )


def puerto_rican_spanish_dialect(*, narrow_intervocalic_obstruents: bool = True) -> SpanishDialect:
    """Puerto Rico: seseo, yeísmo, /h/ for ⟨j⟩/soft ⟨g⟩; word-final /s/ weakened to [h]."""
    return _spanish_dialect_common(
        id="es-PR",
        ce_ci_z_ipa="s",
        yeismo=True,
        y_consonant_ipa="ʝ",
        ll_ipa="ʎ",
        voiceless_velar_fricative="h",
        coda_s_mode="h",
        narrow_intervocalic_obstruents=narrow_intervocalic_obstruents,
    )


def guatemalan_spanish_dialect(*, narrow_intervocalic_obstruents: bool = True) -> SpanishDialect:
    """Guatemala / general Central American highland: seseo, yeísmo, /x/ (Caribbean coast differs)."""
    return _spanish_dialect_common(
        id="es-GT",
        ce_ci_z_ipa="s",
        yeismo=True,
        y_consonant_ipa="ʝ",
        ll_ipa="ʎ",
        voiceless_velar_fricative="x",
        coda_s_mode="keep",
        narrow_intervocalic_obstruents=narrow_intervocalic_obstruents,
    )


_DIALECT_BUILDERS: dict[str, Callable[..., SpanishDialect]] = {
    "es-MX": mexican_spanish_dialect,
    "es-419": latin_american_neutral_dialect,
    "es-ES": castilian_spanish_dialect,
    "es-ES-distincion": peninsular_distincion_dialect,
    "es-CO": colombian_spanish_dialect,
    "es-VE": venezuelan_spanish_dialect,
    "es-EC": ecuadorian_spanish_dialect,
    "es-PE": peruvian_spanish_dialect,
    "es-CL": chilean_spanish_dialect,
    "es-AR": argentine_spanish_dialect,
    "es-UY": uruguayan_spanish_dialect,
    "es-BO": bolivian_andean_spanish_dialect,
    "es-PY": paraguayan_spanish_dialect,
    "es-CU": cuban_spanish_dialect,
    "es-DO": dominican_spanish_dialect,
    "es-PR": puerto_rican_spanish_dialect,
    "es-GT": guatemalan_spanish_dialect,
}


def dialect_ids() -> tuple[str, ...]:
    """Sorted CLI / preset ids for :func:`dialect_from_cli_id`."""
    return tuple(sorted(_DIALECT_BUILDERS.keys()))


def dialect_from_cli_id(
    cli_id: str, *, narrow_intervocalic_obstruents: bool = True
) -> SpanishDialect:
    """Build a :class:`SpanishDialect` from a ``--dialect`` string (see :func:`dialect_ids`)."""
    key = cli_id.strip()
    if key not in _DIALECT_BUILDERS:
        raise ValueError(f"unknown dialect id {cli_id!r}; choose one of {dialect_ids()}")
    return _DIALECT_BUILDERS[key](narrow_intervocalic_obstruents=narrow_intervocalic_obstruents)


# Default eSpeak NG voice per dialect id (used when ``--espeak-voice`` is omitted).
_DEFAULT_ESPEAK_VOICE_BY_DIALECT: dict[str, str] = {
    "es-MX": "es-419",
    "es-419": "es-419",
    "es-ES": "es",
    "es-ES-distincion": "es",
    "es-CO": "es-419",
    "es-VE": "es-419",
    "es-EC": "es-419",
    "es-PE": "es-419",
    "es-CL": "es-419",
    "es-AR": "es-419",
    "es-UY": "es-419",
    "es-BO": "es-419",
    "es-PY": "es-419",
    "es-CU": "es-419",
    "es-DO": "es-419",
    "es-PR": "es-419",
    "es-GT": "es-419",
}


def default_espeak_voice_for_dialect(dialect_cli_id: str) -> str:
    """eSpeak voice name suggested for *dialect_cli_id* (fallback ``es-419``)."""
    return _DEFAULT_ESPEAK_VOICE_BY_DIALECT.get(dialect_cli_id.strip(), _DEFAULT_ESPEAK_VOICE)


_VOWELS = frozenset("aeiouáéíóúü")


def _nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


def _strip_accents(s: str) -> str:
    return "".join(ch for ch in _nfd(s) if unicodedata.category(ch) != "Mn")


def _is_vowel(ch: str) -> bool:
    return ch.lower() in _VOWELS


def _should_hiatus(a: str, b: str) -> bool:
    """Heuristic hiatus vs diphthong for adjacent vowel letters (orthographic)."""
    al, bl = a.lower(), b.lower()
    # río-style (stressed ⟨í⟩ + ⟨o⟩) vs -ción-style (⟨i⟩ + stressed ⟨ó⟩)
    if al == "í" and bl == "o":
        return True
    if al == "i" and bl == "ó":
        return False
    if al in "íú" or bl in "íú":
        return True
    ba, bb = _strip_accents(al), _strip_accents(bl)
    if ba == bb:
        return True
    sa, sb = ba in "aeo", bb in "aeo"
    if sa and sb:
        if al in "áéó" or bl in "áéó":
            return True
        if ba + bb in {"ae", "ea"}:
            return False
        return True
    return False


def _vowel_nucleus_spans(w: str) -> list[tuple[int, int]]:
    """Inclusive-exclusive (start, end) indices for each vowel nucleus in *w* (lowercase letters)."""
    out: list[tuple[int, int]] = []
    i = 0
    n = len(w)
    while i < n:
        ch = w[i]
        if ch == "y":
            if w == "y":
                out.append((i, i + 1))
                i += 1
                continue
            if i == 0 and i + 1 < n and _is_vowel(w[i + 1]):
                i += 1
                continue
            if i > 0 and _is_vowel(w[i - 1]) and i + 1 < n and _is_vowel(w[i + 1]):
                i += 1
                continue
            if i > 0 and _is_vowel(w[i - 1]) and i + 1 >= n:
                out.append((i, i + 1))
                i += 1
                continue
            if i > 0 and not _is_vowel(w[i - 1]) and (i + 1 >= n or not _is_vowel(w[i + 1])):
                out.append((i, i + 1))
                i += 1
                continue
            i += 1
            continue
        if not _is_vowel(ch):
            i += 1
            continue
        if i + 1 < n and _is_vowel(w[i + 1]):
            if _should_hiatus(ch, w[i + 1]):
                out.append((i, i + 1))
                i += 1
            else:
                out.append((i, i + 2))
                i += 2
        else:
            out.append((i, i + 1))
            i += 1
    return out


_VALID_ONSETS_2 = frozenset({"bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl", "pr", "tr", "ch"})


def _split_intervocalic_cluster(cluster: str) -> tuple[str, str]:
    """Split consonants between two nuclei into (coda, onset) for Spanish-like onset maximization."""
    if not cluster:
        return "", ""
    if cluster == "rr":
        return "", "rr"
    if len(cluster) >= 2 and cluster[-2:] in _VALID_ONSETS_2:
        return cluster[:-2], cluster[-2:]
    return cluster[:-1], cluster[-1:]


def orthographic_syllables(word: str) -> list[str]:
    """
    Split a Spanish word into rough orthographic syllables (onset maximization between nuclei).

    Good enough for default stress placement for many words; hiatus/diphthong edge cases exist.
    """
    w = re.sub(r"[^a-záéíóúüñ]", "", word.lower())
    if not w:
        return []
    spans = _vowel_nucleus_spans(w)
    if not spans:
        return [w]
    syllables: list[str] = []
    first_s = spans[0][0]
    cur = w[:first_s]
    for idx, (s, e) in enumerate(spans):
        cur += w[s:e]
        if idx + 1 < len(spans):
            next_s = spans[idx + 1][0]
            cluster = w[e:next_s]
            coda, onset = _split_intervocalic_cluster(cluster)
            syllables.append(cur + coda)
            cur = onset
        else:
            syllables.append(cur + w[e:])
    return [s for s in syllables if s]


def default_stressed_syllable_index(word: str) -> int:
    """0-based index of the stressed syllable (orthographic syllabify)."""
    w_clean = re.sub(r"[^a-záéíóúüñ]", "", word.lower())
    syl = orthographic_syllables(w_clean)
    if not syl:
        return 0
    if any(c in "áéíóú" for c in w_clean):
        for i, s in enumerate(syl):
            if any(c in "áéíóú" for c in s):
                return i
    n = len(syl)
    if n == 1:
        return 0
    last = _strip_accents(w_clean[-1])
    if last in "aeiou" or w_clean.endswith(("n", "s")):
        return max(0, n - 2)
    return n - 1


# ⟨x⟩ overrides — extend per dialect / use a lexicon for full coverage (keys are accent-stripped).
_X_EXCEPTIONS: dict[str, str] = {
    "mexico": "ˈmexiko",
    "mejico": "ˈmexiko",
    "oaxaca": "waˈxaka",
    "texas": "ˈtekas",
    "ximena": "xiˈmena",
    "xavier": "xaˈbjeɾ",
}


def _apply_nasal_assimilation(ipa: str, dialect: SpanishDialect) -> str:
    if not dialect.nasal_assimilation:
        return ipa
    s = ipa
    s = re.sub(r"n(?=[kɡ])", "ŋ", s)
    s = re.sub(r"n(?=[pbm])", "m", s)
    s = re.sub(r"n(?=f)", "ɱ", s)
    return s


def _insert_primary_stress_before_vowel(ipa: str) -> str:
    """One primary-stress mark, immediately before the first vowel letter (a e i o u)."""
    s = ipa.replace("ˈ", "")
    for i, ch in enumerate(s):
        if ch in "aeiou":
            return s[:i] + "ˈ" + s[i:]
    return "ˈ" + s


def _apply_narrow_intervocalic_obstruents(ipa: str) -> str:
    """Intervocalic /b d ɡ/ → β ð ɣ (simplified spirantization)."""
    s = ipa
    for _ in range(len(s) + 1):
        s2 = re.sub(r"(?<=[aeiou])b(?=[aeiou])", "β", s)
        s2 = re.sub(r"(?<=[aeiou])d(?=[aeiou])", "ð", s2)
        s2 = re.sub(r"(?<=[aeiou])ɡ(?=[aeiou])", "ɣ", s2)
        if s2 == s:
            break
        s = s2
    return s


def _apply_coda_s_weakening(ipa: str, mode: str) -> str:
    """
    Word-final /s/ only (avoids mangling onset /s/ in e.g. *estás*).

    *mode* ``\"h\"`` → final ``s`` becomes ``h``; ``\"drop\"`` removes it.
    Skips endings that are part of a ``ks`` cluster (e.g. *taxi*).
    """
    if mode not in ("h", "drop") or not ipa:
        return ipa
    if not ipa.endswith("s") or ipa.endswith("ks"):
        return ipa
    return ipa[:-1] + ("h" if mode == "h" else "")


def _postprocess_lexical_ipa(ipa: str, dialect: SpanishDialect, *, with_stress: bool) -> str:
    """
    Post-process dictionary IPA: align stress with onset–nucleus style unless a lone ˈ
    appears mid-string (e.g. ``xiˈmena``).
    """
    if not with_stress:
        s = ipa.replace("ˈ", "")
    elif ipa.count("ˈ") == 0 or ipa.startswith("ˈ"):
        s = _insert_primary_stress_before_vowel(ipa)
    else:
        s = ipa
    if dialect.narrow_intervocalic_obstruents:
        s = _apply_narrow_intervocalic_obstruents(s)
    s = _apply_coda_s_weakening(s, dialect.coda_s_mode)
    return s


def _y_is_consonant(word: str, i: int) -> bool:
    w = word
    if w[i].lower() != "y":
        return False
    prev_v = i > 0 and _is_vowel(w[i - 1])
    next_v = i + 1 < len(w) and _is_vowel(w[i + 1])
    if prev_v and next_v:
        return True
    if i == 0 and next_v:
        return True
    if not prev_v and not next_v and i == len(w) - 1:
        return False
    if not prev_v and next_v:
        return True
    return False


_VOWEL_LETTER_IPA = {"e": "e", "i": "i", "é": "e", "í": "i"}


def _letters_to_ipa_no_stress(
    letters: str, dialect: SpanishDialect, *, grapheme_offset: int = 0
) -> str:
    """
    Map a Spanish letter string to IPA without stress marks.

    *grapheme_offset* is the index of *letters[0]* inside the full word (for ⟨x⟩-at-word-start).
    """
    i = 0
    out: list[str] = []
    n = len(letters)
    lw = letters.lower()

    def peek_vowel(j: int) -> bool:
        k = j
        while k < n:
            if lw[k] == "h":
                k += 1
                continue
            return _is_vowel(lw[k])
        return False

    def prev_phoneme_was_vowel() -> bool:
        if not out:
            return False
        last = out[-1]
        return any(v in last for v in "aeiou")

    while i < n:
        ch = lw[i]

        if ch == "h":
            i += 1
            continue

        if ch == "y":
            if lw == "y":
                out.append("i")
                i += 1
                continue
            if _y_is_consonant(letters, i):
                out.append(dialect.y_consonant_ipa)
                i += 1
                continue
            out.append("i")
            i += 1
            continue

        if ch == "ñ":
            out.append("ɲ")
            i += 1
            continue

        if i + 1 < n and lw[i : i + 2] == "rr":
            out.append(dialect.trill_ipa)
            i += 2
            continue

        if i + 1 < n and lw[i : i + 2] == "ch":
            out.append("tʃ")
            i += 2
            continue

        if i + 1 < n and lw[i : i + 2] == "ll":
            out.append(dialect.y_consonant_ipa if dialect.yeismo else dialect.ll_ipa)
            i += 2
            continue

        if ch == "q" and i + 2 < n and lw[i + 1] == "u" and lw[i + 2] in _VOWEL_LETTER_IPA:
            out.append("k")
            out.append(_VOWEL_LETTER_IPA[lw[i + 2]])
            i += 3
            continue

        if ch == "g" and i + 2 < n and lw[i + 1] == "ü" and lw[i + 2] in _VOWEL_LETTER_IPA:
            out.append("ɡ")
            out.append("w")
            out.append(_VOWEL_LETTER_IPA[lw[i + 2]])
            i += 3
            continue

        if ch == "g" and i + 2 < n and lw[i + 1] == "u" and lw[i + 2] in _VOWEL_LETTER_IPA:
            out.append("ɡ")
            out.append(_VOWEL_LETTER_IPA[lw[i + 2]])
            i += 3
            continue

        if ch == "g" and i + 1 < n and lw[i + 1] in "eiéí":
            out.append(dialect.voiceless_velar_fricative)
            i += 1
            continue

        if ch == "c" and i + 3 < n and lw[i : i + 4] == "ción":
            out.append(dialect.ce_ci_z_ipa)
            out.append("j")
            out.append("o")
            out.append("n")
            i += 4
            continue
        if ch == "c" and i + 2 < n and lw[i : i + 3] == "ció":
            out.append(dialect.ce_ci_z_ipa)
            out.append("j")
            out.append("o")
            i += 3
            continue

        if ch == "c" and i + 1 < n and lw[i + 1] in "eiéí":
            out.append(dialect.ce_ci_z_ipa)
            i += 1
            continue

        if ch == "z":
            out.append(dialect.ce_ci_z_ipa)
            i += 1
            continue

        if ch == "x":
            abs_pos = grapheme_offset + i
            next_v = peek_vowel(i + 1)
            if abs_pos == 0 and next_v:
                out.append(dialect.x_initial_before_vowel)
            elif prev_phoneme_was_vowel() and next_v:
                out.append(dialect.x_intervocalic_default)
            else:
                out.append(dialect.x_intervocalic_default)
            i += 1
            continue

        if ch == "j":
            out.append(dialect.voiceless_velar_fricative)
            i += 1
            continue

        if ch == "c":
            out.append("k")
            i += 1
            continue

        if ch == "r":
            at_word_start = i == 0 or not lw[i - 1].isalpha()
            after_lns = i > 0 and lw[i - 1] in "lns"
            if at_word_start or after_lns:
                out.append(dialect.trill_ipa)
            elif prev_phoneme_was_vowel() and peek_vowel(i + 1):
                out.append(dialect.tap_ipa)
            else:
                out.append(dialect.tap_ipa)
            i += 1
            continue

        simple_map = {
            "a": "a",
            "e": "e",
            "i": "i",
            "o": "o",
            "u": "u",
            "á": "a",
            "é": "e",
            "í": "i",
            "ó": "o",
            "ú": "u",
            "ü": "w",
            "b": "b",
            "v": "b",
            "d": "d",
            "f": "f",
            "k": "k",
            "l": "l",
            "m": "m",
            "n": "n",
            "p": "p",
            "s": "s",
            "t": "t",
            "w": "w",
            "g": "ɡ",
        }
        if ch in simple_map:
            out.append(simple_map[ch])
            i += 1
            continue

        i += 1

    ipa = "".join(out)
    return _apply_nasal_assimilation(ipa, dialect)


def word_to_ipa(word: str, dialect: SpanishDialect, *, with_stress: bool = True) -> str:
    """
    Convert a single Spanish word (no spaces) to broad IPA using *dialect*.

    Unknown non-letters are dropped from the grapheme pass but may affect boundaries.
    """
    if not word:
        return ""
    wraw = word.strip()
    wkey = _strip_accents(wraw.lower())
    if wkey in _X_EXCEPTIONS and dialect.id.startswith("es"):
        return _postprocess_lexical_ipa(_X_EXCEPTIONS[wkey], dialect, with_stress=with_stress)

    letters = re.sub(r"[^a-záéíóúüñA-ZÁÉÍÓÚÜÑ]", "", wraw)
    if not letters:
        return ""

    lw = letters.lower()
    syl = orthographic_syllables(lw)
    stress_idx = default_stressed_syllable_index(lw) if with_stress else -1
    offset = 0
    parts: list[str] = []
    for s in syl:
        parts.append(_letters_to_ipa_no_stress(s, dialect, grapheme_offset=offset))
        offset += len(s)
    if with_stress and parts and 0 <= stress_idx < len(parts):
        parts[stress_idx] = _insert_primary_stress_before_vowel(parts[stress_idx])
    ipa = "".join(parts)
    if dialect.narrow_intervocalic_obstruents:
        ipa = _apply_narrow_intervocalic_obstruents(ipa)
    ipa = _apply_coda_s_weakening(ipa, dialect.coda_s_mode)
    return ipa


_TOKEN_RE = re.compile(r"[\wáéíóúüñÁÉÍÓÚÜÑ]+|[^\w\sáéíóúüñÁÉÍÓÚÜÑ]+|\s+")


def text_to_ipa(
    text: str,
    dialect: SpanishDialect | None = None,
    *,
    with_stress: bool = True,
    word_exceptions: Mapping[str, str] | None = None,
) -> str:
    """
    Tokenize *text* into words (Unicode letters + Spanish accents) and map each word with
    :func:`word_to_ipa`. Punctuation and whitespace are preserved as separate tokens
    (whitespace collapsed to single spaces between words for readability).
    """
    dialect = dialect or mexican_spanish_dialect()
    exc = dict(_X_EXCEPTIONS)
    if word_exceptions:
        exc.update({k.lower(): v for k, v in word_exceptions.items()})

    def word_ipa(w: str) -> str:
        k = _strip_accents(w.lower())
        if k in exc:
            return _postprocess_lexical_ipa(exc[k], dialect, with_stress=with_stress)
        return word_to_ipa(w, dialect, with_stress=with_stress)

    parts: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok.isspace():
            parts.append(" ")
        elif re.fullmatch(r"[\wáéíóúüñÁÉÍÓÚÜÑ]+", tok):
            parts.append(word_ipa(tok))
        else:
            parts.append(tok)
    out = "".join(parts)
    out = re.sub(r" +", " ", out).strip()
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rule-based Spanish text to broad IPA (dialect-configurable).")
    p.add_argument("text", nargs="*", help="Spanish text (if empty, read stdin)")
    p.add_argument(
        "--dialect",
        default="es-MX",
        choices=dialect_ids(),
        metavar="ID",
        help=f"Pronunciation preset (default: es-MX). IDs: {', '.join(dialect_ids())}.",
    )
    p.add_argument("--no-stress", action="store_true", help="Omit primary stress marks ˈ.")
    p.add_argument("--stdin", action="store_true", help="Read full text from stdin (ignores positional text).")
    p.add_argument(
        "--no-espeak",
        action="store_true",
        help="Do not print a second IPA line from eSpeak NG (requires espeak-phonemizer + libespeak-ng).",
    )
    p.add_argument(
        "--espeak-voice",
        type=str,
        default=None,
        metavar="VOICE",
        help="eSpeak voice for the reference line (default: pick by --dialect, else es-419).",
    )
    p.add_argument(
        "--broad-phonemes",
        action="store_true",
        help="Keep intervocalic /b d ɡ/ as stops (no β ð ɣ); default matches narrower Spanish allophones.",
    )
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)
    if args.stdin or not args.text:
        import sys

        raw = sys.stdin.read()
    else:
        raw = " ".join(args.text)
    dialect = dialect_from_cli_id(
        args.dialect,
        narrow_intervocalic_obstruents=not args.broad_phonemes,
    )
    print(text_to_ipa(raw, dialect, with_stress=not args.no_stress))
    if not args.no_espeak:
        voice = args.espeak_voice or default_espeak_voice_for_dialect(args.dialect)
        espeak_line = espeak_ng_ipa_line(raw, voice=voice)
        if espeak_line is not None:
            print(f"{espeak_line} (espeak-ng)")


if __name__ == "__main__":
    main()
