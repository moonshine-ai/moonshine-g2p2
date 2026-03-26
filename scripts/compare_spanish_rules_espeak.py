#!/usr/bin/env python3
"""Compare spanish_rule_g2p vs eSpeak for a word list; print tiers for manual review."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from spanish_rule_g2p import espeak_ng_ipa_line, mexican_spanish_dialect, word_to_ipa

# ~100 high-frequency / pedagogically common Spanish words (mixed lengths, tricky spellings).
WORDS = [
    "hola",
    "adiós",
    "gracias",
    "por",
    "favor",
    "sí",
    "no",
    "bien",
    "mal",
    "casa",
    "agua",
    "tiempo",
    "día",
    "año",
    "vida",
    "mundo",
    "trabajo",
    "estado",
    "país",
    "ciudad",
    "persona",
    "hombre",
    "mujer",
    "niño",
    "familia",
    "amigo",
    "corazón",
    "mano",
    "ojo",
    "cabeza",
    "número",
    "parte",
    "lugar",
    "momento",
    "forma",
    "caso",
    "punto",
    "cosa",
    "historia",
    "manera",
    "verdad",
    "problema",
    "programa",
    "empresa",
    "servicio",
    "sistema",
    "derecho",
    "política",
    "guerra",
    "gobierno",
    "mercado",
    "desarrollo",
    "proceso",
    "grupo",
    "social",
    "nacional",
    "general",
    "humano",
    "público",
    "grande",
    "nuevo",
    "último",
    "mejor",
    "mismo",
    "menos",
    "mayor",
    "donde",
    "cuando",
    "porque",
    "como",
    "más",
    "cada",
    "otro",
    "tan",
    "muy",
    "que",
    "cual",
    "quien",
    "uno",
    "dos",
    "tres",
    "elección",
    "canción",
    "lluvia",
    "calle",
    "cerveza",
    "zapato",
    "jugar",
    "jalapeño",
    "México",
    "yoga",
    "rey",
    "ley",
    "hacer",
    "decir",
    "ir",
    "ver",
    "dar",
    "saber",
    "querer",
    "llegar",
    "pasar",
    "deber",
    "poner",
    "parecer",
    "quedar",
    "hablar",
    "llevar",
    "dejar",
    "seguir",
    "encontrar",
    "llamar",
    "venir",
    "pensar",
    "salir",
    "volver",
    "tomar",
    "conocer",
    "vivir",
    "sentir",
    "tratar",
    "mirar",
    "contar",
    "empezar",
    "esperar",
    "buscar",
    "existir",
    "entrar",
    "trabajar",
    "escribir",
    "perder",
    "producir",
    "ocurrir",
    "entender",
    "pedir",
    "recibir",
    "recordar",
    "terminar",
    "permitir",
    "aparecer",
    "conseguir",
    "comenzar",
    "servir",
    "sacar",
    "necesitar",
    "mantener",
    "resultar",
    "leer",
    "caer",
    "oír",
    "incluir",
]


def _normalize_loose(s: str) -> str:
    """Aggressive normalization for broad agreement (not linguistic equality)."""
    t = s.lower()
    t = re.sub(r"[ˈˌ]", "", t)
    # Common notation alternates
    t = t.replace("ɾ", "r")
    t = t.replace("β", "b")
    t = t.replace("ð", "d")
    t = t.replace("ɣ", "g")
    t = t.replace("ʝ", "j")
    t = t.replace("ɡ", "g")
    # Glides / diphthong spelling
    t = re.sub(r"ij", "i", t)
    t = re.sub(r"ji", "i", t)
    t = re.sub(r"ja", "ia", t)  # gracias-style
    t = re.sub(r"wa", "ua", t)  # agua-style
    t = re.sub(r"we", "ue", t)
    t = re.sub(r"wi", "ui", t)
    t = re.sub(r"wo", "uo", t)
    t = re.sub(r"tj", "ti", t)  # tiempo-style onset
    return t


def _lev(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(
                min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1))
            )
        prev = cur
    return prev[lb]


def main() -> None:
    d = mexican_spanish_dialect()
    voice = "es-419"
    words = WORDS[:100]
    rows: list[tuple[str, str, str | None, str, int, int]] = []
    missing = 0
    for w in words:
        r = word_to_ipa(w, d)
        e = espeak_ng_ipa_line(w, voice=voice)
        if e is None:
            missing += 1
            rows.append((w, r, None, "no_espeak", 99, 99))
            continue
        nr, ne = _normalize_loose(r), _normalize_loose(e)
        ld = _lev(nr, ne)
        ld_raw = _lev(r.lower().replace("ˈ", "").replace("ˌ", ""), e.lower().replace("ˈ", "").replace("ˌ", ""))
        if nr == ne:
            tier = "A_same_loose"
        elif ld <= 2:
            tier = "B_close"
        elif ld <= 5:
            tier = "C_moderate"
        else:
            tier = "D_large"
        rows.append((w, r, e, tier, ld, ld_raw))

    print(f"words={len(words)} (of {len(WORDS)} in list) espeak_missing={missing}\n")
    for tier_name in ("A_same_loose", "B_close", "C_moderate", "D_large", "no_espeak"):
        bucket = [x for x in rows if x[3] == tier_name]
        print(f"=== {tier_name} ({len(bucket)}) ===")
        for w, r, e, _t, ld, ldr in bucket:
            if e is None:
                print(f"  {w:18} rules={r}")
            else:
                print(f"  {w:18} rules={r}")
                print(f"  {'':18} espeak={e}  (lev_loose={ld} lev_raw~={ldr})")
        print()

    print("=== D_large detail (review) ===")
    for w, r, e, t, ld, ldr in rows:
        if t == "D_large":
            print(f"{w}\n  R {r}\n  E {e}\n  lev_loose={ld}\n")


if __name__ == "__main__":
    main()
