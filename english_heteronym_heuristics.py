"""
Context heuristics for English **heteronym** G2P (same spelling, different IPA).

Used by :class:`english_rule_g2p.EnglishLexiconRuleG2p` when
:meth:`~english_rule_g2p.EnglishLexiconRuleG2p.g2p_span` sees multiple CMUdict
pronunciations. This is not a substitute for the heteronym ONNX model; it is a
lightweight deterministic baseline.

Closed-class **coarse POS** hints come from :mod:`english_minimal_pos` (pronouns,
determiners, modals, prepositions, etc.).

Disambiguation follows a single **ordered decision list** (see
:mod:`english_heteronym_rulelist`): multi-token **exceptions** and specific
phrases are checked before broader neighbor rules.

Candidate ordering (fallback when no rule fires) prefers
``models/en_us/heteronym/homograph_index.json`` ``ordered_candidates`` so the
default matches the heteronym training prior where available.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from english_heteronym_rulelist import NeighborContext, RuleTuple, apply_ordered_rules
from english_minimal_pos import (
    coarse_pos_tag,
    have_has_had_immediate_left,
    immediate_left_is_det_or_poss,
    immediate_left_is_pronoun,
    left_token_suggests_infinitive_or_finite_verb_after,
    modal_or_aux_in_left_window,
    present_hint_in_left_window,
    right_token_starts_prep_phrase,
    temporal_past_hint_in_left_window,
)

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_HOMOGRAPH_INDEX = _REPO_ROOT / "models" / "en_us" / "heteronym" / "homograph_index.json"

# --- context extraction ---------------------------------------------------------

_WORD_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def context_neighbor_words(
    text: str,
    span_s: int,
    span_e: int,
    *,
    window: int = 6,
) -> tuple[list[str], list[str]]:
    """
    Lowercase word tokens to the left and right of ``text[span_s:span_e]``.

    Apostrophe words are kept (``they'll`` → ``they'll`` lowercased).
    """
    left = text[:span_s]
    right = text[span_e:]
    left_toks = [m.group(0).lower() for m in _WORD_TOKEN_RE.finditer(left)]
    right_toks = [m.group(0).lower() for m in _WORD_TOKEN_RE.finditer(right)]
    return left_toks[-window:], right_toks[:window]


# --- homograph index ------------------------------------------------------------


def load_homograph_ordered_ipa(path: str | Path | None = None) -> dict[str, list[str]]:
    """``homograph_key`` -> IPA strings in heteronym-model candidate order."""
    p = Path(path) if path else _DEFAULT_HOMOGRAPH_INDEX
    if not p.is_file():
        return {}
    blob = json.loads(p.read_text(encoding="utf-8"))
    oc = blob.get("ordered_candidates") or {}
    out: dict[str, list[str]] = {}
    for k, v in oc.items():
        if isinstance(v, list) and len(v) >= 1:
            key = str(k).lower()
            out[key] = [unicodedata.normalize("NFC", str(x)) for x in v if x]
    return out


def merge_tsv_and_homograph_candidates(
    word_key: str,
    tsv_ipas: list[str],
    homograph_order: dict[str, list[str]],
) -> list[str]:
    """
    Ordered IPA list: homograph-index order intersected with TSV, then any
    remaining TSV pronunciations (stable, sorted).
    """
    if len(tsv_ipas) <= 1:
        return list(tsv_ipas)
    tset = {unicodedata.normalize("NFC", x) for x in tsv_ipas}
    ho = homograph_order.get(word_key.lower())
    if not ho:
        return sorted(tsv_ipas)
    out: list[str] = []
    seen: set[str] = set()
    for x in ho:
        xn = unicodedata.normalize("NFC", x)
        if xn in tset and xn not in seen:
            out.append(xn)
            seen.add(xn)
    for x in sorted(tsv_ipas):
        xn = unicodedata.normalize("NFC", x)
        if xn not in seen:
            out.append(xn)
            seen.add(xn)
    return out if out else sorted(tsv_ipas)


# --- per-word heuristics ----------------------------------------------------------

_MODALS_AUX = frozenset(
    {
        "will",
        "wo",  # we'll split? token is "we'll" as whole word
        "can",
        "could",
        "should",
        "would",
        "might",
        "must",
        "may",
        "to",
        "do",
        "does",
        "did",
        "don't",
        "doesn't",
        "didn't",
        "ca",
    }
)

_LIVE_ADJ_FOLLOW = frozenset(
    {
        "performances",
        "performance",
        "album",
        "albums",
        "show",
        "shows",
        "concert",
        "concerts",
        "broadcast",
        "broadcasts",
        "television",
        "recording",
        "recordings",
        "dvd",
        "video",
        "music",
        "band",
        "tour",
        "audience",
        "rendition",
        "tracks",
        "versions",
        "version",
        "arcade",
        "lounge",
        "action",
        "instrumentation",
        "marketplace",
        "goldfish",
        "mice",
        "fish",
        "wire",
        "steam",
        "oak",
        "ammunition",
        "virus",
        "bait",
        "culture",
        "nation",
        "together",  # "played live together" — still often adj; weak
    }
)

_LIVE_VERB_FOLLOW = frozenset(
    {
        "in",
        "happily",
        "alone",
        "forever",
        "here",
        "there",
        "now",
        "without",
        "through",
        "near",
        "longer",
        "comfortably",
        "life",
        "young",
        "out",
        "within",
        "or",
    }
)

_PAST_HINTS = frozenset({"yesterday", "ago", "last", "previously", "once", "earlier"})
_PRESENT_HINTS = frozenset(
    {
        "will",
        "can",
        "must",
        "always",
        "often",
        "never",
        "still",
        "please",
        "ca",
    }
)

_DET_VERB_OBJECT = frozenset(
    {
        "the",
        "a",
        "an",
        "my",
        "your",
        "his",
        "her",
        "their",
        "our",
        "this",
        "that",
        "these",
        "those",
        "it",
        "up",
        "down",
        "all",
        "each",
        "every",
    }
)

_CLOSE_ADJ_LEFT = frozenset(
    {
        "so",
        "very",
        "too",
        "how",
        "as",
        "stay",
        "stays",
        "stayed",
        "keeping",
        "keep",
        "kept",
        "come",
        "came",
        "comes",
        "get",
        "gets",
        "got",
        "drawn",
        "drew",
        "seems",
        "seemed",
        "appears",
        "appeared",
        "looks",
        "looked",
    }
)


def _pick_live_verb(cands: list[str]) -> str | None:
    for c in cands:
        if "aɪv" in c or "aɪvz" in c:
            continue
        if "ˈɪv" in c or "ɪvz" in c:
            return c
    return None


def _pick_live_adj(cands: list[str]) -> str | None:
    for c in cands:
        if "aɪv" in c or "aɪvz" in c:
            return c
    return None


def _pick_read_past(cands: list[str]) -> str | None:
    for c in cands:
        if "ˈɛd" in c or c.endswith("ɛd"):
            return c
    return None


def _pick_read_present(cands: list[str]) -> str | None:
    for c in cands:
        if "ˈid" in c and "ɛd" not in c:
            return c
    return None


def _pick_close_verb(cands: list[str]) -> str | None:
    for c in cands:
        if c.endswith("z") or "oʊz" in c:
            return c
    return None


def _pick_close_adj(cands: list[str]) -> str | None:
    for c in cands:
        if c.endswith("s") and not c.endswith("z"):
            return c
        if "oʊs" in c and "oʊz" not in c:
            return c
    return None


def _pick_use_noun(cands: list[str]) -> str | None:
    for c in cands:
        if "ˈuz" in c:
            continue
        if c.endswith("z"):
            continue
        if "ˈus" in c or (c.endswith("s") and not c.endswith("z")):
            return c
    return None


def _pick_use_verb(cands: list[str]) -> str | None:
    for c in cands:
        if "ˈuz" in c or c.endswith("z"):
            return c
    return None


def _pick_content_noun(cands: list[str]) -> str | None:
    for c in cands:
        if c.startswith("kˈ") or "ˈɑnt" in c:
            return c
    return None


def _pick_content_adj(cands: list[str]) -> str | None:
    for c in cands:
        if c.startswith("kə") and "ˈɛnt" in c:
            return c
    return None


# Verbs that typically precede *to* + infinitive (*read* as present stem).
_VERB_BEFORE_TO_READ: frozenset[str] = frozenset(
    {
        "going",
        "want",
        "wants",
        "wanted",
        "need",
        "needs",
        "needed",
        "like",
        "liked",
        "likes",
        "love",
        "loves",
        "loved",
        "hate",
        "hates",
        "hated",
        "try",
        "tries",
        "tried",
        "learn",
        "learns",
        "learned",
        "learnt",
        "begin",
        "begins",
        "began",
        "begun",
        "start",
        "starts",
        "started",
        "cease",
        "ceases",
        "ceased",
        "help",
        "helps",
        "helped",
        "forget",
        "forgets",
        "forgot",
        "remember",
        "remembers",
        "remembered",
        "refuse",
        "refuses",
        "refused",
        "decide",
        "decides",
        "decided",
        "choose",
        "chooses",
        "chose",
        "chosen",
        "hope",
        "hopes",
        "hoped",
        "expect",
        "expects",
        "expected",
        "fail",
        "fails",
        "failed",
        "manage",
        "manages",
        "managed",
        "continue",
        "continues",
        "continued",
    }
)


def _build_ordered_disambig_rules() -> list[RuleTuple]:
    """
    Decision-list order: **exceptions** and long **multi-token** patterns first,
    then POS-backed single-token rules, then defaults (handled by no match →
    caller ``default_primary``).
    """

    def _lt_in_modals_aux(ctx: NeighborContext) -> bool:
        return ctx.wl1 in _MODALS_AUX

    def _read_infinitive_to(ctx: NeighborContext) -> bool:
        return ctx.wl1 == "to" and ctx.wl2 in _VERB_BEFORE_TO_READ

    def _read_past_tail(ctx: NeighborContext) -> bool:
        return ctx.left_tail_contains_any(_PAST_HINTS, window=6)

    def _read_pron_conditional_past(ctx: NeighborContext) -> bool:
        return immediate_left_is_pronoun(ctx.left) and temporal_past_hint_in_left_window(
            ctx.left
        ) and not modal_or_aux_in_left_window(ctx.left, window=5)

    def _read_pron_conditional_present(ctx: NeighborContext) -> bool:
        return immediate_left_is_pronoun(ctx.left) and present_hint_in_left_window(
            ctx.left
        ) and not temporal_past_hint_in_left_window(ctx.left)

    # --- live: multi-token **adj** broadcast / stage phrases before generic ``on`` ---
    _LIVE_ADJ_RIGHT2: tuple[tuple[str, ...], ...] = (
        ("on", "stage"),
        ("on", "tour"),
        ("on", "tv"),
        ("on", "the", "air"),
        ("at", "the"),
        ("from", "the"),
        ("during", "the"),
        ("during", "a"),
    )

    rules: list[RuleTuple] = []

    # live — exceptions: verbal ``live on`` when continuation is clearly not broadcast
    rules.append(
        (
            "live_exception_on_prep_object",
            frozenset({"live"}),
            lambda c: c.startswith_right("on", "the", "edge")
            or c.startswith_right("on", "the", "street")
            or c.startswith_right("on", "the", "planet")
            or c.startswith_right("on", "nothing")
            or c.startswith_right("on", "a", "budget"),
            _pick_live_verb,
        )
    )
    for i, pref in enumerate(_LIVE_ADJ_RIGHT2):
        rules.append(
            (
                f"live_adj_phrase_{i}",
                frozenset({"live"}),
                lambda c, p=pref: c.startswith_right(*p),
                _pick_live_adj,
            )
        )

    rules.extend(
        [
            (
                "live_modal_do_to",
                frozenset({"live"}),
                lambda c: left_token_suggests_infinitive_or_finite_verb_after(c.left)
                or _lt_in_modals_aux(c),
                _pick_live_verb,
            ),
            (
                "live_pron_then_prep",
                frozenset({"live"}),
                lambda c: immediate_left_is_pronoun(c.left)
                and right_token_starts_prep_phrase(c.right),
                _pick_live_verb,
            ),
            (
                "live_wr1_adj_noun",
                frozenset({"live"}),
                lambda c: c.wr1 in _LIVE_ADJ_FOLLOW,
                _pick_live_adj,
            ),
            (
                "live_wr1_verb_hint",
                frozenset({"live"}),
                lambda c: c.wr1 in _LIVE_VERB_FOLLOW and c.wr1 != "together",
                _pick_live_verb,
            ),
            (
                "live_wr1_with",
                frozenset({"live"}),
                lambda c: c.wr1 == "with",
                _pick_live_verb,
            ),
            (
                "live_wr1_on_default_adj",
                frozenset({"live"}),
                lambda c: c.wr1 == "on",
                _pick_live_adj,
            ),
            (
                "live_det_poss_adj",
                frozenset({"live"}),
                lambda c: immediate_left_is_det_or_poss(c.left)
                and bool(c.wr1)
                and c.wr1 not in _LIVE_VERB_FOLLOW,
                _pick_live_adj,
            ),
            (
                "live_a_adj",
                frozenset({"live"}),
                lambda c: c.wl1 == "a" and bool(c.wr1) and c.wr1 not in _LIVE_VERB_FOLLOW,
                _pick_live_adj,
            ),
        ]
    )

    # read — perfect / infinitive before loose temporal tail
    rules.extend(
        [
            (
                "read_have_has_had",
                frozenset({"read", "reads"}),
                lambda c: have_has_had_immediate_left(c.left),
                _pick_read_past,
            ),
            (
                "read_infinitive_to",
                frozenset({"read", "reads"}),
                _read_infinitive_to,
                _pick_read_present,
            ),
            (
                "read_past_adverbs_tail",
                frozenset({"read", "reads"}),
                _read_past_tail,
                _pick_read_past,
            ),
            (
                "read_would_idiom",
                frozenset({"read", "reads"}),
                lambda c: c.wl1 == "'d"
                and len(c.left) >= 2
                and c.wl2 in {"i", "you", "he", "she", "we", "they", "it", "who"},
                _pick_read_past,
            ),
            (
                "read_modal_present_wl1",
                frozenset({"read", "reads"}),
                lambda c: left_token_suggests_infinitive_or_finite_verb_after(c.left)
                or c.wl1
                in {
                    "will",
                    "to",
                    "can",
                    "must",
                    "do",
                    "does",
                    "don't",
                    "doesn't",
                    "ca",
                },
                _pick_read_present,
            ),
            (
                "read_present_hint_tail",
                frozenset({"read", "reads"}),
                lambda c: c.left_tail_contains_any(_PRESENT_HINTS, window=6),
                _pick_read_present,
            ),
            (
                "read_pron_temporal_past",
                frozenset({"read", "reads"}),
                _read_pron_conditional_past,
                _pick_read_past,
            ),
            (
                "read_pron_present_hint",
                frozenset({"read", "reads"}),
                _read_pron_conditional_present,
                _pick_read_present,
            ),
        ]
    )

    # close — phrasal **down**/**up** before determiner object
    rules.extend(
        [
            (
                "close_phrasal_down",
                frozenset({"close"}),
                lambda c: c.startswith_right("down"),
                _pick_close_verb,
            ),
            (
                "close_phrasal_up_object",
                frozenset({"close"}),
                lambda c: c.startswith_right("up")
                and len(c.right) >= 2
                and c.wr2 in _DET_VERB_OBJECT,
                _pick_close_verb,
            ),
            (
                "close_det_object",
                frozenset({"close"}),
                lambda c: c.wr1 in _DET_VERB_OBJECT,
                _pick_close_verb,
            ),
            (
                "close_adj_left",
                frozenset({"close"}),
                lambda c: c.wl1 in _CLOSE_ADJ_LEFT,
                _pick_close_adj,
            ),
            (
                "close_to_adj",
                frozenset({"close"}),
                lambda c: c.wr1 == "to",
                _pick_close_adj,
            ),
        ]
    )

    # use — **make use** / **in use** before infinitival *to*
    rules.extend(
        [
            (
                "use_make_noun",
                frozenset({"use"}),
                lambda c: c.wl1 == "make",
                _pick_use_noun,
            ),
            (
                "use_in_noun",
                frozenset({"use"}),
                lambda c: c.wl1 == "in" and c.wr1 not in {"of", "for", "to"},
                _pick_use_noun,
            ),
            (
                "use_to_infinitive",
                frozenset({"use"}),
                lambda c: c.wl1 == "to"
                and len(c.left) >= 2
                and c.wl2 not in {"the", "a", "an"},
                _pick_use_verb,
            ),
            (
                "use_det_poss",
                frozenset({"use"}),
                lambda c: immediate_left_is_det_or_poss(c.left)
                or coarse_pos_tag(c.wl1) in ("DET", "POSS"),
                _pick_use_noun,
            ),
            (
                "use_prep_frame",
                frozenset({"use"}),
                lambda c: c.wl1
                in {
                    "for",
                    "of",
                    "in",
                    "any",
                    "no",
                    "much",
                    "little",
                    "public",
                    "commercial",
                    "personal",
                    "common",
                },
                _pick_use_noun,
            ),
            (
                "use_modal_lex",
                frozenset({"use"}),
                lambda c: c.wl1
                in {
                    "don't",
                    "doesn't",
                    "didn't",
                    "can",
                    "will",
                    "would",
                    "could",
                    "should",
                    "must",
                    "may",
                    "might",
                },
                _pick_use_verb,
            ),
            (
                "use_modal_pos",
                frozenset({"use"}),
                lambda c: coarse_pos_tag(c.wl1) == "MODAL",
                _pick_use_verb,
            ),
        ]
    )

    # content
    rules.extend(
        [
            (
                "content_with_or_copular_left",
                frozenset({"content"}),
                lambda c: c.wr1 == "with"
                or c.wl1
                in {
                    "feel",
                    "felt",
                    "feels",
                    "seems",
                    "seemed",
                    "look",
                    "looked",
                    "looks",
                    "grow",
                    "grew",
                    "grows",
                    "appear",
                    "appeared",
                    "appears",
                },
                _pick_content_adj,
            ),
            (
                "content_det_poss",
                frozenset({"content"}),
                lambda c: immediate_left_is_det_or_poss(c.left)
                or coarse_pos_tag(c.wl1) in ("DET", "POSS"),
                _pick_content_noun,
            ),
            (
                "content_of_any",
                frozenset({"content"}),
                lambda c: c.wl1
                in {
                    "of",
                    "any",
                    "its",
                    "their",
                    "his",
                    "her",
                    "your",
                    "our",
                    "for",
                    "on",
                    "in",
                    "no",
                },
                _pick_content_noun,
            ),
        ]
    )

    return rules


_ORDERED_DISAMBIG_RULES: list[RuleTuple] = _build_ordered_disambig_rules()


def disambiguate_heteronym_ipa(
    word_key: str,
    candidates: list[str],
    left: list[str],
    right: list[str],
    *,
    default_primary: str,
) -> str:
    """
    Choose one IPA string from *candidates* using *left* / *right* token context.

    *default_primary* is typically ``candidates[0]`` after homograph/TSV merge.
    """
    if len(candidates) < 2:
        return candidates[0] if candidates else ""
    ctx = NeighborContext.from_lists(word_key, left, right)
    hit = apply_ordered_rules(ctx, candidates, _ORDERED_DISAMBIG_RULES)
    if hit is not None:
        return hit
    return default_primary

