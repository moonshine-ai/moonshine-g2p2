"""
Minimal **closed-class** tagging for English (no ML).

Maps a lowercased surface token to a coarse label used by heteronym heuristics.
Open-class words return ``UNK``. Contractions that appear as single tokens
(``they'll``, ``don't``) are listed explicitly.

Labels: ``DET``, ``PRON``, ``POSS``, ``AUX``, ``MODAL``, ``TO``, ``PREP``,
``ADV``, ``CONJ``, ``SUBORD``, ``WH``, ``UNK``.
"""

from __future__ import annotations

# --- determiners & quantifiers (nominal premodifiers) --------------------------------

DETERMINERS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "each",
        "every",
        "either",
        "neither",
        "another",
        "such",
        "what",
        "which",
        "whatever",
        "whichever",
        "some",
        "any",
        "no",
        "all",
        "both",
        "half",
        "enough",
        "several",
        "many",
        "much",
        "few",
        "little",
        "more",
        "most",
        "less",
        "least",
        "other",
        "own",
        "same",
        "certain",
        "quite",
    }
)

# Possessive determiners (often pattern like DET for NP-internal heuristics)
POSSESSIVE_DETERMINERS: frozenset[str] = frozenset(
    {
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "whose",
    }
)

# --- pronouns --------------------------------------------------------------------

SUBJECT_PRONOUNS: frozenset[str] = frozenset(
    {
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "one",
        "someone",
        "somebody",
        "everyone",
        "everybody",
        "anyone",
        "anybody",
        "nobody",
        "nothing",
        "something",
        "everything",
        "who",
        "whoever",
    }
)

OBJECT_PRONOUNS: frozenset[str] = frozenset(
    {
        "me",
        "him",
        "us",
        "them",
        "whom",
        "whomever",
    }
)

# Standalone possessive pronouns
POSSESSIVE_PRONOUNS: frozenset[str] = frozenset(
    {
        "mine",
        "yours",
        "hers",
        "ours",
        "theirs",
    }
)

REFLEXIVE_PRONOUNS: frozenset[str] = frozenset(
    {
        "myself",
        "yourself",
        "himself",
        "herself",
        "itself",
        "ourselves",
        "themselves",
        "oneself",
    }
)

# --- auxiliaries & modals (including common contractions as single tokens) --------

BE_FORMS: frozenset[str] = frozenset(
    {
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "'m",
        "'re",
        "'s",
    }
)

HAVE_FORMS: frozenset[str] = frozenset(
    {
        "have",
        "has",
        "had",
        "'ve",
        "'d",
    }
)

DO_FORMS: frozenset[str] = frozenset(
    {
        "do",
        "does",
        "did",
        "done",
        "doing",
    }
)

MODALS: frozenset[str] = frozenset(
    {
        "will",
        "would",
        "wo",
        "shall",
        "should",
        "can",
        "could",
        "ca",
        "may",
        "might",
        "must",
        "ought",
    }
)

MODAL_AUX_CONTRACTIONS: frozenset[str] = frozenset(
    {
        "don't",
        "doesn't",
        "didn't",
        "won't",
        "wouldn't",
        "shouldn't",
        "couldn't",
        "can't",
        "cannot",
        "mightn't",
        "mustn't",
        "shan't",
        "ain't",
        "i'll",
        "you'll",
        "he'll",
        "she'll",
        "we'll",
        "they'll",
        "it'll",
        "there'll",
        "that'll",
        "i'd",
        "you'd",
        "he'd",
        "she'd",
        "we'd",
        "they'd",
        "i've",
        "you've",
        "we've",
        "they've",
        "who've",
        "what've",
        "where've",
        "how've",
    }
)

# --- function words --------------------------------------------------------------

PREPOSITIONS: frozenset[str] = frozenset(
    {
        "in",
        "on",
        "at",
        "by",
        "for",
        "from",
        "of",
        "to",
        "with",
        "without",
        "about",
        "into",
        "onto",
        "upon",
        "over",
        "under",
        "above",
        "below",
        "between",
        "among",
        "through",
        "during",
        "before",
        "after",
        "since",
        "until",
        "till",
        "within",
        "beyond",
        "across",
        "against",
        "along",
        "around",
        "behind",
        "beside",
        "besides",
        "despite",
        "except",
        "inside",
        "outside",
        "near",
        "toward",
        "towards",
        "via",
        "per",
        "like",
        "unlike",
        "concerning",
        "including",
        "regarding",
        "throughout",
        "upon",
    }
)

ADVERBS_DEGREE_FREQ: frozenset[str] = frozenset(
    {
        "very",
        "so",
        "too",
        "quite",
        "rather",
        "almost",
        "nearly",
        "just",
        "only",
        "even",
        "still",
        "already",
        "always",
        "never",
        "often",
        "sometimes",
        "usually",
        "rarely",
        "seldom",
        "ever",
        "not",
        "no",
        "here",
        "there",
        "now",
        "then",
        "today",
        "tomorrow",
        "yesterday",
        "ago",
        "soon",
        "later",
        "again",
        "once",
        "twice",
        "perhaps",
        "maybe",
        "however",
        "therefore",
        "thus",
        "also",
        "either",
        "neither",
    }
)

CONJUNCTIONS: frozenset[str] = frozenset(
    {
        "and",
        "or",
        "but",
        "nor",
        "yet",
        "for",
    }
)

SUBORDINATORS: frozenset[str] = frozenset(
    {
        "that",
        "if",
        "unless",
        "although",
        "though",
        "because",
        "while",
        "whereas",
        "whether",
        "since",
        "until",
        "till",
        "after",
        "before",
        "when",
        "whenever",
        "where",
        "wherever",
        "why",
        "how",
        "than",
        "as",
    }
)

WH_WORDS: frozenset[str] = frozenset(
    {
        "who",
        "whom",
        "whose",
        "what",
        "which",
        "where",
        "when",
        "why",
        "how",
    }
)

# Copular / linking verbs (often precede adjective "content", etc.)
COPULAR_VERBS: frozenset[str] = frozenset(
    {
        "seem",
        "seems",
        "seemed",
        "appear",
        "appears",
        "appeared",
        "look",
        "looks",
        "looked",
        "feel",
        "feels",
        "felt",
        "grow",
        "grows",
        "grew",
        "become",
        "becomes",
        "became",
        "remain",
        "remains",
        "stayed",
        "stay",
        "stays",
        "sound",
        "sounds",
        "sounded",
    }
)


def coarse_pos_tag(token: str) -> str:
    """
    Return a coarse POS label for *token* (lowercase ASCII word or contraction).

    Unknown / open-class tokens return ``UNK``.
    """
    w = token.lower().strip()
    if not w:
        return "UNK"
    if w in DETERMINERS:
        return "DET"
    if w in POSSESSIVE_DETERMINERS:
        return "POSS"
    if w in SUBJECT_PRONOUNS or w in OBJECT_PRONOUNS:
        return "PRON"
    if w in POSSESSIVE_PRONOUNS or w in REFLEXIVE_PRONOUNS:
        return "PRON"
    if w in MODALS or w in MODAL_AUX_CONTRACTIONS:
        return "MODAL"
    if w in BE_FORMS or w in HAVE_FORMS or w in DO_FORMS:
        return "AUX"
    if w == "to":
        return "TO"
    if w in PREPOSITIONS:
        return "PREP"
    if w in COPULAR_VERBS:
        return "COPULA"
    if w in ADVERBS_DEGREE_FREQ:
        return "ADV"
    if w in CONJUNCTIONS:
        return "CONJ"
    if w in SUBORDINATORS:
        return "SUBORD"
    if w in WH_WORDS:
        return "WH"
    return "UNK"


def immediate_left_is_det_or_poss(left: list[str]) -> bool:
    if not left:
        return False
    t = coarse_pos_tag(left[-1])
    return t in ("DET", "POSS")


def immediate_left_is_modal_aux_or_inf_to(left: list[str]) -> bool:
    if not left:
        return False
    t = coarse_pos_tag(left[-1])
    return t in ("MODAL", "AUX", "TO")


def immediate_left_is_pronoun(left: list[str]) -> bool:
    return bool(left) and coarse_pos_tag(left[-1]) == "PRON"


def immediate_left_is_prep(left: list[str]) -> bool:
    return bool(left) and coarse_pos_tag(left[-1]) == "PREP"


def have_has_had_in_left_window(left: list[str], *, window: int = 4) -> bool:
    tail = left[-window:] if left else []
    return any(w in HAVE_FORMS or w in {"have", "has", "had"} for w in tail)


def have_has_had_immediate_left(left: list[str]) -> bool:
    """True if the token immediately left is *have* / *has* / *had* / *'ve* (not *'d*)."""
    if not left:
        return False
    w = left[-1]
    return w in {"have", "has", "had", "'ve"}


def modal_or_aux_in_left_window(left: list[str], *, window: int = 4) -> bool:
    tail = left[-window:] if left else []
    for w in tail:
        if coarse_pos_tag(w) in ("MODAL", "AUX"):
            return True
    return False


def copular_in_left_window(left: list[str], *, window: int = 3) -> bool:
    tail = left[-window:] if left else []
    return any(w in COPULAR_VERBS for w in tail)


def temporal_past_hint_in_left_window(left: list[str], *, window: int = 6) -> bool:
    tail = left[-window:] if left else []
    pastish = {"yesterday", "ago", "last", "previously", "earlier", "once"}
    return any(w in pastish for w in tail)


def present_hint_in_left_window(left: list[str], *, window: int = 6) -> bool:
    tail = left[-window:] if left else []
    hints = {"will", "always", "often", "never", "still", "please", "ca", "today", "tomorrow", "now"}
    return any(w in hints for w in tail)


def left_token_suggests_infinitive_or_finite_verb_after(left: list[str]) -> bool:
    """
    Token immediately before a gap often filled by a verb: modal, ``to``, or *do*-family.
    Excludes *be*/*have* alone (``are live`` can be adjective).
    """
    if not left:
        return False
    lt = left[-1]
    if lt == "to":
        return True
    if lt in MODALS or lt in MODAL_AUX_CONTRACTIONS:
        return True
    if lt in DO_FORMS or lt in {"don't", "doesn't", "didn't"}:
        return True
    return False


def right_token_starts_prep_phrase(right: list[str]) -> bool:
    if not right:
        return False
    return coarse_pos_tag(right[0]) == "PREP"


def right_token_starts_det_or_poss(right: list[str]) -> bool:
    if not right:
        return False
    return coarse_pos_tag(right[0]) in ("DET", "POSS")
