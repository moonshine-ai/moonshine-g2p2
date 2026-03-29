"""
Ordered **decision-list** rules for heteronym disambiguation.

Each rule is ``(rule_id, homograph_keys, match(ctx), pick(candidates))``.
The first matching rule wins. **Exceptions** and long **multi-token** patterns
are registered before broad defaults (same homograph key).

:class:`NeighborContext` exposes ``wl1``..``wl3``, ``wr1``..``wr3``,
``endswith_left``, ``startswith_right``, and contiguous subsequences on the left.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

# Type aliases
PickFn = Callable[[list[str]], str | None]
MatchFn = Callable[["NeighborContext"], bool]
RuleTuple = tuple[str, frozenset[str], MatchFn, PickFn]


@dataclass(frozen=True)
class NeighborContext:
    """
    *left* is oldest→newest; ``wl1 == left[-1]`` is the token immediately
    before the homograph span. *right* is the token sequence after the span.
    """

    key: str
    left: tuple[str, ...]
    right: tuple[str, ...]

    @classmethod
    def from_lists(cls, key: str, left: list[str], right: list[str]) -> NeighborContext:
        return cls(key=key.lower(), left=tuple(left), right=tuple(right))

    @property
    def wl1(self) -> str:
        return self.left[-1] if self.left else ""

    @property
    def wl2(self) -> str:
        return self.left[-2] if len(self.left) >= 2 else ""

    @property
    def wl3(self) -> str:
        return self.left[-3] if len(self.left) >= 3 else ""

    @property
    def wr1(self) -> str:
        return self.right[0] if self.right else ""

    @property
    def wr2(self) -> str:
        return self.right[1] if len(self.right) >= 2 else ""

    @property
    def wr3(self) -> str:
        return self.right[2] if len(self.right) >= 3 else ""

    def endswith_left(self, *suffix: str) -> bool:
        """True if *left* ends with the token sequence *suffix* (``wl1`` last)."""
        n = len(suffix)
        if n == 0 or n > len(self.left):
            return False
        return self.left[-n:] == suffix

    def startswith_right(self, *prefix: str) -> bool:
        """True if *right* begins with *prefix*."""
        n = len(prefix)
        if n == 0 or n > len(self.right):
            return False
        return self.right[:n] == prefix

    def left_tail_contains_any(self, words: Iterable[str], *, window: int = 8) -> bool:
        tail = self.left[-window:] if self.left else ()
        s = frozenset(words)
        return any(w in s for w in tail)

    def left_has_subsequence(self, *seq: str) -> bool:
        """Contiguous *seq* appears anywhere in *left* (scan for longest matches)."""
        if not seq or not self.left:
            return False
        n, m = len(self.left), len(seq)
        if m > n:
            return False
        for i in range(0, n - m + 1):
            if self.left[i : i + m] == seq:
                return True
        return False


def apply_ordered_rules(
    ctx: NeighborContext,
    candidates: list[str],
    rules: list[RuleTuple],
) -> str | None:
    """Return chosen IPA or ``None`` if no rule matched with a non-empty pick."""
    for _rid, keys, match, pick in rules:
        if ctx.key not in keys:
            continue
        if match(ctx):
            p = pick(candidates)
            if p:
                return p
    return None
