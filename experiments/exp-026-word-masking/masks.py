"""exp-026: build masked captions from spaCy tokens. Pure functions, no model.

A document is a list of (text_with_ws, is_alpha, pos). Only alphabetic words can be masked, so
punctuation stays when words are removed. Random controls remove or keep the same number of words as
their class in the same caption.
"""

from __future__ import annotations

import numpy as np

FUNCTION = {"DET", "ADP", "PRON", "AUX", "CCONJ", "SCONJ", "PART"}  # as exp-024
NOUN = {"NOUN"}
# condition -> (caption set, tokens shuffled)
CONDITIONS = {
    "intact": ("intact", False),
    "shuffled": ("intact", True),
    "nofunc": ("nofunc", False),
    "nofunc_shuf": ("nofunc", True),
    "nouns": ("nouns", False),
    "nouns_shuf": ("nouns", True),
    "nonouns": ("nonouns", False),
    "ctrl_nofunc": ("ctrl_nofunc", False),
    "ctrl_nonouns": ("ctrl_nonouns", False),
    "ctrl_nouns": ("ctrl_nouns", False),
}
# Original captions are short: nouns-only would leave 2925 of 10240 empty
ORIG_SKIP = {"nouns", "nouns_shuf", "ctrl_nouns"}


def norm(text: str) -> str:
    return " ".join(text.split())


def remove_class(doc: list, tags: set[str]) -> str:
    return norm("".join(t for t, alpha, pos in doc if not (alpha and pos in tags)))


def only_class(doc: list, tags: set[str]) -> str:
    return " ".join(t.strip() for t, alpha, pos in doc if alpha and pos in tags)


def count_class(doc: list, tags: set[str]) -> int:
    return sum(1 for _, alpha, pos in doc if alpha and pos in tags)


def remove_random(doc: list, count: int, rng: np.random.Generator) -> str:
    """Remove `count` random words; punctuation stays, as in `remove_class`."""
    words = [i for i, (_, alpha, _) in enumerate(doc) if alpha]
    drop = set(rng.choice(words, size=min(count, len(words)), replace=False).tolist())
    return norm("".join(t for i, (t, _, _) in enumerate(doc) if i not in drop))


def keep_random(doc: list, count: int, rng: np.random.Generator) -> str:
    """Keep `count` random words in order, and nothing else, as in `only_class`."""
    words = [i for i, (_, alpha, _) in enumerate(doc) if alpha]
    keep = np.sort(rng.choice(words, size=min(count, len(words)), replace=False)).tolist()
    return " ".join(doc[i][0].strip() for i in keep)


def build(docs: list[list], raw: list[str], seed: int = 0) -> dict[str, list[str]]:
    """Every caption set. `intact` is the raw text, so it matches earlier runs token for token."""
    rng = np.random.default_rng(seed)
    out = {
        "intact": list(raw),
        "nofunc": [remove_class(d, FUNCTION) for d in docs],
        "nonouns": [remove_class(d, NOUN) for d in docs],
        "nouns": [only_class(d, NOUN) for d in docs],
        "ctrl_nofunc": [remove_random(d, count_class(d, FUNCTION), rng) for d in docs],
        "ctrl_nonouns": [remove_random(d, count_class(d, NOUN), rng) for d in docs],
        "ctrl_nouns": [keep_random(d, count_class(d, NOUN), rng) for d in docs],
    }
    return out


def conditions_for(caption: str) -> list[str]:
    return [c for c in CONDITIONS if caption != "orig" or c not in ORIG_SKIP]
