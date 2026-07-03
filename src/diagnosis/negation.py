"""Lightweight negation-scope detector.

Phase 1 counted "It was NOT scratchy" as mismatch evidence because it matched
the adjective by substring with no regard for negation. This module gates each
adjective match: given the character offset of a matched adjective, it scans a
short window of preceding tokens for a negation cue, stopping at a clause break
so a negation in an earlier clause doesn't leak across ("not big BUT scratchy").

Deliberately rule-based, not a parser: it is fast, dependency-free, fully
explainable, and easy to unit-test (see scripts/check_negation.py). It won't
catch every construction, but it removes the dominant false-positive pattern.
"""

import re

# Words that flip the polarity of a following adjective.
NEGATION_CUES = {
    "not", "no", "never", "without", "hardly", "barely", "scarcely",
    "isnt", "wasnt", "arent", "werent", "dont", "doesnt", "didnt",
    "cant", "couldnt", "wouldnt", "wont", "aint", "nor", "neither", "non",
}

# Tokens that end the current clause — a negation before one of these does not
# reach an adjective after it.
CLAUSE_BREAKS = {"but", "however", "although", "though", "yet", "except", "still"}

_WORD = re.compile(r"[a-z]+")


def is_negated(sentence: str, match_start: int, window: int = 4) -> bool:
    """Return True if the adjective at `match_start` sits in a negated scope.

    Looks back up to `window` word-tokens. Contractions like "isn't" are
    normalised to "... not" first so the cue is visible as a token.
    """
    prefix = sentence[:match_start].lower().replace("n't", " not")
    tokens = _WORD.findall(prefix)
    for tok in reversed(tokens[-window:]):
        if tok in CLAUSE_BREAKS:
            return False
        if tok in NEGATION_CUES:
            return True
    return False
