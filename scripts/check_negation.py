"""Fast, model-free sanity check for the negation gate.

Runs in <1s. Includes the two real false-positives Phase 1 produced, plus
constructions the gate must get right. Exit code 0 = all pass.

    uv run python scripts/check_negation.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.diagnosis.negation import is_negated

# (sentence, adjective, expected_is_negated)
CASES = [
    # The two real Phase-1 false positives — must now be suppressed (True).
    ("It was not scratchy or gross feeling.", "scratchy", True),
    ("Material is also stretchy and light weight without being flimsy.", "flimsy", True),
    # Genuine complaints — must NOT be negated (False).
    ("Material is super cheap and kind of shiny.", "shiny", False),
    ("The fabric is also not soft, it's almost scratchy.", "scratchy", False),
    ("Cheap looking flimsy material inside & out.", "flimsy", False),
    ("Material is uncomfortable and scratchy.", "scratchy", False),
    # Clause break must stop negation leaking across "but".
    ("It's not too big but the fabric is scratchy.", "scratchy", False),
    # Contraction form.
    ("This isn't scratchy at all.", "scratchy", True),
    # "not too heavy" — negated (the item is fine).
    ("The material is not too heavy to wear.", "heavy", True),
    # Plain positive statement with the adjective present, no negation.
    ("The material is heavy, as stated in other reviews.", "heavy", False),
]

_ADJ = lambda adj: re.compile(rf"\b{re.escape(adj)}\w*", re.IGNORECASE)


def main():
    failures = 0
    for sentence, adj, expected in CASES:
        m = _ADJ(adj).search(sentence)
        if not m:
            print(f"FAIL (adj not found): {adj!r} in {sentence!r}")
            failures += 1
            continue
        got = is_negated(sentence, m.start())
        ok = got == expected
        flag = "ok " if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"  [{flag}] negated={got!s:5} (want {expected!s:5}) | {adj}: {sentence}")
    print()
    if failures:
        print(f"{failures}/{len(CASES)} cases FAILED")
        sys.exit(1)
    print(f"All {len(CASES)} negation cases passed.")


if __name__ == "__main__":
    main()
