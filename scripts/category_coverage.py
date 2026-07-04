"""Report how well the fabric ontology covers the category material catalog.

Shows: category/department counts, the distinct catalog materials, how many map
to an ontology fiber, and the most common UNMAPPED materials (candidates to add
to fabric_physics.json or to leave as fiber-ambiguous weaves).

    uv run python scripts/category_coverage.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics.category_materials import all_materials, normalize_to_ontology, _load
from src.physics.fabric_ontology import FabricOntology


def main():
    depts = _load()
    ontology = FabricOntology()
    n_cat = sum(len(c) for c in depts.values())
    print(f"{len(depts)} departments, {n_cat} categories, "
          f"{sum(len(m) for c in depts.values() for m in c.values())} category-material entries")

    def weave_of(material: str) -> list[str]:
        return ontology.weaves_from_listing({"title": material})

    mats = all_materials()
    fiber_mapped, weave_mapped, unmapped = {}, {}, {}
    for m, freq in mats.items():
        if normalize_to_ontology(m):
            fiber_mapped[m] = freq
        elif weave_of(m):
            weave_mapped[m] = freq
        else:
            unmapped[m] = freq
    covered = len(fiber_mapped) + len(weave_mapped)
    print(f"\n{len(mats)} distinct catalog materials — {covered} grounded "
          f"({100*covered/len(mats):.0f}%):")
    print(f"  {len(fiber_mapped)} map to a fiber, {len(weave_mapped)} map to a weave, "
          f"{len(unmapped)} still ungrounded.")

    if weave_mapped:
        print("\nNow grounded via WEAVE ontology (were fiber-ambiguous before):")
        for m, freq in sorted(weave_mapped.items(), key=lambda x: -x[1])[:15]:
            print(f"  {freq:3}x  {m}")

    if unmapped:
        print("\nStill ungrounded (no fiber, no weave):")
        for m, freq in sorted(unmapped.items(), key=lambda x: -x[1]):
            print(f"  {freq:3}x  {m}")

    print("\nOntology fiber coverage across the catalog (how many categories use each):")
    from collections import Counter
    fiber_use = Counter()
    for cats in depts.values():
        for cat_mats in cats.values():
            fibers = set()
            for m in cat_mats:
                fibers.update(normalize_to_ontology(m))
            fiber_use.update(fibers)
    for fiber, n in fiber_use.most_common():
        print(f"  {fiber:16} used in {n} categories")


if __name__ == "__main__":
    main()
