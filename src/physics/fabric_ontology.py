"""Load fabric_physics.json and map product listings to material expectations.

This is a heuristic ontology, not lab physics: it encodes what a genuine fiber
should feel like, which adjectives signal the customer got something else, and
which cheap fiber it is most often substituted with. The diagnosis engine
(Phase 2) compares these expectations against the texture sentences extracted
by src/nlp/semantic_filter.py.
"""

import json
import re
from pathlib import Path

ONTOLOGY_PATH = Path(__file__).resolve().parents[2] / "data" / "fabric_physics.json"

# Listing detail keys that carry material claims (matched case-insensitively).
MATERIAL_KEYS = ("fabric type", "material", "fabric", "composition")


class FabricOntology:
    def __init__(self, path: Path = ONTOLOGY_PATH):
        self.materials: dict = json.loads(Path(path).read_text(encoding="utf-8"))["materials"]
        self.alias_map: dict[str, str] = {}
        for name, spec in self.materials.items():
            for alias in [name.replace("_", " ")] + spec.get("aliases", []):
                self.alias_map[alias.lower()] = name

    def materials_from_listing(self, product: dict) -> list[str]:
        """Detect claimed fibers from a product's structured details, title, features."""
        details = product.get("details") or {}
        texts = [str(v) for k, v in details.items()
                 if any(mk in k.lower() for mk in MATERIAL_KEYS)]
        texts.append(product.get("title") or "")
        texts.extend(product.get("features") or [])
        blob = " ".join(texts).lower()
        found = []
        for alias, canonical in self.alias_map.items():
            if canonical not in found and re.search(rf"\b{re.escape(alias)}\b", blob):
                found.append(canonical)
        return found

    def expectations(self, material: str) -> dict | None:
        return self.materials.get(material)

    def failing_hits(self, material: str, sentences: list[str]) -> list[tuple[str, str]]:
        """Naive Phase-1 mismatch check: which failing adjectives for `material`
        appear verbatim in the texture sentences. Phase 2 replaces this with
        embedding matching so 'felt like a garbage bag' can hit 'plastic'."""
        spec = self.expectations(material)
        if not spec:
            return []
        hits = []
        for adj in spec.get("failing_adjectives", []):
            pattern = re.compile(rf"\b{re.escape(adj)}\w*", re.IGNORECASE)
            for s in sentences:
                if pattern.search(s):
                    hits.append((adj, s))
                    break
        return hits
