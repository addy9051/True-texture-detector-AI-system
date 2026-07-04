"""Load fabric_physics.json and map product listings to material expectations.

This is a heuristic ontology, not lab physics. Two orthogonal axes:
  - materials (fibers): what a genuine fiber should feel like, the adjectives
    that signal the customer got something else, and its cheap substitution.
    Governs thermal/weather and substitution reasoning.
  - weaves (constructions): surface feel and structure independent of fiber
    (satin should be smooth+glossy; velvet plush; net sheer+stiff). A garment
    can carry both a fiber and a weave; the tactile experience combines them.

The diagnosis engine (Phase 2) and the concierge compare these expectations
against what the customer reports.
"""

import json
import re
from pathlib import Path

ONTOLOGY_PATH = Path(__file__).resolve().parents[2] / "data" / "fabric_physics.json"

# Listing detail keys that carry material claims (matched case-insensitively).
MATERIAL_KEYS = ("fabric type", "material", "fabric", "composition")


def _build_alias_map(entries: dict) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for name, spec in entries.items():
        if name.startswith("_"):  # skip _note metadata
            continue
        for alias in [name.replace("_", " ")] + (spec.get("aliases") or []):
            alias_map[alias.lower()] = name
    return alias_map


class FabricOntology:
    def __init__(self, path: Path = ONTOLOGY_PATH):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.materials: dict = data["materials"]
        self.weaves: dict = {k: v for k, v in (data.get("weaves") or {}).items()
                             if not k.startswith("_")}
        self.alias_map = _build_alias_map(self.materials)
        self.weave_alias_map = _build_alias_map(self.weaves)

    @staticmethod
    def _listing_blob(product: dict) -> str:
        details = product.get("details") or {}
        texts = [str(v) for k, v in details.items()
                 if any(mk in k.lower() for mk in MATERIAL_KEYS)]
        texts.append(product.get("title") or "")
        texts.extend(product.get("features") or [])
        return " ".join(texts).lower()

    def materials_from_listing(self, product: dict) -> list[str]:
        """Detect claimed fibers from a product's structured details, title, features."""
        blob = self._listing_blob(product)
        found = []
        for alias, canonical in self.alias_map.items():
            if canonical not in found and re.search(rf"\b{re.escape(alias)}\b", blob):
                found.append(canonical)
        return found

    def weaves_from_listing(self, product: dict) -> list[str]:
        """Detect weave/construction names (satin, velvet, denim...) from a listing."""
        blob = self._listing_blob(product)
        found = []
        for alias, canonical in self.weave_alias_map.items():
            if canonical not in found and re.search(rf"\b{re.escape(alias)}\b", blob):
                found.append(canonical)
        return found

    def expectations(self, material: str) -> dict | None:
        return self.materials.get(material)

    def weave_expectations(self, weave: str) -> dict | None:
        return self.weaves.get(weave)

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
