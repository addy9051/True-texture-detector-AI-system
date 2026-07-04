"""Category -> likely materials prior, and mapping catalog materials to the
physics ontology fibers.

Two uses:
  1. When a listing does not state its fabric (recall: only ~39% of sampled
     products had a detectable material), likely_materials(category) supplies a
     prior so the concierge can probe the fabrics that category usually uses.
  2. normalize_to_ontology() maps catalog material names (which mix fibers like
     "cotton" with constructions like "denim" and weaves like "satin") to the
     fiber keys in fabric_physics.json, so category materials plug into the same
     grounded diagnosis. Fiber-ambiguous weaves (satin, crepe, net, jacquard...)
     map to nothing on purpose — the fiber can't be inferred from the weave.
"""

import json
import re
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "category_materials.json"

# Substrings -> ontology fiber key. First match per token wins; a blend like
# "poly-cotton" yields both polyester and cotton.
FIBER_KEYWORDS = {
    "cotton": "cotton", "khadi": "cotton", "chambray": "cotton",
    "seersucker": "cotton", "corduroy": "cotton", "denim": "cotton",
    "flannel": "cotton", "twill": "cotton", "terry": "cotton",
    "waffle": "cotton", "jersey": "cotton", "french terry": "cotton",
    "linen": "linen",
    "poly": "polyester", "microfiber": "polyester", "fleece": "polyester",
    "velour": "polyester", "nida": "polyester", "koshibo": "polyester",
    "ponte": "polyester", "shimmer": "polyester", "dri-fit": "polyester",
    "wool": "wool", "merino": "wool", "tweed": "wool", "angora": "wool",
    "cashmere": "cashmere",
    "viscose": "viscose_rayon", "rayon": "viscose_rayon", "modal": "viscose_rayon",
    "bamboo": "viscose_rayon", "lyocell": "viscose_rayon", "tencel": "viscose_rayon",
    "art silk": "viscose_rayon", "artificial silk": "viscose_rayon",
    "silk": "silk",  # matched after "art silk" so real silk stays silk
    "nylon": "nylon", "polyamide": "nylon", "lace": "nylon", "powernet": "nylon",
    "spandex": "spandex", "elastane": "spandex", "lycra": "spandex",
    "acrylic": "acrylic",
    "georgette": "georgette",
    "chiffon": "chiffon",
}

# Fiber-ambiguous weaves/finishes and non-fabrics: recorded but not fiber-mapped.
UNMAPPED_HINT = {"satin", "crepe", "net", "mesh", "jacquard", "brocade",
                 "organza", "taffeta", "chanderi", "tissue", "zari", "velvet",
                 "faux leather", "leather", "faux fur", "blended"}

_TOKEN = re.compile(r"[a-z]+(?: [a-z]+)?")


def _load() -> dict:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))["departments"]


def likely_materials(category: str, department: str | None = None) -> list[str]:
    """Return the catalog material prior for a category (case-insensitive).
    Searches the given department first, then all departments."""
    depts = _load()
    cat_lower = category.strip().lower()
    order = ([department] if department else []) + list(depts)
    for dept in order:
        for name, mats in depts.get(dept, {}).items():
            if name.lower() == cat_lower:
                return mats
    return []


def normalize_to_ontology(material: str) -> list[str]:
    """Map one catalog material string to ontology fiber key(s)."""
    text = material.lower()
    found: list[str] = []
    # Longest keywords first so "art silk" beats "silk", "french terry" beats "terry".
    for kw in sorted(FIBER_KEYWORDS, key=len, reverse=True):
        if kw in text and FIBER_KEYWORDS[kw] not in found:
            # avoid "silk" firing inside an already-consumed "art silk"
            if kw == "silk" and ("art silk" in text or "artificial silk" in text) \
                    and "silk" not in text.replace("art silk", "").replace("artificial silk", ""):
                continue
            found.append(FIBER_KEYWORDS[kw])
    return found


def all_materials() -> dict[str, int]:
    """Distinct catalog materials across every category, with frequency."""
    depts = _load()
    counts: dict[str, int] = {}
    for cats in depts.values():
        for mats in cats.values():
            for m in mats:
                counts[m] = counts.get(m, 0) + 1
    return counts
