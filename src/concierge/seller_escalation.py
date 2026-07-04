"""Seller-side escalation for Case D (product is fine, but customers still
return it).

Per session, a NO_ISSUE return triggers no seller action — one customer not
loving a sound product isn't a supplier problem. But if such returns for the
same product cross a threshold, that IS a signal (styling, expectation-setting,
sizing perception, photography), and the seller agent should consult the
distributor on how to reduce returns.

This runs over the accumulated seller_insights rows — it's an aggregate view,
not a per-session decision.
"""

from collections import defaultdict

from src.concierge.concierge import CASE_NO_ISSUE

DEFAULT_THRESHOLD = 3


def escalations(insights: list[dict], threshold: int = DEFAULT_THRESHOLD) -> dict[str, dict]:
    """Return {parent_asin: escalation} for products whose NO_ISSUE returns
    reach `threshold`. Each escalation carries the count and a recommendation."""
    no_issue = defaultdict(list)
    for row in insights:
        dx = row.get("diagnosis") or {}
        if dx.get("case_class") == CASE_NO_ISSUE:
            no_issue[row.get("parent_asin")].append(row)

    out = {}
    for asin, rows in no_issue.items():
        if len(rows) >= threshold:
            title = rows[-1].get("title", "")
            out[asin] = {
                "parent_asin": asin,
                "title": title,
                "no_issue_returns": len(rows),
                "threshold": threshold,
                "recommendation": (
                    f"{len(rows)} customers returned this product despite no fabric "
                    f"or weather-fit fault found. This points to expectation or "
                    f"presentation gaps (styling, fit perception, photography, "
                    f"description), not a material defect. Consult the distributor "
                    f"on product/listing improvements to reduce returns."),
            }
    return out
