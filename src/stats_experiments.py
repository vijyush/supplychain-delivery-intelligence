"""
stats_experiments.py — statistics on DataCo data.
  * chi-square: Shipping Mode vs late delivery (+ Cramer's V)
  * late rate by Market (proportions)
  * profit across Customer Segments (Kruskal-Wallis)
  * discount rate vs profit ratio (Spearman correlation)
Writes outputs/experiment_results.json.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy import stats

try:
    from .config import OUT
    from .clean_validate import modeling_frame
except ImportError:
    from config import OUT
    from clean_validate import modeling_frame


def run():
    df = modeling_frame()

    tab = pd.crosstab(df["Shipping Mode"], df["Late_delivery_risk"])
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    v = np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))

    by_mode = (df.groupby("Shipping Mode")["Late_delivery_risk"].mean()
               .sort_values(ascending=False).round(4).to_dict())
    by_market = (df.groupby("Market")["Late_delivery_risk"].mean()
                 .sort_values(ascending=False).round(4).to_dict())

    groups = [g["profit"].values for _, g in df.groupby("Customer Segment")]
    h, ph = stats.kruskal(*groups)

    sub = df[["discount_rate", "profit_ratio"]].dropna()
    rho, pr = stats.spearmanr(sub["discount_rate"], sub["profit_ratio"])

    results = {
        "shipping_mode_vs_late": {"chi2": float(chi2), "p_value": float(p),
                                  "cramers_v": round(float(v), 4), "late_by_mode": by_mode},
        "late_by_market": by_market,
        "profit_by_segment_kruskal": {"H": float(h), "p_value": float(ph)},
        "discount_vs_profit_spearman": {"rho": round(float(rho), 4), "p_value": float(pr)},
    }
    (OUT/"experiment_results.json").write_text(json.dumps(results, indent=2))
    return results


def main():
    r = run()
    s = r["shipping_mode_vs_late"]
    print(f"Shipping mode vs late: chi2 p={s['p_value']:.2e}, Cramer's V={s['cramers_v']}")
    print("late rate by mode:", s["late_by_mode"])
    print("late rate by market:", r["late_by_market"])
    print(f"profit across segments (Kruskal): p={r['profit_by_segment_kruskal']['p_value']:.2e}")
    print(f"discount vs profit (Spearman rho): {r['discount_vs_profit_spearman']['rho']}")
    print("wrote outputs/experiment_results.json")


if __name__ == "__main__":
    main()
