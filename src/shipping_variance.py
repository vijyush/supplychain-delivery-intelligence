"""
shipping_variance.py — delivery performance: real vs scheduled shipping days.
On-time rate and average day-variance by shipping mode and market.
Writes shipping_variance.csv and shipping_summary.json.
"""
from __future__ import annotations
import json
import pandas as pd
try:
    from .config import OUT
    from .clean_validate import modeling_frame
except ImportError:
    from config import OUT
    from clean_validate import modeling_frame

def run():
    df = modeling_frame().copy()
    df["day_variance"] = df["real_days"] - df["sched_days"]
    df["on_time"] = (df["Delivery Status"].isin(["Advance shipping", "Shipping on time"])).astype(int)
    by_mode = df.groupby("Shipping Mode").agg(
        avg_sched=("sched_days", "mean"), avg_real=("real_days", "mean"),
        avg_variance=("day_variance", "mean"), on_time_rate=("on_time", "mean"),
        late_rate=("Late_delivery_risk", "mean")).reset_index().round(3)
    by_mode.to_csv(OUT/"shipping_variance.csv", index=False)
    summary = {"overall_on_time_rate": round(float(df["on_time"].mean()), 4),
               "overall_avg_variance_days": round(float(df["day_variance"].mean()), 3),
               "worst_mode": by_mode.sort_values("late_rate", ascending=False).iloc[0]["Shipping Mode"]}
    (OUT/"shipping_summary.json").write_text(json.dumps(summary, indent=2))
    return summary, by_mode

def main():
    s, by_mode = run()
    print(f"overall on-time rate: {s['overall_on_time_rate']*100:.1f}% | "
          f"avg variance: {s['overall_avg_variance_days']} days | worst mode: {s['worst_mode']}")
    print(by_mode.to_string(index=False))
    print("wrote shipping_variance.csv, shipping_summary.json")

if __name__ == "__main__":
    main()
