"""
profit_costtoserve.py — where the money is made and lost (DataCo profit).
Margins by region / segment / category, and the share of loss-making orders.
Writes profit_by_region.csv, profit_by_segment.csv, profit_by_category.csv,
and profit_summary.json.
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


def _agg(df, by):
    g = df.groupby(by).agg(sales=("sales", "sum"), profit=("profit", "sum"),
                           orders=("order_id", "nunique")).reset_index()
    g["margin_pct"] = (100 * g["profit"] / g["sales"]).round(2)
    g["loss_order_rate"] = df.groupby(by)["profit"].apply(lambda s: (s < 0).mean()).values.round(4)
    return g.sort_values("profit", ascending=False)


def run():
    df = modeling_frame()
    by_region = _agg(df, "Order Region")
    by_segment = _agg(df, "Customer Segment")
    by_category = _agg(df, "Category Name")
    by_region.to_csv(OUT/"profit_by_region.csv", index=False)
    by_segment.to_csv(OUT/"profit_by_segment.csv", index=False)
    by_category.to_csv(OUT/"profit_by_category.csv", index=False)

    summary = {
        "total_sales": round(float(df["sales"].sum()), 0),
        "total_profit": round(float(df["profit"].sum()), 0),
        "overall_margin_pct": round(100*float(df["profit"].sum()/df["sales"].sum()), 2),
        "loss_order_rate": round(float((df["profit"] < 0).mean()), 4),
        "loss_value": round(float(df.loc[df["profit"] < 0, "profit"].sum()), 0),
        "worst_categories": by_category.nsmallest(5, "profit")[["Category Name", "profit"]].to_dict("records"),
    }
    (OUT/"profit_summary.json").write_text(json.dumps(summary, indent=2))
    return summary, by_segment


def main():
    s, by_segment = run()
    print(f"sales {s['total_sales']:,.0f} | profit {s['total_profit']:,.0f} "
          f"| margin {s['overall_margin_pct']}%")
    print(f"loss-making orders: {s['loss_order_rate']*100:.1f}% "
          f"(bleeding {s['loss_value']:,.0f})")
    print("profit by segment:\n", by_segment.to_string(index=False))
    print("wrote profit_by_region/segment/category.csv, profit_summary.json")


if __name__ == "__main__":
    main()
