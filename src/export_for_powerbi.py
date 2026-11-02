"""
export_for_powerbi.py — consumption layer for Power BI / Tableau.
Writes fact_orders_sample.csv + aggregate CSVs + dim_date.csv to outputs/powerbi/,
and an Excel late-risk review queue to excel/.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

try:
    from .config import OUT, PBI, EXCEL, SEED
    from .clean_validate import modeling_frame
except ImportError:
    from config import OUT, PBI, EXCEL, SEED
    from clean_validate import modeling_frame


def build():
    df = modeling_frame()
    scores = pd.read_parquet(OUT/"late_scores.parquet") if (OUT/"late_scores.parquet").exists() else None

    fact = df[["order_id", "order_item_id", "order_date", "sales", "profit",
               "quantity", "sched_days", "real_days", "Shipping Mode", "Market",
               "Order Region", "Customer Segment", "Department Name", "Category Name",
               "Late_delivery_risk", "Delivery Status"]].copy()
    if scores is not None:
        fact = fact.merge(scores[["order_item_id", "late_prob"]], on="order_item_id", how="left")

    late = fact[fact.Late_delivery_risk == 1]
    rest = fact[fact.Late_delivery_risk == 0].sample(min(80000, (fact.Late_delivery_risk == 0).sum()),
                                                     random_state=SEED)
    sample = pd.concat([late.sample(min(80000, len(late)), random_state=SEED), rest]).sample(frac=1, random_state=SEED)
    sample.to_csv(PBI/"fact_orders_sample.csv", index=False)

    for src in ["profit_by_region.csv", "profit_by_segment.csv", "profit_by_category.csv",
                "shipping_variance.csv", "forecast.csv", "customer_segments.csv"]:
        p = OUT/src
        if p.exists():
            pd.read_csv(p).to_csv(PBI/src, index=False)

    dd = pd.DataFrame({"date": pd.to_datetime(sample["order_date"]).dt.date.unique()})
    dd["date"] = pd.to_datetime(dd["date"]); dd["year"] = dd["date"].dt.year
    dd["month"] = dd["date"].dt.to_period("M").astype(str)
    dd.to_csv(PBI/"dim_date.csv", index=False)

    _excel(sample)
    return sample


def _excel(sample):
    if "late_prob" not in sample.columns:
        return
    q = sample.sort_values("late_prob", ascending=False).head(500)[
        ["order_id", "Shipping Mode", "Market", "Customer Segment", "sales",
         "late_prob", "Late_delivery_risk"]]
    wb = Workbook(); head = Font(bold=True, color="FFFFFF"); fill = PatternFill("solid", fgColor="1F3B57")
    ws = wb.active; ws.title = "Summary"
    ws["A1"] = "Late-delivery review queue"; ws["A1"].font = Font(bold=True, size=14)
    rows = [("Metric", "Value"), ("Orders (sample)", len(sample)),
            ("Actual late in sample", int(sample["Late_delivery_risk"].sum())),
            ("Queue size", len(q)),
            ("Late caught in top 500", int(q["Late_delivery_risk"].sum())),
            ("Precision in top 500", round(float(q["Late_delivery_risk"].mean()), 3))]
    for i, (a, b) in enumerate(rows, start=3):
        ws[f"A{i}"], ws[f"B{i}"] = a, b
        if i == 3:
            for c in ("A", "B"): ws[f"{c}{i}"].font = head; ws[f"{c}{i}"].fill = fill
    ws.column_dimensions["A"].width = 24; ws.column_dimensions["B"].width = 16
    ws2 = wb.create_sheet("Review Queue")
    cols = list(q.columns)
    for j, c in enumerate(cols, start=1):
        cell = ws2.cell(row=1, column=j, value=c); cell.font = head; cell.fill = fill
        cell.alignment = Alignment(horizontal="center")
    for i, (_, r) in enumerate(q.iterrows(), start=2):
        for j, c in enumerate(cols, start=1):
            v = r[c]
            ws2.cell(row=i, column=j, value=(None if pd.isna(v) else
                     (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v)))
    for j, c in enumerate(cols, start=1):
        ws2.column_dimensions[get_column_letter(j)].width = max(12, len(c)+3)
    ws2.freeze_panes = "A2"
    wb.save(EXCEL/"late_risk_review_queue.xlsx")


def main():
    s = build()
    print(f"Power BI tables -> {PBI}/ ; fact sample rows: {len(s):,}")
    print("Excel: excel/late_risk_review_queue.xlsx")


if __name__ == "__main__":
    main()
