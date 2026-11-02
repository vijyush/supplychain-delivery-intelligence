"""
clean_validate.py — load DataCo supply-chain data, validate, engineer features.

Governance: PII columns (email, names, password, street) are dropped.
Leakage: to predict Late_delivery_risk AT ORDER TIME, post-shipment fields
(Days for shipping (real), Delivery Status, shipping date) are EXCLUDED from the
model — they're only known after delivery. The model uses pre-shipment features.
Writes outputs/data_quality_report.csv, outputs/features.parquet, data/sample_5k.csv.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

try:
    from .config import raw_csv, DATA, OUT, PII, SEED
except ImportError:
    from config import raw_csv, DATA, OUT, PII, SEED

# pre-shipment, leakage-free features for the late-delivery model
MODEL_FEATURES = ["sched_days", "quantity", "discount_rate", "product_price",
                  "sales", "order_month", "order_dow",
                  "Shipping Mode", "Market", "Order Region",
                  "Customer Segment", "Department Name"]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(raw_csv(), encoding="latin-1")
    df = df.drop(columns=[c for c in PII if c in df.columns], errors="ignore")
    df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
    df["ship_date"] = pd.to_datetime(df["shipping date (DateOrders)"], errors="coerce")
    return df


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df); checks = []
    def add(rule, fails):
        checks.append({"rule": rule, "failures": int(fails), "rows": n,
                       "pass_rate_%": round(100*(1-fails/n), 3)})
    add("Sales >= 0", (df["Sales"] < 0).sum())
    add("no null order_date", df["order_date"].isna().sum())
    add("Late_delivery_risk in {0,1}", (~df["Late_delivery_risk"].isin([0, 1])).sum())
    add("quantity > 0", (df["Order Item Quantity"] <= 0).sum())
    add("no PII columns remain", sum(c in df.columns for c in PII))
    add("scheduled days >= 0", (df["Days for shipment (scheduled)"] < 0).sum())
    rep = pd.DataFrame(checks); rep.to_csv(OUT/"data_quality_report.csv", index=False)
    return rep


def modeling_frame() -> pd.DataFrame:
    cache = OUT/"features.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = load_raw()
    out = pd.DataFrame()
    out["order_id"] = df["Order Id"]
    out["order_item_id"] = df["Order Item Id"]
    out["order_date"] = df["order_date"]
    out["ship_date"] = df["ship_date"]
    out["sched_days"] = df["Days for shipment (scheduled)"].astype(float)
    out["real_days"] = df["Days for shipping (real)"].astype(float)   # kept for analysis, NOT a feature
    out["quantity"] = df["Order Item Quantity"].astype(float)
    out["discount_rate"] = df["Order Item Discount Rate"].astype(float)
    out["product_price"] = df["Product Price"].astype(float)
    out["sales"] = df["Sales"].astype(float)
    out["profit"] = df["Order Profit Per Order"].astype(float)
    out["profit_ratio"] = df["Order Item Profit Ratio"].astype(float)
    out["order_month"] = df["order_date"].dt.month.astype("Int16")
    out["order_dow"] = df["order_date"].dt.dayofweek.astype("Int16")
    out["Shipping Mode"] = df["Shipping Mode"].astype(str)
    out["Market"] = df["Market"].astype(str)
    out["Order Region"] = df["Order Region"].astype(str)
    out["Customer Segment"] = df["Customer Segment"].astype(str)
    out["Department Name"] = df["Department Name"].astype(str)
    out["Category Name"] = df["Category Name"].astype(str)
    out["customer_id"] = df["Order Customer Id"]
    out["Late_delivery_risk"] = df["Late_delivery_risk"].astype(int)
    out["Delivery Status"] = df["Delivery Status"].astype(str)   # for analysis only
    out = out.dropna(subset=["order_date"]).reset_index(drop=True)
    out.to_parquet(cache, index=False)
    out.sample(5000, random_state=SEED).to_csv(DATA/"sample_5k.csv", index=False)
    return out


def main():
    df = load_raw()
    print(data_quality_report(df).to_string(index=False))
    m = modeling_frame()
    print(f"\nrows: {len(m):,} | late rate: {m['Late_delivery_risk'].mean():.3f} "
          f"| loss-making orders: {(m['profit']<0).mean():.3f}")
    print("model features (leakage-free):", MODEL_FEATURES)
    print("wrote data_quality_report.csv, features.parquet, data/sample_5k.csv")


if __name__ == "__main__":
    main()
