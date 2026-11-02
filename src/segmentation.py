"""
segmentation.py — customer RFM segmentation (KMeans) on DataCo orders.
Recency, Frequency, Monetary per customer -> 4 clusters with named profiles.
Writes customer_segments.csv (profiles) and rfm_scatter.csv (2D sample).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
try:
    from .config import OUT, SEED
    from .clean_validate import modeling_frame
except ImportError:
    from config import OUT, SEED
    from clean_validate import modeling_frame

def run():
    df = modeling_frame()
    snapshot = df["order_date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_id").agg(
        recency=("order_date", lambda s: (snapshot - s.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("sales", "sum")).reset_index()
    X = StandardScaler().fit_transform(
        np.log1p(rfm[["recency", "frequency", "monetary"]].clip(lower=0)))
    rfm["cluster"] = KMeans(n_clusters=4, random_state=SEED, n_init=10).fit_predict(X)
    prof = rfm.groupby("cluster").agg(
        customers=("customer_id", "count"), recency=("recency", "mean"),
        frequency=("frequency", "mean"), monetary=("monetary", "mean")).reset_index().round(1)
    # name by monetary rank
    # Name clusters from their RFM profile (recency, spend), not spend rank alone:
    # highest spend -> High-value; of the rest, longest since last order -> Lapsed;
    # then higher spend -> Recent buyers, lowest spend -> New / low-value.
    top = prof.sort_values("monetary", ascending=False).iloc[0]["cluster"]
    rest = prof[prof["cluster"] != top]
    lapsed = rest.sort_values("recency", ascending=False).iloc[0]["cluster"]
    rest2 = rest[rest["cluster"] != lapsed].sort_values("monetary", ascending=False)
    names = {top: "High-value", lapsed: "Lapsed",
             rest2.iloc[0]["cluster"]: "Recent buyers",
             rest2.iloc[1]["cluster"]: "New / low-value"}
    prof["segment"] = prof["cluster"].map(names)
    prof.to_csv(OUT/"customer_segments.csv", index=False)
    samp = rfm.sample(min(800, len(rfm)), random_state=SEED).copy()
    samp["segment"] = samp["cluster"].map(names)
    samp[["frequency", "monetary", "segment"]].to_csv(OUT/"rfm_scatter.csv", index=False)
    return prof

def main():
    prof = run()
    print("Customer RFM segments:")
    print(prof[["segment", "customers", "recency", "frequency", "monetary"]].to_string(index=False))
    print("wrote customer_segments.csv, rfm_scatter.csv")

if __name__ == "__main__":
    main()
