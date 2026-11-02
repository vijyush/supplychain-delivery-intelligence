"""make_charts.py — render real PNG charts into assets/ for the README."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .config import OUT, ASSETS
except ImportError:
    from config import OUT, ASSETS

plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.edgecolor": "#888888"})
INK, TEAL, RED, AMBER = "#1F3B57", "#2DB4A6", "#E2443B", "#F0A330"


def _save(fig, name):
    fig.tight_layout(); fig.savefig(ASSETS/name, bbox_inches="tight"); plt.close(fig)


def late_by_mode():
    d = pd.read_csv(OUT/"shipping_variance.csv").sort_values("late_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(d["Shipping Mode"], d["late_rate"]*100, color=RED)
    ax.set_ylabel("late-delivery rate (%)"); ax.set_title("Late-delivery rate by shipping mode")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, "late_by_mode.png")


def pr_roc():
    z = np.load(OUT/"pr_roc.npz")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].plot(z["rec"], z["prec"], color=TEAL, lw=2); ax[0].set_title("Precision-Recall")
    ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision")
    ax[1].plot(z["fpr"], z["tpr"], color=INK, lw=2); ax[1].plot([0, 1], [0, 1], "--", color="#aaa")
    ax[1].set_title("ROC"); ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR")
    _save(fig, "pr_roc.png")


def confusion():
    m = json.loads((OUT/"model_metrics.json").read_text())
    cm = np.array(m["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(3.9, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max()/2 else INK)
    ax.set_xticks([0, 1], ["pred on-time", "pred late"]); ax.set_yticks([0, 1], ["on-time", "late"])
    ax.set_title(f"Confusion @F1\nGBM AUC {m['roc_auc']} | NN AUC {m['nn_auc']}")
    _save(fig, "confusion_matrix.png")


def importance():
    d = pd.read_csv(OUT/"feature_importance.csv").head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.barh(d["feature"], d["importance"], color=INK)
    ax.set_title("What drives late delivery (gradient boosting)")
    _save(fig, "feature_importance.png")


def forecast():
    d = pd.read_csv(OUT/"forecast.csv")
    m = json.loads((OUT/"forecast_metrics.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(pd.to_datetime(d["month"]), d["actual"]/1e6, color=INK, lw=2, marker="o", label="actual")
    ax.plot(pd.to_datetime(d["month"]), d["forecast"]/1e6, color=TEAL, lw=2, ls="--", marker="s", label="forecast")
    ax.plot(pd.to_datetime(d["month"]), d["baseline"]/1e6, color=AMBER, lw=1.5, ls=":", label="seasonal-naive")
    ax.set_ylabel("sales (USD M)"); ax.legend()
    ax.set_title(f"Demand forecast backtest (WAPE {m['wape_model']*100:.1f}%)")
    _save(fig, "demand_forecast.png")


def profit_region():
    d = pd.read_csv(OUT/"profit_by_region.csv").sort_values("profit", ascending=False).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    colors = [RED if p < 0 else TEAL for p in d["profit"]]
    ax.barh(d["Order Region"], d["profit"]/1e3, color=colors)
    ax.set_xlabel("profit (USD k)"); ax.set_title("Profit by order region (red = loss)")
    _save(fig, "profit_by_region.png")


def shipping_variance():
    d = pd.read_csv(OUT/"shipping_variance.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(d)); w = 0.38
    ax.bar(x-w/2, d["avg_sched"], w, label="scheduled", color=INK)
    ax.bar(x+w/2, d["avg_real"], w, label="actual", color=TEAL)
    ax.set_xticks(x, d["Shipping Mode"], rotation=15); ax.set_ylabel("days"); ax.legend()
    ax.set_title("Scheduled vs actual shipping days")
    _save(fig, "shipping_variance.png")


def rfm():
    d = pd.read_csv(OUT/"rfm_scatter.csv")
    segs = d["segment"].unique()
    cmap = {s: c for s, c in zip(segs, [TEAL, INK, AMBER, RED, "#8b6dff"])}
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for s in segs:
        sub = d[d.segment == s]
        ax.scatter(sub["frequency"], sub["monetary"], s=16, alpha=0.6, label=s, color=cmap.get(s, INK))
    ax.set_xlabel("frequency (orders)"); ax.set_ylabel("monetary (sales)")
    ax.set_yscale("log"); ax.set_title("Customer RFM segments"); ax.legend(fontsize=8)
    _save(fig, "rfm_segments.png")


def main():
    for fn in [late_by_mode, pr_roc, confusion, importance, forecast,
               profit_region, shipping_variance, rfm]:
        try:
            fn(); print("chart:", fn.__name__)
        except Exception as e:
            print("skip", fn.__name__, "->", e)
    print("charts written to assets/")


if __name__ == "__main__":
    main()
