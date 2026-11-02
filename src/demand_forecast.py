"""
demand_forecast.py — monthly sales forecasting (Holt-Winters seasonal).

Honest modelling note: DataCo demand is near-stationary and the usable series is
short (~34 clean months after excluding the truncated tail past 2017-10). On a
series like this a seasonal model is the right tool; a neural net was tried and
did NOT beat it (too few points, flat signal) -- so the neural network is placed
on the 180k-row late-delivery task instead, where the data supports it.
This module forecasts with Holt-Winters and benchmarks against seasonal-naive.
Writes outputs/forecast.csv and outputs/forecast_metrics.json.
"""
from __future__ import annotations
import json
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
try:
    from .config import OUT
    from .clean_validate import modeling_frame
except ImportError:
    from config import OUT
    from clean_validate import modeling_frame

CUTOFF = "2017-10-31"
H = 6

def _series():
    df = modeling_frame()
    s = df.set_index("order_date")["sales"].resample("MS").sum()
    return s[s.index <= CUTOFF]

def run():
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    s = _series()
    train, test = s.iloc[:-H], s.iloc[-H:]
    model = ExponentialSmoothing(train, trend="add", seasonal="add",
                                 seasonal_periods=12).fit()
    fc = model.forecast(H)
    wape = float(np.sum(np.abs(test.values - fc.values)) / np.sum(np.abs(test.values)))
    naive = s.shift(12).iloc[-H:]
    wape_naive = float(np.sum(np.abs(test.values - naive.values)) / np.sum(np.abs(test.values)))

    # forward forecast (next 6 months)
    full = ExponentialSmoothing(s, trend="add", seasonal="add", seasonal_periods=12).fit()
    fwd = full.forecast(H)

    bt = pd.DataFrame({"month": test.index, "actual": test.values,
                       "forecast": fc.values, "baseline": naive.values})
    bt.to_csv(OUT/"forecast.csv", index=False)
    pd.DataFrame({"month": fwd.index, "forecast": fwd.values}).to_csv(
        OUT/"forecast_forward.csv", index=False)
    metrics = {"model": "Holt-Winters (add trend/season)", "months_used": int(len(s)),
               "excluded_tail_after": CUTOFF, "backtest_months": H,
               "wape_model": round(wape, 4), "wape_seasonal_naive": round(wape_naive, 4),
               "improvement_vs_naive_pct": round(100*(wape_naive-wape)/wape_naive, 1) if wape_naive else None}
    (OUT/"forecast_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics

def main():
    m = run()
    print(f"{m['model']} | {m['months_used']} months (tail after {m['excluded_tail_after']} excluded)")
    print(f"backtest WAPE: model {m['wape_model']} vs seasonal-naive {m['wape_seasonal_naive']} "
          f"({m['improvement_vs_naive_pct']}% vs naive)")
    print("wrote outputs/forecast.csv, forecast_forward.csv, forecast_metrics.json")

if __name__ == "__main__":
    main()
