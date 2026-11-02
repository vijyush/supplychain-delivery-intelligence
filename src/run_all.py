"""run_all.py — full supply-chain pipeline. Requires DataCo CSV in data/."""
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
STEPS = [
    ("Data quality & features (leakage-free)", "src/clean_validate.py"),
    ("Statistics", "src/stats_experiments.py"),
    ("Late-delivery model (GBM + neural net)", "src/late_delivery_model.py"),
    ("Demand forecast (Holt-Winters)", "src/demand_forecast.py"),
    ("Profit & cost-to-serve", "src/profit_costtoserve.py"),
    ("Shipping variance", "src/shipping_variance.py"),
    ("Customer segmentation (RFM)", "src/segmentation.py"),
    ("Export for Power BI + Excel", "src/export_for_powerbi.py"),
    ("Charts for README", "src/make_charts.py"),
]
def main():
    for label, script in STEPS:
        print("\n"+"#"*72+f"\n# {label}\n"+"#"*72)
        if subprocess.run([sys.executable, str(ROOT/script)], cwd=ROOT).returncode != 0:
            print("step failed:", script); sys.exit(1)
    print("\nPipeline complete. See outputs/, assets/, excel/.")
if __name__ == "__main__":
    main()
