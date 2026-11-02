"""Shared paths + constants."""
from pathlib import Path
import glob
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; OUT = ROOT / "outputs"; PBI = OUT / "powerbi"
EXCEL = ROOT / "excel"; ASSETS = ROOT / "assets"
for d in (OUT, PBI, EXCEL, ASSETS): d.mkdir(parents=True, exist_ok=True)
def raw_csv() -> Path:
    c = [x for x in glob.glob(str(DATA/'*.csv')) if 'sample' not in x.lower() and 'Description' not in x]
    if not c: raise FileNotFoundError("DataCoSupplyChainDataset.csv not found in data/")
    return Path(c[0])
SEED = 42
PII = ["Customer Email","Customer Fname","Customer Lname","Customer Password",
       "Customer Street","Product Description","Product Image"]
