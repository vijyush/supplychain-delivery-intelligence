"""Run the analytical SQL on the clean feature parquet via DuckDB. python sql/run_sql.py"""
import glob, duckdb
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
def main():
    pq = ROOT/"outputs"/"features.parquet"
    if not pq.exists():
        raise SystemExit("Run the pipeline first (python src/run_all.py) to build features.parquet")
    con = duckdb.connect()
    con.execute(f"CREATE VIEW orders AS SELECT * FROM read_parquet('{pq}')")
    for f in sorted(glob.glob(str(ROOT/"sql"/"0*.sql"))):
        print("\n"+"="*60+f"\n{Path(f).name}\n"+"="*60)
        print(con.execute(Path(f).read_text()).df().to_string(index=False))
if __name__ == "__main__":
    main()
