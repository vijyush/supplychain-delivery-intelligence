.PHONY: install pipeline sql all clean
install:
	pip install -r requirements.txt
pipeline:
	python src/run_all.py
sql:
	python sql/run_sql.py
all: pipeline sql
clean:
	rm -f outputs/*.parquet outputs/*.csv outputs/*.json outputs/*.npz \
	      outputs/powerbi/*.csv assets/*.png excel/*.xlsx data/sample_5k.csv
