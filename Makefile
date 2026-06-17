.PHONY: create_database create_datasets train-datasets pipeline clean clean-datasets

PY := uv run python

create_database:
	$(PY) -m scripts.create_database

create_datasets:
	$(PY) -m scripts.create_datasets $(ARGS)

train-datasets:
	$(PY) -m scripts.train_datasets $(ARGS)

pipeline: clean-datasets create_datasets train-datasets

clean-datasets:
	rm -f data/processed/dataset_level_*.parquet

clean:
	rm -rf artifacts/

%:
	@:
