.PHONY: create_database process-data build-datasets train-datasets pipeline clean clean-datasets

PY := uv run python

create_database:
	$(PY) -m scripts.create_database

process-data:
	$(PY) -m scripts.process_data $(ARGS)

build-datasets:
	$(PY) -m scripts.build_datasets $(ARGS)

train-datasets:
	$(PY) -m scripts.train_datasets $(ARGS)

pipeline: clean-datasets process-data build-datasets train-datasets

clean-datasets:
	rm -f data/processed/level_*.parquet
	rm -f artifacts/datasets/dataset_level_*.parquet

clean:
	rm -rf artifacts/

%:
	@:
