.PHONY: create-database process-data build-datasets train-dataset pipeline test-sets

PY := uv run python

create-database:
	$(PY) -m scripts.create_database

process-data:
	$(PY) -m scripts.process_data $(ARGS)

build-datasets:
	$(PY) -m scripts.build_datasets $(ARGS)

train-dataset:
	$(PY) -m scripts.train_dataset $(ARGS)

test-sets:
	$(PY) -m scripts.build_test_sets

pipeline: process-data build-datasets train-dataset