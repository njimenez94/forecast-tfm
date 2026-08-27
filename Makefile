.PHONY: create-database process-data build-datasets train-datasets train-all test-sets

PY := uv run python

create-database:
	$(PY) -m scripts.create_database

process-data:
	$(PY) -m scripts.process_data $(ARGS)

build-datasets:
	$(PY) -m scripts.build_datasets $(ARGS)

train-datasets:
	$(PY) -m scripts.train_datasets $(ARGS)

train-all:
	$(MAKE) train-datasets ARGS="--levels 10,11,12"

test-sets:
	$(PY) -m scripts.build_test_sets
