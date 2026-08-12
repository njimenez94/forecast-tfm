.PHONY: create-database process-data build-datasets train-datasets pipe

PY := uv run python

create-database:
	$(PY) -m scripts.create_database

process-data:
	$(PY) -m scripts.process_data $(ARGS)

build-datasets:
	$(PY) -m scripts.build_datasets $(ARGS)

train-datasets:
	$(PY) -m scripts.train_datasets $(ARGS)

data-pipe:
	make create-database process-data build-datasets
