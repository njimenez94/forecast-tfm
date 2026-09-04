.PHONY: create-database process-data build-datasets train-dataset pipeline test-sets serve-api \
	docker-build docker-api docker-stop docker-logs testing-api

PY := uv run python
DOCKER_IMAGE := forecast-tfm-api
DOCKER_CONTAINER := forecast-tfm-api

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

serve-api:
	uv run uvicorn api.main:app --reload --port 8000

# Build de la imagen + (re)levanta el contenedor en un solo comando ("desplegar").
docker-api: docker-build
	docker rm -f $(DOCKER_CONTAINER) >/dev/null 2>&1 || true
	docker run -d --name $(DOCKER_CONTAINER) -p 8000:8000 $(DOCKER_IMAGE)
	@echo "API en http://localhost:8000 (ver logs: make docker-logs)"

docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-stop:
	docker rm -f $(DOCKER_CONTAINER)

docker-logs:
	docker logs -f $(DOCKER_CONTAINER)

# Prueba la API ya corriendo (make serve-api o make docker-api) con datos reales.
testing-api:
	$(PY) -m scripts.test_api $(ARGS)