FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# libgomp1: runtime OpenMP que LightGBM carga en dlopen al importar (no trae wheel
# propia, la imagen slim no la incluye por defecto).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Capa de dependencias cacheada aparte del código: pyproject.toml/uv.lock cambian poco,
# esto evita reinstalar el stack de ML completo (catboost/xgboost/prophet/...) en cada
# build que solo toca api/config/src.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups --no-install-project

COPY api/ api/
COPY config/ config/
COPY src/ src/
COPY artifacts/models/ artifacts/models/

RUN uv sync --frozen --no-default-groups

# Venv directo en PATH: `uv run` en el CMD reintentaría sincronizar contra los grupos
# default (dev/modeling) en cada arranque -- sin g++ en esta imagen, statsforecast
# fallaría al compilar. El venv ya quedó armado en el build, no hace falta re-sync.
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
