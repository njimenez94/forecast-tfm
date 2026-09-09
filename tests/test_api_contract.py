"""Contrato de la API (api/main.py::_build_feature_row y api/registry.py): fija en
un test el comportamiento que hoy solo está garantizado por el código en runtime
-- columnas exactas del payload, reconstrucción de categóricas contra el dominio
entrenado, y resolución de versiones -- para que un refactor de esas funciones no
rompa el contrato con quien llame a /predict sin que nada lo avise.

Se prueba _build_feature_row() directamente (función pura), sin levantar un
TestClient/servidor: evita sumar httpx como dependencia nueva solo para esto (ver
plan de reproducibilidad, Fase 4) -- si más adelante se quiere un test end-to-end
de /predict, ahí sí se justifica.
"""
import json

import pandas as pd
import pytest
from fastapi import HTTPException

import api.registry as registry
from api.main import _build_feature_row


def _artifact(**overrides) -> dict:
    base = {
        "features": ["store_id", "avg_sell_price", "is_weekend"],
        "categorical_features": ["store_id"],
        "categorical_categories": {"store_id": ["CA_1", "CA_2", "CA_3"]},
    }
    base.update(overrides)
    return base


def test_valid_payload_builds_dataframe_with_trained_column_order():
    artifact = _artifact()
    payload = {"avg_sell_price": 9.5, "is_weekend": 0, "store_id": "CA_2"}
    X = _build_feature_row(payload, artifact)
    assert list(X.columns) == artifact["features"]
    assert X["store_id"].iloc[0] == "CA_2"
    assert list(X["store_id"].cat.categories) == ["CA_1", "CA_2", "CA_3"]


def test_missing_column_is_rejected_with_400_and_lists_it():
    artifact = _artifact()
    payload = {"avg_sell_price": 9.5, "is_weekend": 0}  # falta store_id
    with pytest.raises(HTTPException) as exc:
        _build_feature_row(payload, artifact)
    assert exc.value.status_code == 400
    assert exc.value.detail["faltantes"] == ["store_id"]
    assert exc.value.detail["sobrantes"] == []


def test_extra_column_is_rejected_with_400_and_lists_it():
    artifact = _artifact()
    payload = {"avg_sell_price": 9.5, "is_weekend": 0, "store_id": "CA_2", "extra_col": 1}
    with pytest.raises(HTTPException) as exc:
        _build_feature_row(payload, artifact)
    assert exc.value.status_code == 400
    assert exc.value.detail["sobrantes"] == ["extra_col"]


def test_invalid_numeric_value_is_rejected_with_400():
    artifact = _artifact()
    payload = {"avg_sell_price": "no-es-un-numero", "is_weekend": 0, "store_id": "CA_2"}
    with pytest.raises(HTTPException) as exc:
        _build_feature_row(payload, artifact)
    assert exc.value.status_code == 400
    assert "columna numérica inválida" in exc.value.detail


def test_null_categorical_without_saved_categories_falls_back_to_empty_domain():
    # artifact viejo (previo a categorical_categories, ver export.py) -- el valor de
    # la fila sigue siendo NaN pase lo que pase, pero el dominio de categorías queda
    # vacío en vez de "adivinado" a partir de esta única fila.
    artifact = _artifact(categorical_categories={})
    payload = {"avg_sell_price": 9.5, "is_weekend": 0, "store_id": None}
    X = _build_feature_row(payload, artifact)
    assert list(X["store_id"].cat.categories) == []
    assert pd.isna(X["store_id"].iloc[0])


def test_registry_resolves_latest_version_by_level_id_prefix(tmp_path, monkeypatch):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({
        "level_01_daily_total_sales": {
            "latest": "20260101T000000Z",
            "versions": {
                "20260101T000000Z": {"path": "artifacts/models/versions/v1.pkl"},
            },
        },
    }))
    monkeypatch.setattr(registry, "REGISTRY_PATH", registry_path)

    assert registry.resolve_version(level_id=1, target="sales") == "20260101T000000Z"
    assert registry._registry_key(level_id=1, target="sales") == "level_01_daily_total_sales"
    assert registry._registry_key(level_id=2, target="sales") is None


def test_registry_missing_file_behaves_as_empty_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "does_not_exist.json")
    assert registry.load_registry() == {}
    assert registry.resolve_version(level_id=1, target="sales") == "unversioned"
