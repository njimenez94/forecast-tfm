"""Prueba end-to-end de la API contra una instancia ya corriendo (make serve-api o
make docker-api). Arma un payload real a partir del X_test reconstruido de un
artifact local (mismas filas que backtestea notebooks/03_predictions.ipynb), lo
manda a POST /predict/{level} y compara la predicción devuelta contra
model.predict(...) directo sobre esa fila -- mismo chequeo que se hizo a mano para
validar api/main.py.

    make testing-api ARGS="--level 9 --n 3"
    # o
    uv run python -m scripts.test_api --level 9 --n 3
"""
import argparse
import json
import urllib.error
import urllib.request

import joblib
from loguru import logger

from api.registry import resolve_artifact_path
from src.data.split import reconstruct_test_data
from src.logging_setup import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=9)
    parser.add_argument("--target", default="sales")
    parser.add_argument("--n", type=int, default=1, help="filas de test a probar")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    artifact = joblib.load(resolve_artifact_path(args.level, args.target))
    X_test, *_ = reconstruct_test_data(artifact)
    n = min(args.n, len(X_test))

    ok = 0
    for i in range(n):
        row = X_test.iloc[[i]]
        expected = float(artifact["model"].predict(row)[0])

        features = {}
        for col in artifact["features"]:
            value = row.iloc[0][col]
            features[col] = str(value) if col in artifact["categorical_features"] else float(value)
        payload = json.dumps({"features": features}).encode()

        req = urllib.request.Request(
            f"{args.url}/predict/{args.level}?target={args.target}",
            data=payload, headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                got = json.loads(resp.read())["prediction"]
        except urllib.error.URLError as exc:
            logger.error(
                "No se pudo conectar a {} ({}) -- ¿está la API corriendo? (make serve-api / make docker-api)",
                args.url, exc,
            )
            raise SystemExit(1)

        match = abs(got - expected) < 1e-6
        ok += match
        logger.info("fila {} | esperado={:.4f} api={:.4f} | {}", i, expected, got, "OK" if match else "MISMATCH")

    logger.success("{}/{} predicciones coinciden", ok, n)
    if ok != n:
        raise SystemExit(1)


if __name__ == "__main__":
    configure_logging("test_api")
    main()
