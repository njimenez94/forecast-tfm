"""Inspección ad hoc de artifacts/datasets/daily/: filas, series, columnas ID y
tamaño en disco/RAM por parquet. Script de diagnóstico manual, no parte del
pipeline (create-database/process-data/build-datasets/train-dataset).

Uso: uv run python -m scripts.inspect_datasets
"""
import duckdb
import pandas as pd

import config

GRAIN = "daily"
PROC_DIR = config.DATASETS / GRAIN
OUT_PATH = config.OUTPUT_DIR / f"datasets_{GRAIN}_info.tsv"

ID_COLS = ["state_id", "store_id", "cat_id", "dept_id", "item_id"]
SAMPLE_SIZE = 100_000


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    files = sorted(PROC_DIR.glob("*.parquet"))
    rows = []

    for file in files:
        filename = file.stem
        parts = filename.split("__")
        base = parts[0]                                 # dataset_level_10_daily_item
        filtro = parts[1] if len(parts) > 1 else None    # FOODS_3 / CA_3_FOODS_3

        split = base.split("_")
        cols = [c[0] for c in con.execute(f"DESCRIBE SELECT * FROM '{file}'").fetchall()]

        stats = {"level": split[2]}
        stats["grainly"] = split[3]
        stats["aggregation"] = "_".join(split[4:])
        stats["filtro"] = filtro

        stats["n_rows"] = con.execute(f"SELECT COUNT(*) FROM '{file}'").fetchone()[0]
        stats["n_days"] = con.execute(f"SELECT COUNT(DISTINCT date) FROM '{file}'").fetchone()[0] if "date" in cols else None

        # n_series = combinaciones únicas de las columnas ID presentes
        present_ids = [c for c in ID_COLS if c in cols]
        if present_ids:
            concat_expr = " || '_' || ".join(f"CAST({c} AS VARCHAR)" for c in present_ids)
            stats["n_series"] = con.execute(f"SELECT COUNT(DISTINCT {concat_expr}) FROM '{file}'").fetchone()[0]
        else:
            stats["n_series"] = 1

        for col in ID_COLS:
            stats[f"n_{col}"] = con.execute(f"SELECT COUNT(DISTINCT {col}) FROM '{file}'").fetchone()[0] if col in cols else None

        # Tamaño en disco (parquet comprimido)
        stats["disk_mb"] = round(file.stat().st_size / 1024**2, 2)

        # Estimación de RAM real que ocuparía en pandas
        if stats["n_rows"] <= SAMPLE_SIZE:
            sample_df = con.execute(f"SELECT * FROM '{file}'").df()
            stats["ram_mb"] = round(sample_df.memory_usage(deep=True).sum() / 1024**2, 2)
        else:
            sample_df = con.execute(f"SELECT * FROM '{file}' USING SAMPLE {SAMPLE_SIZE} ROWS").df()
            mem_sample = sample_df.memory_usage(deep=True).sum()
            stats["ram_mb"] = round(
                (mem_sample / len(sample_df)) * stats["n_rows"] / 1024**2, 2
            )

        print(file, stats["n_rows"])
        rows.append(stats)

    con.close()

    resumen = pd.DataFrame(rows)
    resumen.to_csv(OUT_PATH, sep="\t", index=False)
    print(f"Guardado en {OUT_PATH}")
    print(resumen)


if __name__ == "__main__":
    main()
