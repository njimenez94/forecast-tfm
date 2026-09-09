"""Self-check de partición temporal / anti-leakage, relocalizado desde el antiguo
`if __name__ == "__main__":` de src/data/split.py. Sin pytest: correr directo con
`python tests/test_temporal_split.py`.
"""
import numpy as np
import pandas as pd

from src.data.temporal_split import block_origins, rolling_cv_folds, valid_blocks


def test_block_origins_and_valid_blocks():
    dates = pd.Series(pd.date_range("2016-01-01", periods=364, freq="D"))
    ref = dates.iloc[0]
    origins = block_origins(dates, ref, block_periods=28, grain="daily")
    # dentro del primer bloque el origen es siempre el día anterior a ref
    assert (origins[:28] == np.datetime64(ref - pd.Timedelta(days=1))).all()
    # el bloque 2 (día 28) resetea el origen 28 días más tarde
    assert origins[28] == np.datetime64(ref + pd.Timedelta(days=27))

    valid_start, test_start = dates.iloc[0], dates.iloc[-1] + pd.Timedelta(days=1)
    blocks = valid_blocks(valid_start, test_start, block_periods=28, grain="daily")
    assert len(blocks) == 13
    assert blocks[0][1] == valid_start
    assert blocks[-1][2] == test_start
    assert all(b[2] == blocks[i + 1][1] for i, b in enumerate(blocks[:-1]))  # sin huecos/solapes


def test_rolling_cv_folds():
    n = 200
    cv_dates = pd.Series(pd.date_range("2015-01-01", periods=n, freq="D"))
    cv_X_train = pd.DataFrame({"lag7": np.arange(n, dtype=float)})
    cv_y_train = pd.Series(np.arange(n, dtype=float))
    cv_valid_start = cv_dates.iloc[-1] + pd.Timedelta(days=1)
    cv_X_valid = pd.DataFrame({"lag7": np.arange(28, dtype=float)})
    cv_y_valid = pd.Series(np.arange(28, dtype=float))

    cv_folds = rolling_cv_folds(cv_X_train, cv_y_train, cv_X_valid, cv_y_valid,
                                cv_dates, cv_valid_start, grain="daily", n_folds=3)
    assert len(cv_folds) == 3
    # último fold == split actual, sin recortar
    assert cv_folds[-1][0] is cv_X_train and cv_folds[-1][2] is cv_X_valid
    # folds más viejos: 28 filas de valid, train nunca llega a la fecha de corte
    for i, (X_tr, _, X_val, _) in enumerate(cv_folds[:-1]):
        assert len(X_val) == 28
        assert len(X_tr) == n - (len(cv_folds) - 1 - i) * 28
    # enmascarado de horizonte en el fold más viejo: lag7 solo válido para h<=7
    oldest_val = cv_folds[0][2]["lag7"]
    assert oldest_val.iloc[:7].notna().all()
    assert oldest_val.iloc[7:].isna().all()


if __name__ == "__main__":
    test_block_origins_and_valid_blocks()
    test_rolling_cv_folds()
    print("tests/test_temporal_split.py OK")
