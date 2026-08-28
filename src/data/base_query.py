"""Generador del SQL base por nivel de agregación.

`build_base_query(dims, grain)` produce el esquema base:

    series_id, item_id, dept_id, cat_id, store_id, state_id, date,
    event_name_1, event_type_1, event_name_2, event_type_2, snap, sales, avg_sell_price

`build_exog_query(base_sql, grain)` añade features exógenas ligeras (precio, eventos,
calendario) sin lags de y ni rolling windows — esos los gestiona mlforecast internamente.
"""

ALL_DIMS = ("item_id", "dept_id", "cat_id", "store_id", "state_id")

_SNAP = (
    "CASE s.state_id WHEN 'CA' THEN c.snap_CA "
    "WHEN 'TX' THEN c.snap_TX WHEN 'WI' THEN c.snap_WI END"
)
_EVENTS = ("event_name_1", "event_type_1", "event_name_2", "event_type_2")

# Precio lag por granularidad (evita leakage: la semana anterior / el día anterior)
_PRICE_LAG = {"daily": 7, "weekly": 1}


def build_base_query(dims, grain: str) -> str:
    dims = list(dims)
    if grain not in ("daily", "weekly"):
        raise ValueError(f"grain inválido: {grain!r} (usa 'daily' o 'weekly')")
    weekly = grain == "weekly"

    event_agg = "MAX" if weekly else "any_value"

    if dims:
        series_id = "concat_ws('_', " + ", ".join(f"s.{d}" for d in dims) + ") AS series_id"
    else:
        series_id = "'TOTAL' AS series_id"

    dim_cols = [f"s.{d}" if d in dims else f"'TOTAL' AS {d}" for d in ALL_DIMS]
    events = [f"{event_agg}(c.{c}) AS {c}" for c in _EVENTS]

    if "state_id" in dims:
        snap_sel = f"MAX({_SNAP}) AS snap" if weekly else f"any_value({_SNAP}) AS snap"
    else:
        snap_sel = "0 AS snap"

    date_sel = "MIN(c.date) AS date" if weekly else "c.date"
    time_group = "c.wm_yr_wk" if weekly else "c.date"

    select = [
        series_id,
        *dim_cols,
        date_sel,
        *events,
        snap_sel,
        "SUM(s.sales) AS sales",
        "SUM(s.sales * p.sell_price) AS gross_sales",
        "SUM(s.sales * p.sell_price) / NULLIF(SUM(s.sales), 0) AS avg_sell_price",
    ]
    group = [f"s.{d}" for d in dims] + [time_group]

    return (
        "SELECT\n    " + ",\n    ".join(select) + "\n"
        "FROM sales_train_evaluation s\n"
        "LEFT JOIN calendar c ON s.d = c.d\n"
        "LEFT JOIN sell_prices p\n"
        "    ON s.store_id = p.store_id AND s.item_id = p.item_id AND c.wm_yr_wk = p.wm_yr_wk\n"
        "GROUP BY " + ", ".join(group) + "\n"
        "ORDER BY series_id, date"
    )


def build_exog_query(base_sql: str, grain: str) -> str:
    """Añade features exógenas sobre base_sql: precio (con un único LAG), eventos y
    calendario básico. Sin lags de y ni rolling windows — esos los genera mlforecast.
    """
    price_lag = _PRICE_LAG[grain]

    return f"""WITH base AS (
{base_sql}
),
exog AS (
    SELECT *,
        LAG(avg_sell_price, {price_lag}) OVER (PARTITION BY series_id ORDER BY date) AS _plg,
        AVG(avg_sell_price) OVER (PARTITION BY series_id) AS _pmean
    FROM base
)
SELECT
    series_id, item_id, dept_id, cat_id, store_id, state_id, date,
    CAST(sales AS FLOAT) AS sales,
    CAST(gross_sales AS FLOAT) AS gross_sales,
    CAST(avg_sell_price AS FLOAT) AS avg_sell_price,
    -- Price features (un solo LAG, sin power-set de lags de y)
    CAST(_plg AS FLOAT) AS price_lag_{price_lag},
    CAST(avg_sell_price / NULLIF(_plg, 0) - 1 AS FLOAT) AS price_change,
    CAST(avg_sell_price / NULLIF(_pmean, 0) AS FLOAT) AS price_vs_mean,
    -- Events
    CAST(event_name_1 IS NOT NULL AS TINYINT) AS has_event,
    CAST(event_name_2 IS NOT NULL AS TINYINT) AS has_event_2,
    CAST(snap AS TINYINT) AS snap,
    event_name_1, event_type_1, event_name_2, event_type_2,
    -- Calendar básico (mlforecast añade date_features adicionales desde ds)
    CAST(EXTRACT(YEAR   FROM date) AS SMALLINT) AS year,
    CAST(EXTRACT(MONTH  FROM date) AS TINYINT)  AS month,
    CAST(EXTRACT(DAY    FROM date) AS TINYINT)  AS day,
    CAST(EXTRACT(ISODOW FROM date) AS TINYINT) AS dayofweek,
    CAST(EXTRACT(WEEK   FROM date) AS TINYINT)  AS weekofyear,
    CAST(EXTRACT(ISODOW FROM date) >= 6 AS TINYINT) AS is_weekend
FROM exog
ORDER BY series_id, date"""


def build_cum_query(exog_sql: str, ns: list[int]) -> str:
    """Añade columnas cumN = suma de N períodos empezando HOY, una por cada N en ns.

    cumN[t] = sales[t] + ... + sales[t+N-1], vía ROWS BETWEEN CURRENT ROW AND N-1 FOLLOWING
    (partición por series_id, orden por date) -- "los próximos N días incluyendo el actual"
    (p.ej. cum7 un lunes = venta de esa semana lunes-domingo). DuckDB no devuelve NULL cuando
    el frame tiene menos de N filas disponibles (suma parcial silenciosa) — el CASE/COUNT
    fuerza NULL en las últimas N filas de cada serie (ventana incompleta), que se filtran en
    train time según el target elegido, no aquí.
    """
    def _cum_col(n: int) -> str:
        frame = f"PARTITION BY series_id ORDER BY date ROWS BETWEEN CURRENT ROW AND {n - 1} FOLLOWING"
        return (
            f"CASE WHEN COUNT(sales) OVER ({frame}) = {n} "
            f"THEN CAST(SUM(sales) OVER ({frame}) AS FLOAT) ELSE NULL END AS cum{n}"
        )

    cum_cols = ",\n        ".join(_cum_col(n) for n in ns)
    return f"""WITH _exog AS (
{exog_sql}
)
SELECT
    *,
        {cum_cols}
FROM _exog
ORDER BY series_id, date"""


def count_series_query(dims) -> str:
    """Cuenta el nº de series (combinaciones distintas de dims) de un nivel."""
    dims = list(dims)
    if not dims:
        return "SELECT 1 AS n"
    cols = ", ".join(f"s.{d}" for d in dims)
    return f"SELECT COUNT(*) AS n FROM (SELECT DISTINCT {cols} FROM sales_train_evaluation s)"
