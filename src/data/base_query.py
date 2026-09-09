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


def _where_clause(filters, alias: str = "s") -> str:
    """WHERE opcional a partir de un dict {dim: (valores,)}, p.ej. {"dept_id": ("FOODS_3",)}.
    Vacío/None -> sin filtrar (comportamiento por defecto)."""
    if not filters:
        return ""
    conds = []
    for dim, values in filters.items():
        vals = ", ".join(f"'{v}'" for v in values)
        conds.append(f"{alias}.{dim} IN ({vals})")
    return "WHERE " + " AND ".join(conds) + "\n"


def build_base_query(dims, grain: str, filters=None) -> str:
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
    # Semana ISO lunes-domingo (DuckDB date_trunc('week', ...) trunca al lunes), no la
    # semana retail wm_yr_wk de M5 (sábado-viernes) -- MIN(c.date) por grupo da ese
    # lunes como "date" de la fila (ver config.MLFORECAST_FREQ, anclado igual a lunes).
    time_group = "date_trunc('week', c.date)" if weekly else "c.date"

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
        + _where_clause(filters) +
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


def build_level_query(dims, grain: str, filters=None) -> str:
    """Compone base + exog en el SQL final de un nivel/grain (ver
    build_base_query/build_exog_query). `filters` (opcional) restringe las filas de
    origen, p.ej. {"dept_id": ("FOODS_3",)} -- ver Level.filters."""
    base_sql = build_base_query(dims, grain, filters)
    return build_exog_query(base_sql, grain)


def count_series_query(dims, filters=None) -> str:
    """Cuenta el nº de series (combinaciones distintas de dims) de un nivel, aplicando
    `filters` si se pasa (ver Level.filters)."""
    dims = list(dims)
    if not dims:
        return "SELECT 1 AS n"
    cols = ", ".join(f"s.{d}" for d in dims)
    where = _where_clause(filters)
    return f"SELECT COUNT(*) AS n FROM (SELECT DISTINCT {cols} FROM sales_train_evaluation s\n{where})"
