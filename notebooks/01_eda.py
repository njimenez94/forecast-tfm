import marimo

__generated_with = "0.23.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Config
    """)
    return


@app.cell
def _():
    import marimo as mo
    import duckdb
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns

    conn = duckdb.connect("../data/m5.db")
    return conn, mdates, mo, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Funciones
    """)
    return


@app.cell
def _(pd, plt, sns):
    def plot_share_pie(
        df: pd.DataFrame,
        label_col: str,
        value_col: str,
        threshold: float = 0.05,
        title: str | None = None,
        other_label: str = "Otros",
        palette: str = "tab10",
        figsize: tuple[float, float] = (8, 6),
        autopct: str = "%1.0f%%",
    ):
        """Pie de participación agrupando las categorías bajo `threshold` (0-1) en una sola porción."""
        d = df[[label_col, value_col]].copy()
        d[value_col] = pd.to_numeric(d[value_col])
        total = d[value_col].sum()

        is_major = (d[value_col] / total) >= threshold
        d["label"] = d[label_col].astype(str).where(is_major, other_label)

        # Si solo una categoría cae bajo el umbral, no vale la pena renombrarla
        if (~is_major).sum() == 1:
            d["label"] = d[label_col].astype(str)

        d_plot = (
            d.groupby("label", as_index=False)[value_col]
            .sum()
            .sort_values(value_col, ascending=False)
            .reset_index(drop=True)
        )

        fig = plt.figure(figsize=figsize)
        plt.pie(
            d_plot[value_col],
            labels=d_plot["label"],
            autopct=autopct,
            colors=sns.color_palette(palette, len(d_plot)),
        )
        plt.title(title or f"{value_col} por {label_col}")
        plt.tight_layout()
        return fig

    return (plot_share_pie,)


@app.cell
def _(pd, plt, sns):
    def plot_top_bars(
        df: pd.DataFrame,
        label_col: str,
        value_col: str,
        top_n: int | None = None,
        label_n: int | None = None,
        title: str | None = None,
        palette: str = "Set2",
    ):
        """Horizontal bars sorted desc. top_n=None plots all; label_n=None labels all."""
        s = (
            df.groupby(label_col)[value_col]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )

        n = len(s) if label_n is None else label_n
        labels = [f"{v:,.0f}" if i < n else "" for i, v in enumerate(s.values)]

        plt.figure(figsize=(9, max(3.0, 0.32 * len(s))))
        plt.barh(s.index.astype(str), s.values, color=sns.color_palette(palette, len(s)))
        plt.gca().invert_yaxis()
        plt.bar_label(plt.gca().containers[0], labels=labels, padding=3, fontsize=9)
        plt.margins(x=0.12)
        plt.title(title or f"{value_col} por {label_col}")
        plt.xlabel(value_col)
        plt.ylabel("")
        plt.tight_layout()
        return plt.gcf()

    return (plot_top_bars,)


@app.cell
def _(pd):
    def get_agg_stats(
        conn,
        group_cols: list[str],
        count_cols: list[str] | None = None,
        table: str = "dataset_raw",
        where: str = "units_sales != 0",
    ) -> pd.DataFrame:
        """Estadísticas de ventas agregadas al nivel de granularidad indicado por `group_cols`.

        `count_cols` añade columnas `COUNT(DISTINCT col)` (p.ej. store_id -> n_stores).

        `n_days` es el total de periodos observados para el grupo (sin aplicar `where`);
        `n_days_with_sales` los periodos que pasan el filtro `where` (por defecto, venta != 0);
        `n_days_zero_sales = n_days - n_days_with_sales`. Esta descomposición solo es
        directamente interpretable como "días sin venta de la serie" cuando `group_cols`
        está al grano de la serie (p.ej. `agg_id`); en niveles más agregados, `n_days`
        cuenta fechas distintas entre todas las filas base del grupo, no días con venta
        total del grupo = 0.

        Ambas CTEs se calculan solo sobre el rango "activo" de cada `agg_id`, es decir,
        desde su `release_date` (primera fecha con `price` no nulo, proxy de que el item
        ya estaba a la venta en esa tienda) en adelante. Sin este recorte, los días previos
        al lanzamiento cuentan como "sin venta" y desinflan artificialmente ratios como
        `n_days_with_sales / n_days` (o infllan el ADI) para items lanzados tarde, aunque
        una vez activos vendan casi todos los días.

        `adi` (Average Demand Interval) = `n_days / n_days_with_sales`: nº medio de
        periodos entre dos ventas consecutivas (>= 1; a mayor valor, demanda más
        intermitente).
        """
        cols = ", ".join(group_cols)
        count_names = [f"n_{c.removesuffix('_id')}s" for c in (count_cols or [])]
        counts = "".join(
            f",\n            COUNT(DISTINCT {c}) AS {name}"
            for c, name in zip(count_cols or [], count_names)
        )

        _dim_order = ["cat_id", "dept_id", "item_id", "state_id", "store_id", "agg_id"]
        ordered_group_cols = [c for c in _dim_order if c in group_cols]
        ordered_group_cols += [c for c in group_cols if c not in _dim_order]
        select_order = (
            ordered_group_cols + count_names + [
                "date_min", "date_max", "n_days", "n_days_with_sales", "n_days_zero_sales",
                "adi", "mean_gross_sales_day", "units_sales", "mnt_gross_sales", "mean_price",
            ]
        )
        select_cols = ",\n            ".join(select_order)

        query = f"""
        WITH released AS (
            SELECT agg_id, MIN(date) AS release_date
            FROM {table}
            WHERE price IS NOT NULL
            GROUP BY agg_id
        ),
        active AS (
            SELECT t.*
            FROM {table} t
            JOIN released r USING (agg_id)
            WHERE t.date >= r.release_date
        ),
        totals AS (
            SELECT
                {cols},
                COUNT(DISTINCT date) AS n_days
            FROM active
            GROUP BY {cols}
        ),
        filtered AS (
            SELECT
                {cols}{counts},
                MIN(date)                                   AS date_min,
                MAX(date)                                   AS date_max,
                COUNT(DISTINCT date)                        AS n_days_with_sales,
                SUM(units_sales)                            AS units_sales,
                SUM(mnt_gross_sales)                        AS mnt_gross_sales,
                SUM(mnt_gross_sales) / SUM(units_sales)     AS mean_price,
                SUM(mnt_gross_sales) / COUNT(DISTINCT date) AS mean_gross_sales_day
            FROM active
            WHERE {where}
            GROUP BY {cols}
        ),
        with_adi AS (
            SELECT
                filtered.*,
                totals.n_days,
                totals.n_days - filtered.n_days_with_sales AS n_days_zero_sales,
                totals.n_days / NULLIF(filtered.n_days_with_sales, 0) AS adi
            FROM filtered
            LEFT JOIN totals USING ({cols})
        )
        SELECT
            {select_cols}
        FROM with_adi
        ORDER BY mnt_gross_sales DESC
        """
        return conn.sql(query).df().round(2)

    return (get_agg_stats,)


@app.cell
def _(pd):
    def get_sample_ids(df_ranked, conn, col_id='item_id', n_samples=30, table='dataset_raw') -> pd.DataFrame:
        """Devuelve todas las filas de `table` para `n_samples` ids top, medios y bottom (según orden de `df_ranked`).

        `col_id` suele ser `item_id`, para abarcar el producto completo (todas sus
        combinaciones de tienda/estado, es decir, todas sus series `agg_id`), no una
        muestra parcial de series sueltas.
        """
        n = len(df_ranked)
        mid_start = n // 2 - n_samples // 2

        top = df_ranked.iloc[:n_samples]
        mid = df_ranked.iloc[mid_start:mid_start + n_samples]
        bottom = df_ranked.iloc[-n_samples:]

        selected_ids = (
            top[col_id].tolist() +
            mid[col_id].tolist() +
            bottom[col_id].tolist()
        )

        ids_str = ", ".join(f"'{x}'" for x in selected_ids)

        query = f"""
        SELECT * FROM {table}
        WHERE {col_id} IN ({ids_str})
        """

        df_sample = conn.sql(query).df()

        return df_sample, selected_ids

    return (get_sample_ids,)


@app.cell
def _(mdates, pd, plt):
    def plot_series(
        df: pd.DataFrame,
        agg_id: str,
        n: int = 28,
        value_col: str = "units_sales",
        id_col: str = "agg_id",
        date_col: str = "date",
        figsize: tuple[float, float] = (11, 4),
    ):
        """Serie temporal de los últimos `n` días para un `agg_id`."""
        s = (
            df.query(f"{id_col} == @agg_id")
            .sort_values(date_col)
            .set_index(date_col)[value_col]
            .tail(n)
        )

        if s.empty:
            raise ValueError(f"Sin datos para {id_col} == {agg_id!r}")

        plt.figure(figsize=figsize)
        plt.plot(s.index, s.values, marker="o", ms=4, lw=1.5)

        # Ticks semanales si la ventana es corta, mensuales si es larga
        if n <= 90:
            plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
            plt.gca().xaxis.set_minor_locator(mdates.DayLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
        else:
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))

        plt.title(f"{agg_id} — últimos {len(s)} días")
        plt.xlabel("")
        plt.ylabel(value_col)
        plt.grid(alpha=0.3)
        plt.margins(x=0.02)
        plt.ylim(0)
        plt.tight_layout()
        plt.xticks(rotation=45, ha='right')
        return plt.gcf()

    return (plot_series,)


@app.cell
def _(np):
    def add_sbc_class(
        df,
        conn,
        id_cols: str | list[str] = "agg_id",
        value_col: str = "units_sales",
        table: str = "dataset_raw",
        adi_cut: float = 1.32,
        cv2_cut: float = 0.49,
    ):
        """Devuelve `id_cols` + `adi` + `cv2` + `sbc_class` por serie.

        Clasificación SBC (Syntetos-Boylan-Croston): Smooth / Erratic / Intermittent /
        Lumpy, según cortes ADI=1.32 y CV²=0.49 (Syntetos, Boylan & Croston, 2005).
        CV² se calcula solo sobre demandas positivas (incluir ceros lo infla artificialmente).
        """
        id_cols = [id_cols] if isinstance(id_cols, str) else id_cols
        cols = ", ".join(id_cols)

        cv2 = conn.sql(f"""
            SELECT {cols}, POWER(STDDEV_SAMP({value_col}) / AVG({value_col}), 2) AS cv2
            FROM {table}
            WHERE {value_col} > 0
            GROUP BY {cols}
        """).df()

        out = df.merge(cv2, on=id_cols, how="left")

        intermittent = out["adi"] > adi_cut
        erratic = out["cv2"] > cv2_cut

        out["sbc_class"] = np.select(
            [~intermittent & ~erratic, ~intermittent & erratic,
             intermittent & ~erratic, intermittent & erratic],
            ["Smooth", "Erratic", "Intermittent", "Lumpy"],
            default="Sin demanda",
        )

        return (
            out[id_cols + ["adi", "cv2", "sbc_class"]]
            .sort_values(["sbc_class", "adi"])
            .reset_index(drop=True)
        )

    return (add_sbc_class,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploración
    """)
    return


@app.cell
def _(conn, mo):
    _df = mo.sql(
        f"""
        SHOW ALL TABLES
        """,
        engine=conn
    )
    return


@app.cell
def _(conn, mo):
    _df = mo.sql(
        f"""
        SELECT *
        FROM calendar
        LIMIT 5
        """,
        engine=conn
    )
    return


@app.cell
def _(conn, mo):
    _df = mo.sql(
        f"""
        SELECT date, d
        FROM calendar
        GROUP BY date, d
        ORDER BY date
        """,
        engine=conn
    )
    return


@app.cell
def _(conn, mo):
    _df = mo.sql(
        f"""
        SELECT *
        FROM dataset_raw
        LIMIT 5
        """,
        engine=conn
    )
    return


@app.cell
def _(conn, mo):
    _df = mo.sql(
        f"""
        SELECT item_id, dept_id, cat_id -- , store_id, state_id
        FROM dataset_raw
        GROUP BY item_id, dept_id, cat_id
        ORDER BY item_id, dept_id, cat_id
        LIMIT 5
        """,
        engine=conn
    )
    return


@app.cell
def _(conn):
    df_stats = conn.sql("""
        SELECT COUNT(*)                      AS n_rows,
               COUNT(DISTINCT cat_id)        AS n_cat_id,
               COUNT(DISTINCT dept_id)       AS n_dept_id,
               COUNT(DISTINCT item_id)       AS n_item_id,
               COUNT(DISTINCT state_id)      AS n_state_id,
               COUNT(DISTINCT store_id)      AS n_store_id,
               COUNT(DISTINCT agg_id)        AS n_agg_id,
               MIN(date)                     AS date_min,
               MAX(date)                     AS date_max,
               COUNT(DISTINCT date)          AS n_days
        FROM dataset_raw
    """).df()

    df_stats.T.rename(columns={0: "value"})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Magnitud por nivel de agregación

    `dataset_raw` está al grano más fino (item-store diario, nivel 12). A partir de ahí se
    puede derivar, con `COUNT(DISTINCT ...)`, cuántas series y cuántas filas tendría un
    dataset a cualquiera de los 12 niveles de agregación M5 (ver `config/levels.py`), para
    entender la magnitud de cada uno antes de generarlo. El panel es denso (una fila por
    serie y período), así que `n_rows = n_series * n_periodos`. Se muestran ambos granos
    (diario y semanal) para cada nivel, independientemente del grano con el que se genera
    hoy en `artifacts/datasets`.
    """)
    return


@app.cell
def _(conn, pd):
    LEVELS = [
        (1,  "total",       []),
        (2,  "state",       ["state_id"]),
        (3,  "cat",         ["cat_id"]),
        (4,  "dept",        ["cat_id", "dept_id"]),
        (5,  "state_cat",   ["state_id", "cat_id"]),
        (6,  "store",       ["state_id", "store_id"]),
        (7,  "state_dept",  ["state_id", "cat_id", "dept_id"]),
        (8,  "store_cat",   ["state_id", "store_id", "cat_id"]),
        (9,  "store_dept",  ["state_id", "store_id", "cat_id", "dept_id"]),
        (10, "item",        ["cat_id", "dept_id", "item_id"]),
        (11, "item_state",  ["state_id", "cat_id", "dept_id", "item_id"]),
        (12, "item_store",  ["state_id", "store_id", "cat_id", "dept_id", "item_id"]),
    ]

    n_days = conn.sql("SELECT COUNT(DISTINCT date) FROM dataset_raw").fetchone()[0]
    n_weeks = -(-n_days // 7)  # ceil

    rows = []
    for level_id, name, cols in LEVELS:
        key_expr = "concat_ws('||', " + ", ".join(cols) + ")" if cols else "'total'"
        n_series = conn.sql(f"SELECT COUNT(DISTINCT {key_expr}) FROM dataset_raw").fetchone()[0]
        rows.append({
            "level_id": level_id,
            "level": name,
            "n_series": n_series,
            "n_days": n_days,
            "n_weeks": n_weeks,
            "n_rows_daily": n_series * n_days,
            "n_rows_weekly": n_series * n_weeks,
        })

    df_levels = pd.DataFrame(rows)
    df_levels
    return (df_levels,)


@app.cell
def _(df_levels, plt, sns):
    _df_plot = df_levels.melt(
        id_vars=["level_id", "level"],
        value_vars=["n_rows_daily", "n_rows_weekly"],
        var_name="grain",
        value_name="n_rows",
    )
    _df_plot["grain"] = _df_plot["grain"].map({"n_rows_daily": "daily", "n_rows_weekly": "weekly"})
    _df_plot = _df_plot.sort_values("level_id")

    _fig, _ax = plt.subplots(figsize=(9, 7))
    sns.barplot(_df_plot, x="n_rows", y="level", hue="grain", ax=_ax)
    _ax.set_xscale("log")
    for _container in _ax.containers:
        _ax.bar_label(_container, labels=[f"{v:,.0f}" for v in _container.datavalues], padding=3, fontsize=7)
    _ax.set_title("Filas por nivel de agregación, diario vs. semanal (escala log)", loc="left")
    _ax.set_xlabel("n_rows")
    _ax.set_ylabel("")
    sns.despine()
    plt.tight_layout()
    return


@app.cell
def _(conn, get_agg_stats):
    df_cat = get_agg_stats(conn, ["cat_id"], count_cols=["store_id", "dept_id", "item_id"])
    df_cat
    return (df_cat,)


@app.cell
def _(df_cat, plot_share_pie):
    plot_share_pie(df_cat, value_col='mnt_gross_sales', label_col='cat_id', title="Distribución venta bruta categorías productos")
    return


@app.cell
def _(conn, get_agg_stats):
    df_dept = get_agg_stats(conn, ["cat_id", "dept_id"], count_cols=["store_id", "item_id"])
    df_dept
    return (df_dept,)


@app.cell
def _(df_dept, plot_top_bars):
    plot_top_bars(
        df_dept,
        value_col='mnt_gross_sales',
        label_col='dept_id',
        title="Venta bruta por departamento del producto"
    )
    return


@app.cell
def _(conn, get_agg_stats):
    df_item = get_agg_stats(conn, ["cat_id", "dept_id", "item_id"])
    df_item.to_clipboard(index=False)
    df_item
    return (df_item,)


@app.cell
def _(df_item, plot_top_bars):
    plot_top_bars(df_item, "item_id", "mnt_gross_sales", top_n=20,
                  title="Top 20 ítems por ventas brutas")
    return


@app.cell
def _(conn, get_agg_stats):
    df_state = get_agg_stats(conn, ["state_id"], count_cols=["store_id"])
    df_state
    return (df_state,)


@app.cell
def _(df_state, plot_share_pie):
    plot_share_pie(df_state, value_col='mnt_gross_sales', label_col='state_id', title="Distribución venta bruta por estado")
    return


@app.cell
def _(conn, get_agg_stats):
    df_store = get_agg_stats(conn, ["state_id", "store_id"], count_cols=["dept_id", "item_id"])
    df_store
    return (df_store,)


@app.cell
def _(df_store, plot_top_bars):
    plot_top_bars(df_store, "store_id", "mnt_gross_sales",
                  title="Tiendas por ventas brutas")
    return


@app.cell
def _(conn, get_agg_stats):
    df_agg_id =  get_agg_stats(conn, ["cat_id", "dept_id", "item_id",'state_id','store_id','agg_id'])
    df_agg_id
    return (df_agg_id,)


@app.cell
def _(df_agg_id, plot_top_bars):
    plot_top_bars(df_agg_id, "agg_id", "mnt_gross_sales", top_n=20,
                  title="Top 20 series por venta bruta")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Estudio demanda
    """)
    return


@app.cell
def _(add_sbc_class, conn, df_agg_id):
    df_agg_id_sbc = add_sbc_class(df_agg_id, conn)
    df_agg_id_sbc = df_agg_id.merge(df_agg_id_sbc)
    df_agg_id_sbc
    return (df_agg_id_sbc,)


@app.cell
def _(df_agg_id_sbc):
    df_agg_id_sbc.groupby(["sbc_class"]).agg(
        count = ('agg_id', 'count'),
        adi = ('adi','mean'),
        cv2 = ('cv2','mean'),
        units_sales = ('units_sales', 'sum'),
        mnt_gross_sales = ('mnt_gross_sales', 'sum'),
    ).round(2)#.to_clipboard(index=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Samples
    """)
    return


@app.cell
def _(df_item):
    unique_products = df_item["item_id"].nunique()

    MB_PER_SERIES = 3.2
    N_SAMPLES = 100
    N_SELECTED = 3 * N_SAMPLES

    n_products_selected = min(N_SELECTED, unique_products)
    pct_dataset = n_products_selected / unique_products
    mb_selected =  N_SELECTED * MB_PER_SERIES

    print(
        f"Productos únicos totales : {unique_products}\n"
        f"Productos seleccionados  : {n_products_selected} ({pct_dataset:.2%} del total)\n "
        f"MB seleccionado : {mb_selected} MB"
    )
    return (N_SAMPLES,)


@app.cell
def _(N_SAMPLES, conn, df_item, get_sample_ids):
    df_sample, item_ids = get_sample_ids(df_item, conn, col_id='item_id', n_samples=N_SAMPLES, table='dataset_raw')
    df_sample.info()
    return df_sample, item_ids


@app.cell
def _(df_item, item_ids):
    df_item[df_item['item_id'].isin(item_ids)]
    return


@app.cell
def _(df_sample, plot_series):
    plot_series(df_sample, "FOODS_3_090_WI_3", n=90)
    return


@app.cell
def _(df_sample, plot_series):
    plot_series(df_sample, "FOODS_3_120_CA_3", n=90)
    return


@app.cell
def _(df_sample, plot_series):
    plot_series(df_sample, "FOODS_3_444_WI_2", n=90)
    return


if __name__ == "__main__":
    app.run()
