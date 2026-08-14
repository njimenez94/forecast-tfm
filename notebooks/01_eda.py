import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import duckdb
    import polars as pl
    import pandas as pd

    conn = duckdb.connect("../data/m5.db")
    return conn, mo, pd


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
               COUNT(DISTINCT item_id)       AS n_item_id,
               COUNT(DISTINCT dept_id)       AS n_dept_id,
               COUNT(DISTINCT cat_id)        AS n_cat_id,
               COUNT(DISTINCT state_id)      AS n_state_id,
               COUNT(DISTINCT store_id)      AS n_store_id,
               COUNT(DISTINCT agg_id)        AS n_agg_id,
               COUNT(DISTINCT date)          AS n_days,
               MIN(date)                     AS date_min,
               MAX(date)                     AS date_max
        FROM dataset_raw
    """).df()

    df_stats.T.rename(columns={0: "value"})
    return


@app.cell
def _(conn):
    query = """
    SELECT
        agg_id,
        COUNT(DISTINCT date) as n_days,
        SUM(units_sales) as units_sales,
        SUM(mnt_gross_sales) as mnt_gross_sales
    FROM dataset_raw
    WHERE units_sales != 0
    GROUP BY agg_id
    ORDER BY mnt_gross_sales DESC
    """

    df_agg_id = conn.sql(query).df().round(0)
    df_agg_id
    return (df_agg_id,)


@app.cell
def _(conn):
    query_cat = """
    SELECT
        cat_id,
        COUNT(DISTINCT date) as n_days,
        SUM(units_sales) as units_sales,
        SUM(mnt_gross_sales) as mnt_gross_sales
    FROM dataset_raw
    WHERE units_sales != 0
    GROUP BY cat_id
    ORDER BY mnt_gross_sales DESC
    """

    df_cat = conn.sql(query_cat).df().round(0)
    df_cat
    return


@app.cell
def _(conn):
    query_dept = """
    SELECT
        cat_id,
        COUNT(DISTINCT date) as n_days,
        SUM(units_sales) as units_sales,
        SUM(mnt_gross_sales) as mnt_gross_sales
    FROM dataset_raw
    WHERE units_sales != 0
    GROUP BY cat_id
    ORDER BY mnt_gross_sales DESC
    """

    df_dept = conn.sql(query_dept).df().round(0)
    df_dept
    return


@app.cell
def _(conn, df_agg_id, pd):
    def get_sample_ids(df_agg_id, conn, col_id='agg_id', n_samples=30, table='dataset_raw') -> pd.DataFrame:
        n = len(df_agg_id)
        mid_start = n // 2 - n_samples // 2

        top = df_agg_id.iloc[:n_samples]
        mid = df_agg_id.iloc[mid_start:mid_start + n_samples]
        bottom = df_agg_id.iloc[-n_samples:]

        series_id = (
            top[col_id].tolist() +
            mid[col_id].tolist() +
            bottom[col_id].tolist()
        )

        ids_str = ", ".join(f"'{x}'" for x in series_id)

        query = f"""
        SELECT * FROM {table}
        WHERE {col_id} IN ({ids_str})
        """

        df_sample = conn.sql(query).df()

        return df_sample, series_id


    df_sample, series_id = get_sample_ids(df_agg_id, conn, col_id='agg_id', n_samples=100, table='dataset_raw')
    df_sample
    return df_sample, series_id


@app.cell
def _(df_agg_id, series_id):
    df_agg_id[df_agg_id['agg_id'].isin(series_id)]
    return


@app.cell
def _(df_sample):
    df_series = df_sample.query('agg_id == "FOODS_3_090_CA_3"').set_index('date')['sales']

    df_series.plot(x='date', y='sales')
    return


if __name__ == "__main__":
    app.run()
