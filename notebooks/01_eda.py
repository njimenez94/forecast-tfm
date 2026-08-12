import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import duckdb

    conn = duckdb.connect("../data/m5.db")
    return conn, mo


@app.cell
def _(conn, mo):
    _df = mo.sql(
        f"""
        SELECT COUNT(*) as n_rows,
               COUNT(DISTINCT cat_id) as n_cat_id,
               COUNT(DISTINCT dept_id) as n_dept_id,
               COUNT(DISTINCT item_id) as n_item_id,
               COUNT(DISTINCT date) as n_date,
               MIN(date) as fecha_min,
               MAX(date) as fecha_max
        FROM dataset_raw
        """,
        engine=conn
    )
    return


if __name__ == "__main__":
    app.run()