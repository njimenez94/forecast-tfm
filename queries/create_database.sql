PRAGMA memory_limit='4GB';
PRAGMA threads=4;

DROP TABLE IF EXISTS calendar;
DROP TABLE IF EXISTS sell_prices;
DROP TABLE IF EXISTS sales_train_evaluation;
DROP TABLE IF EXISTS sample_submission;
DROP TABLE IF EXISTS dataset_raw;

-- ######################################################################################################################## --

CREATE TABLE calendar AS
    SELECT * FROM read_csv_auto('data/raw/calendar.csv');

CREATE TABLE sell_prices AS
    SELECT * FROM read_csv_auto('data/raw/sell_prices.csv');

CREATE TABLE sales_train_evaluation AS
    UNPIVOT (SELECT * FROM read_csv_auto('data/raw/sales_train_evaluation.csv'))
    ON COLUMNS(* EXCLUDE (id, item_id, dept_id, cat_id, store_id, state_id))
    INTO NAME d VALUE sales;

CREATE TABLE sample_submission AS
    SELECT * FROM read_csv_auto('data/raw/sample_submission.csv');

-- ######################################################################################################################## --

CREATE TABLE dataset_raw AS
    SELECT
        s.item_id || '_' || s.store_id AS series_id,
        s.state_id,
        s.store_id,
        s.cat_id,
        s.dept_id,
        s.item_id,
        c.date,
        p.sell_price,
        s.sales,
        IFNULL(s.sales, 0) * p.sell_price AS mnt_gross_sales,
        CASE s.state_id
            WHEN 'CA' THEN c.snap_CA
            WHEN 'TX' THEN c.snap_TX
            WHEN 'WI' THEN c.snap_WI
        END AS snap,
        COALESCE(c.event_name_1, c.event_name_2) AS event_name,
        COALESCE(c.event_type_1, c.event_type_2) AS event_type
    FROM sales_train_evaluation s
    LEFT JOIN calendar c
        ON s.d = c.d
    LEFT JOIN sell_prices p
        ON  s.store_id = p.store_id
        AND s.item_id = p.item_id
        AND c.wm_yr_wk = p.wm_yr_wk
    ORDER BY
        series_id,
        c.date;