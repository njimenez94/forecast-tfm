PRAGMA memory_limit='2.5GB';
PRAGMA threads=2;
PRAGMA temp_directory='data/duckdb_tmp';

DROP TABLE IF EXISTS dataset_raw;
DROP TABLE IF EXISTS sales_train_evaluation;
DROP TABLE IF EXISTS sell_prices;
DROP TABLE IF EXISTS calendar;

-- ######################################################################################################################## --

CREATE TEMP TABLE _sales_raw AS
    SELECT * FROM read_csv_auto('data/raw/sales_train_evaluation.csv');

CREATE TEMP TABLE _calendar_raw AS
    SELECT * FROM read_csv_auto('data/raw/calendar.csv');

CREATE TEMP TABLE _sell_prices_raw AS
    SELECT * FROM read_csv_auto('data/raw/sell_prices.csv');

-- ######################################################################################################################## --

CREATE TABLE calendar (
    date VARCHAR,
    wm_yr_wk INTEGER,
    weekday VARCHAR,
    wday INTEGER,
    month INTEGER,
    year INTEGER,
    d VARCHAR PRIMARY KEY,
    event_name_1 VARCHAR,
    event_type_1 VARCHAR,
    event_name_2 VARCHAR,
    event_type_2 VARCHAR,
    snap_CA INTEGER,
    snap_TX INTEGER,
    snap_WI INTEGER
);
INSERT INTO calendar SELECT * FROM _calendar_raw;

CREATE TABLE sell_prices (
    store_id VARCHAR,
    item_id VARCHAR,
    wm_yr_wk INTEGER,
    sell_price DOUBLE,
    PRIMARY KEY (store_id, item_id, wm_yr_wk)
);
INSERT INTO sell_prices SELECT * FROM _sell_prices_raw;

CREATE TABLE sales_train_evaluation (
    id VARCHAR,
    item_id VARCHAR,
    dept_id VARCHAR,
    cat_id VARCHAR,
    store_id VARCHAR,
    state_id VARCHAR,
    d VARCHAR REFERENCES calendar(d),
    sales INTEGER
);
INSERT INTO sales_train_evaluation
    SELECT id, item_id, dept_id, cat_id, store_id, state_id, d, sales
    FROM (
        UNPIVOT _sales_raw
        ON COLUMNS(* EXCLUDE (id, item_id, dept_id, cat_id, store_id, state_id))
        INTO NAME d VALUE sales
    );

DROP TABLE _sales_raw;
DROP TABLE _calendar_raw;
DROP TABLE _sell_prices_raw;

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
