DROP TABLE IF EXISTS calendar;
DROP TABLE IF EXISTS sell_prices;
DROP TABLE IF EXISTS sales_train_evaluation;
DROP TABLE IF EXISTS sales_train_validation;
DROP TABLE IF EXISTS sample_submission;

CREATE TABLE calendar AS
    SELECT * FROM read_csv_auto('data/raw/calendar.csv');

CREATE TABLE sell_prices AS
    SELECT * FROM read_csv_auto('data/raw/sell_prices.csv');

CREATE TABLE sales_train_evaluation AS
    UNPIVOT (SELECT * FROM read_csv_auto('data/raw/sales_train_evaluation.csv'))
    ON COLUMNS(* EXCLUDE (id, item_id, dept_id, cat_id, store_id, state_id))
    INTO NAME d VALUE sales;

CREATE TABLE sales_train_validation AS
    UNPIVOT (SELECT * FROM read_csv_auto('data/raw/sales_train_validation.csv'))
    ON COLUMNS(* EXCLUDE (id, item_id, dept_id, cat_id, store_id, state_id))
    INTO NAME d VALUE sales;

CREATE TABLE sample_submission AS
    SELECT * FROM read_csv_auto('data/raw/sample_submission.csv');
