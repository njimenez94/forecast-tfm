-- Solo usamos sales_train_evaluation: ya contiene toda la serie de
-- sales_train_validation MÁS los últimos 28 días (no son splits disjuntos,
-- evaluation es un superconjunto de validation). El split train/valid se hace
-- por fecha en el notebook (cutoff = max_date - HORIZON).
WITH sales AS (
    SELECT * FROM sales_train_evaluation
)
SELECT
    s.dept_id || '_' || s.cat_id || '_' || s.store_id || '_' || s.state_id AS agg_id,
    s.dept_id,
    s.cat_id,
    s.store_id,
    s.state_id,
    c.date,
    c.event_name_1,
    c.event_type_1,
    c.event_name_2,
    c.event_type_2,
    CASE s.state_id
        WHEN 'CA' THEN c.snap_CA
        WHEN 'TX' THEN c.snap_TX
        WHEN 'WI' THEN c.snap_WI
    END                             AS snap,
    SUM(s.sales)                                        AS sales,
    SUM(s.sales * p.sell_price) / NULLIF(SUM(s.sales), 0) AS avg_sell_price
FROM sales s
LEFT JOIN calendar c
    ON s.d = c.d
LEFT JOIN sell_prices p
    ON  s.store_id  = p.store_id
    AND s.item_id   = p.item_id
    AND c.wm_yr_wk  = p.wm_yr_wk
GROUP BY
    s.dept_id,
    s.cat_id,
    s.state_id,
    s.store_id,
    c.date,
    c.event_name_1,
    c.event_type_1,
    c.event_name_2,
    c.event_type_2,
    snap
ORDER BY
    s.dept_id,
    s.cat_id,
    s.state_id,
    s.store_id,
    c.date;
