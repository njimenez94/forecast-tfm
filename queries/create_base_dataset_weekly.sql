SELECT
    s.id,
    s.item_id,
    s.dept_id,
    s.cat_id,
    s.store_id,
    s.state_id,
    date_trunc('week', c.date)                          AS week_start_date,
    MAX(c.date)                                         AS week_end_date,
    COUNT(DISTINCT c.event_name_1)
        FILTER (WHERE c.event_name_1 IS NOT NULL)       AS n_events_1,
    COUNT(DISTINCT c.event_name_2)
        FILTER (WHERE c.event_name_2 IS NOT NULL)       AS n_events_2,
    SUM(
        CASE s.state_id
            WHEN 'CA' THEN c.snap_CA
            WHEN 'TX' THEN c.snap_TX
            WHEN 'WI' THEN c.snap_WI
        END
    )                                                   AS snap_days,
    SUM(s.sales)                                        AS sales,
    SUM(s.sales * p.sell_price) / NULLIF(SUM(s.sales), 0) AS avg_sell_price
FROM sales_train_evaluation s
LEFT JOIN calendar c
    ON s.d = c.d
LEFT JOIN sell_prices p
    ON  s.store_id  = p.store_id
    AND s.item_id   = p.item_id
    AND c.wm_yr_wk  = p.wm_yr_wk
GROUP BY
    s.id,
    s.item_id,
    s.dept_id,
    s.cat_id,
    s.store_id,
    s.state_id,
    date_trunc('week', c.date)
ORDER BY
    s.id,
    week_start_date;
