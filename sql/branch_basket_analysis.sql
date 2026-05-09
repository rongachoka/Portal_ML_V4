SELECT
    branch,
    DATE_TRUNC('month', sale_date)          AS month,
    COUNT(*)                                AS transactions,
    ROUND(AVG(basket_value), 0)             AS avg_basket,
    ROUND(AVG(departments_per_txn), 2)      AS avg_departments,
    ROUND(AVG(unique_products_per_txn), 2)  AS avg_unique_products,
    ROUND(AVG(total_units_per_txn), 2)      AS avg_units
FROM vw_basket_analysis
WHERE sale_date >= '2026-01-01'
GROUP BY branch, DATE_TRUNC('month', sale_date)
ORDER BY branch, month;