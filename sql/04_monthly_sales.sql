-- Monthly sales trend (note: volume is truncated after 2017-10 in this dataset).
SELECT date_trunc('month', order_date) AS month, ROUND(SUM(sales),0) AS sales
FROM orders GROUP BY 1 ORDER BY 1;
