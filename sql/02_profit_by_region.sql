-- Sales, profit and margin by order region.
SELECT "Order Region" AS region, ROUND(SUM(sales),0) AS sales,
       ROUND(SUM(profit),0) AS profit,
       ROUND(100.0*SUM(profit)/SUM(sales),2) AS margin_pct
FROM orders GROUP BY 1 ORDER BY profit DESC;
