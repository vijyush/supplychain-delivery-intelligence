-- Worst categories by total profit + their loss-order rate (cost-to-serve).
SELECT "Category Name" AS category, COUNT(*) AS orders,
       ROUND(SUM(profit),0) AS profit,
       ROUND(100.0*AVG(CASE WHEN profit<0 THEN 1 ELSE 0 END),2) AS loss_order_rate_pct
FROM orders GROUP BY 1 ORDER BY profit ASC LIMIT 10;
