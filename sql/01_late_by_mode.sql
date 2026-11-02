-- Late-delivery rate by shipping mode (faster classes are MORE late here).
SELECT "Shipping Mode" AS shipping_mode, COUNT(*) AS orders,
       ROUND(100.0*AVG(Late_delivery_risk),2) AS late_rate_pct
FROM orders GROUP BY 1 ORDER BY late_rate_pct DESC;
