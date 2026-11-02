# DAX measures — Supply Chain Intelligence

Load outputs/powerbi/*.csv, relate dim_date[date] to fact_orders_sample[order_date].
Create each as a New measure on fact_orders_sample.

```DAX
Orders = DISTINCTCOUNT(fact_orders_sample[order_id])
Sales = SUM(fact_orders_sample[sales])
Profit = SUM(fact_orders_sample[profit])
Margin % = DIVIDE([Profit],[Sales])
Late Rate = AVERAGE(fact_orders_sample[Late_delivery_risk])
Loss Orders = CALCULATE(COUNTROWS(fact_orders_sample), fact_orders_sample[profit] < 0)
Loss Rate = DIVIDE([Loss Orders], COUNTROWS(fact_orders_sample))
Flagged Late = CALCULATE(COUNTROWS(fact_orders_sample), fact_orders_sample[late_prob] >= 0.5)
Avg Late Risk = AVERAGE(fact_orders_sample[late_prob])
```
