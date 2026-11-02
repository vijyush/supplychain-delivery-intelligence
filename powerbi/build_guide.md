# Power BI build guide — Supply Chain Intelligence

1. `make all` -> writes outputs/powerbi/*.csv.
2. Get Data -> Text/CSV: load fact_orders_sample, profit_by_region, profit_by_segment,
   profit_by_category, shipping_variance, forecast, customer_segments, dim_date.
3. View -> Themes -> Browse -> powerbi/theme.json.
4. Model: relate dim_date[date] -> fact_orders_sample[order_date].
5. Add the measures from DAX_measures.md.
6. Pages:
   - Executive: cards (Sales, Profit, Margin %, Late Rate, Loss Rate); profit by region bar; monthly sales/forecast line.
   - Delivery: late rate by shipping mode; scheduled vs actual days; late-risk review-queue table (sort late_prob desc); feature importance (load outputs/feature_importance.csv).
   - Profit & customers: profit by category (loss-makers red); RFM segment table (customer_segments); segment slicer.
> Tableau: same CSVs, measures as calculated fields.
