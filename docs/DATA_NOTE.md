# Data note

**Dataset:** DataCo Smart Supply Chain (Kaggle, Fabian Constante), ~180k orders,
2015-01 to 2018-01, markets across 5 regions.

## Governance
PII columns (customer email, first/last name, password, street; product image/description)
are dropped on load and never used.

## Leakage (late-delivery model)
To predict `Late_delivery_risk` AT ORDER TIME, post-shipment fields are EXCLUDED
from the model: `Days for shipping (real)`, `Delivery Status`, and `shipping date`
are only known after delivery. The model uses pre-shipment features only
(scheduled days, shipping mode, market, region, segment, department, order calendar,
quantity, price, discount, sales). Those post-shipment fields are used only for
descriptive shipping-variance analysis.

## Forecasting
Order volume collapses after 2017-10 (the feed is truncated), so those incomplete
months are excluded from the forecast series. Demand is near-stationary on the clean
period; a neural net did not beat a seasonal model on so few monthly points, so
forecasting uses Holt-Winters and the neural network is applied to the 180k-row
late-delivery task instead, where the data supports it.
