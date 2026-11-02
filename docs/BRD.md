# Business Requirements Document — Supply Chain & Delivery Intelligence

**Problem:** late deliveries hurt satisfaction and ~19% of orders lose money.
Leadership needs to predict late shipments at order time, see where margin leaks,
and forecast demand.

**Objectives / KPIs:** late-delivery rate, on-time %, model precision/recall,
profit margin, loss-order rate, demand forecast accuracy (WAPE).

**Scope:** order-level transactions, shipping performance, profit, customers.
Out of scope: PII, live carrier integration.

**Requirements:** FR1 EDA + data quality; FR2 late-delivery risk model (pre-shipment);
FR3 demand forecast; FR4 profit & cost-to-serve; FR5 shipping-mode performance;
FR6 customer RFM segmentation; FR7 dashboard.

**Stakeholders:** Ops/logistics, Finance (margin), Commercial (customers).
