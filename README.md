# CS-404 Big Data Analytics — Assignment 03
## ETL Pipeline & Analytics

**NUST SEECS | Spring 2026 | Instructor: Ms. Zahida Kausar**

---

## Project Overview

This assignment builds directly on A2. The cleaned Olist Brazilian E-Commerce dataset
(already ingested into HDFS in A2) is taken through a complete data warehousing lifecycle:

1. **etl.py** — PySpark ETL: transforms raw CSVs into a star schema (Parquet, HDFS)
2. **analytics.py** — Spark SQL: answers 5 business questions with window functions, visualizations

### Dataset
- **Source:** [Kaggle — Olistbr Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- **Tables:** 8 relational CSV files (~1.1 million rows total)

### Star Schema
```
fact_orders
  ├── dim_customers
  ├── dim_products
  ├── dim_sellers
  ├── dim_date
  ├── dim_payments
  └── dim_reviews
```

---

## Repository Structure

```
Ecommerce_Analysis_BDA/
├── etl.py                  # PySpark ETL: transform → load → validate
├── analytics.py            # Spark SQL queries + 4 visualizations
├── final_report.pdf        # Complete A3 report
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── hdfs_screenshot.png     # /warehouse/processed/ directory screenshot
├── charts/                 # Generated chart PNGs
│   ├── chart1_monthly_revenue.png
│   ├── chart2_category_revenue.png
│   ├── chart3_delivery_heatmap.png
│   └── chart4_dashboard.png
└── logs/                   # ETL and analytics run logs
```

---

## Prerequisites

### 1. Complete A2 first
Ensure `ingest.py` from A2 has been run and all 8 CSV files are in HDFS at:
```
/warehouse/raw/olist/year=2026/month=04/
```

### 2. Python 3.10+
```bash
python --version
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Apache Spark 3.4+
```bash
spark-submit --version
```

Ensure `SPARK_HOME` and `JAVA_HOME` are set:
```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export SPARK_HOME=/opt/spark
export PATH=$SPARK_HOME/bin:$PATH
```

### 5. Hadoop HDFS running
```bash
start-dfs.sh
hdfs dfsadmin -report
```

---

## How to Run

### Step 1 — Run the ETL pipeline

**Local mode (for testing, reads from `data/raw/`):**
```bash
python etl.py
```

**Cluster mode (reads from HDFS):**
Edit `etl.py` and set `USE_HDFS = True`, then:
```bash
spark-submit --master yarn \
             --deploy-mode client \
             --num-executors 4 \
             --executor-memory 4g \
             etl.py
```

This will:
1. Read raw CSVs from HDFS (or `data/raw/`)
2. Apply all transformations referencing A2 profiling findings
3. Build star schema tables
4. Write Parquet to `/warehouse/processed/`
5. Run validation (null checks, row count assertions)
6. Log everything to `logs/etl_*.log`

**Expected HDFS output:**
```
/warehouse/processed/
├── fact_orders/order_year=2016/order_month=10/...
├── dim_customers/
├── dim_products/
├── dim_sellers/
├── dim_payments/
├── dim_reviews/
├── dim_date/
└── dim_geolocation/
```

### Step 2 — Run the analytics pipeline

**Local mode:**
```bash
python analytics.py
```

**Cluster mode:**
Edit `analytics.py` and set `USE_HDFS = True`, then:
```bash
spark-submit --master yarn \
             --deploy-mode client \
             analytics.py
```

This will:
1. Load Parquet tables as Spark SQL views
2. Execute 5 business-question queries (with window functions)
3. Save 4 chart PNGs to `charts/`
4. Log results and interpretations to `logs/analytics_*.log`

---

## Optimization Techniques Applied

| Technique | Where | Details |
|-----------|-------|---------|
| Partitioning | `etl.py` → `load_table()` | `fact_orders` partitioned by `order_year`, `order_month` — enables partition pruning for time-bounded queries |
| Broadcast Join | `etl.py` → `demo_broadcast_join()` | `dim_sellers` (~3k rows) broadcast to all executors, avoiding a full shuffle of `fact_orders` |
| Caching | `analytics.py` → `load_views()` | `fact_orders` and `dim_products` cached in memory — reused across 5 queries without re-reading disk |
| Query Plan Analysis | `etl.py` → `main()` | `.explain(True)` output logged for the partitioned time-filtered aggregation |

---

## Group Members

| Name | Student ID | Role |
|------|------------|------|
| Hassan Aamir | 453976 | ETL pipeline (etl.py), HDFS setup, README |
| Umair Naeem | 479984 | Analytics queries, visualizations, final report |

---

## Submission Details

- **Assignment:** CS-404 BDA Assignment 03
- **Submission deadline:** Wednesday, 6 May 2026 (11:59 PM)
- **Submission file:** `HassanAamir_UmairNaeem_A3_BDA.zip`
- **Instructor:** Ms. Zahida Kausar
- **GitHub:** [Ecommerce Analysis BDA](https://github.com/hassan2-aamir/Ecommerce_Analysis_BDA)
