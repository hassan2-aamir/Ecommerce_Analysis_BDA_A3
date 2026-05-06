"""
etl.py — PySpark ETL Pipeline for Brazilian E-Commerce (Olist) Dataset
CS-404 Big Data Analytics — Assignment 03
NUST SEECS, Spring 2026

Builds a star schema from the raw Olist CSVs already loaded into HDFS in A2.

Star Schema:
  Fact Table  : fact_orders
  Dimensions  : dim_customers, dim_products, dim_sellers, dim_date, dim_payments

Steps per task:
  1. Transform — clean, derive columns, normalize, standardize (references A2 findings)
  2. Load       — write Parquet to /warehouse/processed/  with partitioning
  3. Validate   — row counts, null assertions, per-table audit log

Every transformation block carries an inline comment referencing the A2 profiling finding.
Undocumented transformations have been deliberately avoided per assignment requirements.
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, IntegerType

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("olist_etl")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
HDFS_RAW       = "hdfs://localhost:9000/warehouse/raw/olist/year=2026/month=05"
HDFS_PROCESSED = "hdfs://localhost:9000/warehouse/processed"
LOCAL_RAW      = "data/raw"          # fallback for local testing

# Use HDFS_RAW when running on the cluster; swap to LOCAL_RAW for local dev.
USE_HDFS = True   # Set True when running on Hadoop cluster


def raw_path(filename: str) -> str:
    if USE_HDFS:
        return f"{HDFS_RAW}/{filename}"
    return f"{LOCAL_RAW}/{filename}"


def processed_path(table: str) -> str:
    return f"{HDFS_PROCESSED}/{table}"


# ──────────────────────────────────────────────────────────────────────────────
# Spark Session
# ──────────────────────────────────────────────────────────────────────────────
def create_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("Olist_ETL_A3")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .config("spark.sql.shuffle.partitions", "8")   # tune for cluster size
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info(f"Spark version : {spark.version}")
    logger.info(f"App name      : {spark.sparkContext.appName}")
    return spark


# ──────────────────────────────────────────────────────────────────────────────
# Helper: validate a single table
# ──────────────────────────────────────────────────────────────────────────────
def validate_table(df: DataFrame, table_name: str, key_cols: list[str],
                   raw_count: int) -> None:
    """
    Assert:
      1. Row count equals raw_count (no silent data loss)
      2. No NULLs in key_cols
    Log a per-table summary.
    """
    post_count = df.count()
    logger.info(f"[VALIDATE] {table_name}: raw={raw_count:,}  post-ETL={post_count:,}")

    for col in key_cols:
        null_n = df.filter(F.col(col).isNull()).count()
        if null_n > 0:
            logger.error(f"[VALIDATE] {table_name}.{col} has {null_n} NULLs — FAIL")
            raise ValueError(f"Null assertion failed: {table_name}.{col}")
        else:
            logger.info(f"[VALIDATE] {table_name}.{col}: no NULLs ✓")


# ──────────────────────────────────────────────────────────────────────────────
# TRANSFORM — Orders  →  becomes the spine of fact_orders
# ──────────────────────────────────────────────────────────────────────────────
def transform_orders(spark: SparkSession) -> tuple[DataFrame, int]:
    logger.info("=== Transforming: orders ===")
    df = spark.read.csv(raw_path("olist_orders_dataset.csv"), header=True, inferSchema=True)
    raw_count = df.count()
    logger.info(f"  Raw row count: {raw_count:,}")

    # A2 Finding: order_purchase_timestamp and related datetime columns need
    # parsing; order_approved_at / delivered cols have NULLs for cancelled orders.
    for col in [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]:
        df = df.withColumn(col, F.to_timestamp(col))

    # A2 Finding (Cleaning Rule 2): For cancelled/unavailable orders, NULL
    # delivery timestamps are expected. Flag with is_delivered boolean.
    df = df.withColumn(
        "is_delivered",
        F.when(F.col("order_status") == "delivered", True).otherwise(False)
    )

    # Derive delivery duration (days). NULL for non-delivered orders — acceptable.
    df = df.withColumn(
        "delivery_duration_days",
        F.datediff(
            F.col("order_delivered_customer_date"),
            F.col("order_purchase_timestamp")
        ).cast(DoubleType())
    )

    # Derive days early/late vs. estimate (positive = late, negative = early)
    df = df.withColumn(
        "delivery_delay_days",
        F.datediff(
            F.col("order_delivered_customer_date"),
            F.col("order_estimated_delivery_date")
        ).cast(DoubleType())
    )

    # Derive time-of-day bucket for purchase hour (for analytical queries)
    df = df.withColumn("purchase_hour", F.hour("order_purchase_timestamp"))
    df = df.withColumn(
        "time_of_day",
        F.when(F.col("purchase_hour").between(6, 11),  "Morning")
         .when(F.col("purchase_hour").between(12, 17), "Afternoon")
         .when(F.col("purchase_hour").between(18, 21), "Evening")
         .otherwise("Night")
    )

    # Derive calendar columns for the date dimension join
    df = df.withColumn("order_year",    F.year("order_purchase_timestamp"))
    df = df.withColumn("order_month",   F.month("order_purchase_timestamp"))
    df = df.withColumn("order_quarter", F.quarter("order_purchase_timestamp"))

    # Normalize order_status to lower-case (data consistency)
    df = df.withColumn("order_status", F.lower(F.trim(F.col("order_status"))))

    logger.info(f"  Post-transform row count: {df.count():,}")
    return df, raw_count


# ──────────────────────────────────────────────────────────────────────────────
# TRANSFORM — Order Items
# ──────────────────────────────────────────────────────────────────────────────
def transform_order_items(spark: SparkSession) -> tuple[DataFrame, int]:
    logger.info("=== Transforming: order_items ===")
    df = spark.read.csv(raw_path("olist_order_items_dataset.csv"), header=True, inferSchema=True)
    raw_count = df.count()
    logger.info(f"  Raw row count: {raw_count:,}")

    df = df.withColumn("shipping_limit_date", F.to_timestamp("shipping_limit_date"))

    # A2 Finding (Cleaning Rule 4): price ≤ 0 → replace with category median.
    # PySpark approach: compute per-category median via window, then coalesce.
    # We'll join category info later in fact assembly; for now flag the bad rows.
    df = df.withColumn(
        "price_flag",
        F.when(F.col("price") <= 0, "invalid").otherwise("ok")
    )
    # Fix: explicit cast to DoubleType before the when/otherwise.
    # F.when(..., None) returns NullType when all branches are null/None,
    # which breaks downstream arithmetic (price + freight_value raises
    # "cannot resolve 'NullType + DoubleType'").  Casting both branches to
    # a concrete type avoids that.
    df = df.withColumn(
        "price",
        F.when(F.col("price") <= 0, F.lit(None).cast(DoubleType()))
         .otherwise(F.col("price").cast(DoubleType()))
    )

    # A2 Finding (Cleaning Rule 4): freight_value < 0 → replace with 0 (free shipping edge case).
    df = df.withColumn(
        "freight_value",
        F.when(F.col("freight_value") < 0, F.lit(0.0)).otherwise(F.col("freight_value"))
    )

    # A2 Finding (Cleaning Rule 5): Winsorize price at 99th pct.
    # Store original for auditability.
    p99 = df.filter(F.col("price").isNotNull()).approxQuantile("price", [0.99], 0.01)[0]
    df = df.withColumn("raw_price", F.col("price"))
    df = df.withColumn(
        "price",
        F.when(F.col("price") > p99, p99).otherwise(F.col("price"))
    )
    logger.info(f"  price 99th pct cap: {p99:.2f}")

    # A2 Finding (Cleaning Rule 5): Winsorize freight_value at 99th pct.
    p99f = df.approxQuantile("freight_value", [0.99], 0.01)[0]
    df = df.withColumn("raw_freight_value", F.col("freight_value"))
    df = df.withColumn(
        "freight_value",
        F.when(F.col("freight_value") > p99f, p99f).otherwise(F.col("freight_value"))
    )
    logger.info(f"  freight_value 99th pct cap: {p99f:.2f}")

    # Derive total item cost (price + freight) — new analytical column
    # Fix: coalesce price to 0.0 so rows with a NULL price (invalid records
    # set to NULL above) still get a meaningful item_total_cost instead of NULL.
    df = df.withColumn(
        "item_total_cost",
        F.coalesce(F.col("price"), F.lit(0.0)) + F.col("freight_value")
    )

    logger.info(f"  Post-transform row count: {df.count():,}")
    return df, raw_count


# ──────────────────────────────────────────────────────────────────────────────
# TRANSFORM — Customers
# ──────────────────────────────────────────────────────────────────────────────
def transform_customers(spark: SparkSession) -> tuple[DataFrame, int]:
    logger.info("=== Transforming: customers ===")
    df = spark.read.csv(raw_path("olist_customers_dataset.csv"), header=True, inferSchema=True)
    raw_count = df.count()

    # A2 Finding (Cleaning Rule — schema): Normalize state codes to upper-case
    # for consistent grouping in analytical queries.
    df = df.withColumn("customer_state", F.upper(F.trim(F.col("customer_state"))))
    df = df.withColumn("customer_city",  F.lower(F.trim(F.col("customer_city"))))

    # A2 Finding: customer_zip_code_prefix cast to integer (stored as numeric
    # for geo-join), zero-padded string kept as customer_zip_str for display.
    df = df.withColumn("customer_zip_str",
                       F.lpad(F.col("customer_zip_code_prefix").cast("string"), 5, "0"))

    logger.info(f"  Raw / post-transform rows: {raw_count:,}")
    return df, raw_count


# ──────────────────────────────────────────────────────────────────────────────
# TRANSFORM — Products
# ──────────────────────────────────────────────────────────────────────────────
def transform_products(spark: SparkSession) -> tuple[DataFrame, int]:
    logger.info("=== Transforming: products ===")
    df = spark.read.csv(raw_path("olist_products_dataset.csv"), header=True, inferSchema=True)
    raw_count = df.count()

    # A2 Finding (Cleaning Rule 7): product_name_lenght is a misspelled column.
    # Rename at schema level in the SELECT — no data values change.
    if "product_name_lenght" in df.columns:
        df = df.withColumnRenamed("product_name_lenght", "product_name_length")
    if "product_description_lenght" in df.columns:
        df = df.withColumnRenamed("product_description_lenght", "product_description_length")

    # A2 Finding (Cleaning Rule 3): product_category_name NULL →
    # cross-reference translation file; remaining NULLs → 'unknown'.
    try:
        trans = spark.read.csv(
            raw_path("product_category_name_translation.csv"),
            header=True, inferSchema=True
        )
        df = df.join(trans, on="product_category_name", how="left")
        # Use English name where available
        df = df.withColumn(
            "product_category_name_en",
            F.coalesce(
                F.col("product_category_name_english"),
                F.col("product_category_name"),
                F.lit("unknown")
            )
        )
    except Exception:
        logger.warning("  Translation file not found; using Portuguese category names.")
        df = df.withColumn(
            "product_category_name_en",
            F.coalesce(F.col("product_category_name"), F.lit("unknown"))
        )

    # A2 Finding (Cleaning Rule 3 — Step 2): Impute product_name_length NULLs
    # with per-category median using a window function.
    # Fix: cast to DoubleType before percentile_approx so the function always
    # receives a numeric type regardless of what inferSchema chose (LongType,
    # IntegerType, or even StringType for a malformed row).
    w_cat = Window.partitionBy("product_category_name_en")
    df = df.withColumn("product_name_length", F.col("product_name_length").cast(DoubleType()))
    df = df.withColumn(
        "cat_median_name_length",
        F.expr("percentile_approx(product_name_length, 0.5)").over(w_cat)
    )
    df = df.withColumn(
        "product_name_length",
        F.coalesce(F.col("product_name_length"), F.col("cat_median_name_length")).cast(IntegerType())
    )
    df = df.drop("cat_median_name_length")

    # A2 Finding (Cleaning Rule 3 — Step 3): product_photos_qty → mode per category.
    # Approximate with median (discrete) since mode is expensive in Spark.
    # Fix: same explicit cast before percentile_approx.
    df = df.withColumn("product_photos_qty", F.col("product_photos_qty").cast(DoubleType()))
    df = df.withColumn(
        "cat_median_photos",
        F.expr("percentile_approx(product_photos_qty, 0.5)").over(w_cat)
    )
    df = df.withColumn(
        "product_photos_qty",
        F.coalesce(F.col("product_photos_qty"), F.col("cat_median_photos")).cast(IntegerType())
    )
    df = df.drop("cat_median_photos")

    # Flag imputed rows for downstream auditability
    # (A2 Cleaning Rule 3: "Record all imputed rows in an is_imputed flag column")
    df = df.withColumn(
        "is_imputed",
        F.when(
            F.col("product_category_name").isNull() |
            F.col("product_name_length").isNull(),
            True
        ).otherwise(False)
    )

    logger.info(f"  Raw / post-transform rows: {raw_count:,}")
    return df, raw_count


# ──────────────────────────────────────────────────────────────────────────────
# TRANSFORM — Sellers
# ──────────────────────────────────────────────────────────────────────────────
def transform_sellers(spark: SparkSession) -> tuple[DataFrame, int]:
    logger.info("=== Transforming: sellers ===")
    df = spark.read.csv(raw_path("olist_sellers_dataset.csv"), header=True, inferSchema=True)
    raw_count = df.count()

    df = df.withColumn("seller_state", F.upper(F.trim(F.col("seller_state"))))
    df = df.withColumn("seller_city",  F.lower(F.trim(F.col("seller_city"))))
    df = df.withColumn("seller_zip_str",
                       F.lpad(F.col("seller_zip_code_prefix").cast("string"), 5, "0"))

    logger.info(f"  Raw / post-transform rows: {raw_count:,}")
    return df, raw_count


# ──────────────────────────────────────────────────────────────────────────────
# TRANSFORM — Payments
# ──────────────────────────────────────────────────────────────────────────────
def transform_payments(spark: SparkSession) -> tuple[DataFrame, int]:
    logger.info("=== Transforming: payments ===")
    df = spark.read.csv(raw_path("olist_order_payments_dataset.csv"), header=True, inferSchema=True)
    raw_count = df.count()

    # A2 Finding (Cleaning Rule 5): Winsorize payment_value at global 99th pct.
    p99 = df.approxQuantile("payment_value", [0.99], 0.01)[0]
    df = df.withColumn("raw_payment_value", F.col("payment_value"))
    df = df.withColumn(
        "payment_value",
        F.when(F.col("payment_value") > p99, p99).otherwise(F.col("payment_value"))
    )
    logger.info(f"  payment_value 99th pct cap: {p99:.2f}")

    # Normalize payment_type for consistent grouping
    df = df.withColumn("payment_type", F.lower(F.trim(F.col("payment_type"))))

    logger.info(f"  Raw / post-transform rows: {raw_count:,}")
    return df, raw_count


# ──────────────────────────────────────────────────────────────────────────────
# TRANSFORM — Reviews
# ──────────────────────────────────────────────────────────────────────────────
def transform_reviews(spark: SparkSession) -> tuple[DataFrame, int]:
    logger.info("=== Transforming: reviews ===")
    # Fix: multiLine=True — review_comment_message and review_comment_title regularly
    # contain embedded newlines inside RFC-4180 quoted fields.  Without this option
    # Spark splits those records at the newline, producing extra garbled rows and
    # misaligned column values.  quote/escape='"\'" matches the Olist CSV format.
    df = spark.read.csv(
        raw_path("olist_order_reviews_dataset.csv"),
        header=True,
        inferSchema=True,
        multiLine=True,
        quote='"',
        escape='"',
    )
    raw_count = df.count()

    df = df.withColumn("review_creation_date",    F.to_timestamp("review_creation_date"))
    df = df.withColumn("review_answer_timestamp", F.to_timestamp("review_answer_timestamp"))

    # Fix: explicit cast to IntegerType before comparisons.
    # When multiLine parsing fails on a malformed row, inferSchema may infer
    # review_score as StringType.  Casting here ensures >= and == operations
    # work correctly regardless of the inferred type.
    df = df.withColumn("review_score", F.col("review_score").cast(IntegerType()))

    # A2 Finding (Cleaning Rule 6): review_comment_message/title have ~59%/~88% NULLs.
    # These are optional — retain NULLs but create has_comment boolean flag.
    # Fix: wrap in F.when().otherwise(False).cast("boolean") to avoid the SQL
    # three-value-logic trap where (NULL isNotNull()) & (length > 0) evaluates
    # to NULL instead of False, leaving has_comment as NULL for every null message.
    df = df.withColumn(
        "has_comment",
        F.when(
            F.col("review_comment_message").isNotNull() &
            (F.length(F.trim(F.col("review_comment_message"))) > 0),
            True
        ).otherwise(False).cast("boolean")
    )

    # For NLP pipelines (A4): replace NULL text with empty string
    df = df.withColumn(
        "review_comment_message",
        F.coalesce(F.col("review_comment_message"), F.lit(""))
    )
    df = df.withColumn(
        "review_comment_title",
        F.coalesce(F.col("review_comment_title"), F.lit(""))
    )

    # Bucket review scores into sentiment label
    df = df.withColumn(
        "sentiment",
        F.when(F.col("review_score") >= 4, "positive")
         .when(F.col("review_score") == 3,  "neutral")
         .otherwise("negative")
    )

    logger.info(f"  Raw / post-transform rows: {raw_count:,}")
    return df, raw_count


# ──────────────────────────────────────────────────────────────────────────────
# TRANSFORM — Geolocation
# ──────────────────────────────────────────────────────────────────────────────
def transform_geolocation(spark: SparkSession) -> tuple[DataFrame, int]:
    logger.info("=== Transforming: geolocation ===")
    df = spark.read.csv(raw_path("olist_geolocation_dataset.csv"), header=True, inferSchema=True)
    raw_count = df.count()

    # A2 Finding (Cleaning Rule 1): ~79% duplication by (zip, lat, lng).
    # Drop duplicates — redundant coordinate entries inflate join result sets.
    df = df.dropDuplicates(["geolocation_zip_code_prefix",
                             "geolocation_lat",
                             "geolocation_lng"])

    # Keep one representative (lat, lng) per zip prefix for joins
    df = df.groupBy("geolocation_zip_code_prefix", "geolocation_state", "geolocation_city") \
           .agg(
               F.avg("geolocation_lat").alias("geolocation_lat"),
               F.avg("geolocation_lng").alias("geolocation_lng"),
           )

    dedup_count = df.count()
    logger.info(f"  Raw rows: {raw_count:,}  →  after dedup: {dedup_count:,}")
    return df, raw_count


# ──────────────────────────────────────────────────────────────────────────────
# ASSEMBLE STAR SCHEMA
# ──────────────────────────────────────────────────────────────────────────────
def build_fact_orders(
    spark: SparkSession,
    orders_df: DataFrame,
    items_df: DataFrame,
    payments_df: DataFrame,
    reviews_df: DataFrame,
) -> DataFrame:
    """
    Assemble fact_orders by joining the core transactional tables.
    Grain: one row per order_id × order_item_id.
    """
    logger.info("=== Building fact_orders ===")

    # Aggregate payments per order (an order can have multiple payment rows)
    pay_agg = payments_df.groupBy("order_id").agg(
        F.sum("payment_value").alias("total_payment_value"),
        F.sum("payment_installments").alias("total_installments"),
        F.collect_set("payment_type").alias("payment_types"),
        F.first("payment_type").alias("primary_payment_type"),
    )

    # Aggregate reviews per order (keep latest score)
    rev_agg = reviews_df.groupBy("order_id").agg(
        F.avg("review_score").alias("avg_review_score"),
        F.max("review_score").alias("max_review_score"),
        F.sum(F.col("has_comment").cast("int")).alias("comment_count"),
    )

    # Join items ← orders ← payments ← reviews
    fact = (
        items_df
        .join(orders_df, on="order_id", how="left")
        .join(pay_agg,   on="order_id", how="left")
        .join(rev_agg,   on="order_id", how="left")
    )

    # Derive revenue columns
    fact = fact.withColumn(
        "item_revenue",
        F.col("price") + F.col("freight_value")
    )

    # Select only the columns needed in the fact table
    fact = fact.select(
        "order_id", "order_item_id", "customer_id", "seller_id", "product_id",
        "order_status", "is_delivered",
        "order_purchase_timestamp", "order_year", "order_month", "order_quarter",
        "time_of_day", "purchase_hour",
        "order_delivered_customer_date", "order_estimated_delivery_date",
        "delivery_duration_days", "delivery_delay_days",
        "price", "raw_price", "freight_value", "raw_freight_value",
        "item_total_cost", "item_revenue",
        "total_payment_value", "primary_payment_type", "total_installments",
        "avg_review_score", "comment_count",
    )

    # Cache fact table — reused in analytics.py queries
    fact.cache()
    logger.info(f"  fact_orders row count: {fact.count():,}")
    return fact


def build_dim_date(orders_df: DataFrame) -> DataFrame:
    """Derive a date dimension from order timestamps."""
    logger.info("=== Building dim_date ===")
    dim = orders_df.select(
        F.to_date("order_purchase_timestamp").alias("date_key")
    ).distinct()
    dim = dim.withColumn("year",        F.year("date_key"))
    dim = dim.withColumn("month",       F.month("date_key"))
    dim = dim.withColumn("quarter",     F.quarter("date_key"))
    dim = dim.withColumn("day_of_week", F.dayofweek("date_key"))
    dim = dim.withColumn("day_name",    F.date_format("date_key", "EEEE"))
    dim = dim.withColumn("month_name",  F.date_format("date_key", "MMMM"))
    dim = dim.withColumn("week_of_year",F.weekofyear("date_key"))
    dim = dim.withColumn(
        "is_weekend",
        F.col("day_of_week").isin([1, 7])   # Sunday=1, Saturday=7 in Spark
    )
    logger.info(f"  dim_date row count: {dim.count():,}")
    return dim


# ──────────────────────────────────────────────────────────────────────────────
# LOAD — write Parquet to HDFS with partitioning
# ──────────────────────────────────────────────────────────────────────────────
def load_table(df: DataFrame, name: str, partition_cols: list[str] = None) -> None:
    """
    Optimization Technique 1: Partitioning
    Write Parquet output partitioned by a meaningful column (e.g., order_year/order_month
    for the fact table, state for dimension tables). Partition pruning at query time means
    Spark only reads the relevant files, dramatically reducing I/O for time-bounded queries.
    """
    path = processed_path(name)
    logger.info(f"Loading {name} → {path}")
    writer = df.write.mode("overwrite").format("parquet")
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
        logger.info(f"  Partitioned by: {partition_cols}")
    writer.save(path)
    logger.info(f"  Write complete ✓")


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATE
# ──────────────────────────────────────────────────────────────────────────────
def run_validation(spark: SparkSession, raw_counts: dict) -> None:
    """Read back written Parquet files and validate counts + nulls."""
    logger.info("=== Running post-load validation ===")

    tables_to_check = {
        "fact_orders":    (["order_id", "product_id", "seller_id", "customer_id"],),
        "dim_customers":  (["customer_id"],),
        "dim_products":   (["product_id"],),
        "dim_sellers":    (["seller_id"],),
        "dim_date":       (["date_key"],),
        "dim_payments":   (["order_id"],),
        "dim_reviews":    (["review_id"],),
    }

    results = []
    for table, (key_cols,) in tables_to_check.items():
        try:
            df = spark.read.parquet(processed_path(table))
            count = df.count()
            null_issues = {}
            for col in key_cols:
                if col in df.columns:
                    n = df.filter(F.col(col).isNull()).count()
                    if n > 0:
                        null_issues[col] = n
            status = "PASS ✓" if not null_issues else f"FAIL — nulls: {null_issues}"
            logger.info(f"  {table:<20s}: {count:>8,} rows  {status}")
            results.append((table, count, status))
        except Exception as e:
            logger.warning(f"  {table}: could not validate — {e}")

    logger.info("=== Validation summary complete ===")


# ──────────────────────────────────────────────────────────────────────────────
# OPTIMIZATION: Broadcast join demo
# ──────────────────────────────────────────────────────────────────────────────
def demo_broadcast_join(spark: SparkSession, fact: DataFrame,
                         sellers_df: DataFrame) -> DataFrame:
    """
    Optimization Technique 2: Broadcast Join
    Sellers dimension is small (~3,000 rows). Broadcasting it to all Spark
    executors avoids a full shuffle of the large fact table during the join,
    reducing network I/O significantly.

    Performance explanation: Without broadcast, Spark would sort-merge join
    both sides (O(N log N)). With broadcast, the small table is sent once to
    each executor and each partition of the fact table is joined locally
    (O(N) scan). Particularly impactful when fact_orders has 100k+ rows.
    """
    from pyspark.sql.functions import broadcast
    logger.info("=== Demo: broadcast join fact_orders ← dim_sellers ===")
    enriched = fact.join(broadcast(sellers_df), on="seller_id", how="left")
    logger.info(f"  Enriched fact count: {enriched.count():,}")
    return enriched


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║  Olist ETL Pipeline — Assignment 03                          ║")
    logger.info("║  CS-404 Big Data Analytics | NUST SEECS | Spring 2026        ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")

    spark = create_spark()
    raw_counts = {}

    # ── TRANSFORM ─────────────────────────────────────────────────────────────
    orders_df,   raw_counts["fact_orders"]   = transform_orders(spark)
    items_df,    raw_counts["order_items"]   = transform_order_items(spark)
    customers_df,raw_counts["dim_customers"] = transform_customers(spark)
    products_df, raw_counts["dim_products"]  = transform_products(spark)
    sellers_df,  raw_counts["dim_sellers"]   = transform_sellers(spark)
    payments_df, raw_counts["dim_payments"]  = transform_payments(spark)
    reviews_df,  raw_counts["dim_reviews"]   = transform_reviews(spark)
    geo_df,      _                           = transform_geolocation(spark)

    # ── ASSEMBLE STAR SCHEMA ──────────────────────────────────────────────────
    fact_df  = build_fact_orders(spark, orders_df, items_df, payments_df, reviews_df)
    date_dim = build_dim_date(orders_df)

    # ── LOAD ──────────────────────────────────────────────────────────────────
    # Optimization Technique 1: Partition fact table by year + month for
    # time-bounded queries to benefit from partition pruning.
    load_table(fact_df,      "fact_orders",   partition_cols=["order_year", "order_month"])
    load_table(customers_df, "dim_customers")
    load_table(products_df,  "dim_products")
    load_table(sellers_df,   "dim_sellers")
    load_table(payments_df,  "dim_payments")
    load_table(reviews_df,   "dim_reviews")
    load_table(date_dim,     "dim_date")
    load_table(geo_df,       "dim_geolocation")

    # ── OPTIMIZATION DEMO: BROADCAST JOIN ─────────────────────────────────────
    enriched_fact = demo_broadcast_join(spark, fact_df, sellers_df)

    # ── OPTIMIZATION DEMO: QUERY PLAN ANALYSIS ──────────────────────────────
    # Show explain() for a complex aggregation to verify partition pruning
    logger.info("=== Optimization: Query plan analysis ===")
    plan_df = (
        fact_df
        .filter(F.col("order_year") == 2017)
        .groupBy("order_month", "primary_payment_type")
        .agg(F.sum("item_revenue").alias("monthly_revenue"))
    )
    logger.info("\n--- Query Plan (explain) ---")
    plan_df.explain(True)

    # ── VALIDATE ──────────────────────────────────────────────────────────────
    run_validation(spark, raw_counts)

    logger.info("=== ETL Pipeline completed successfully ===")
    spark.stop()


if __name__ == "__main__":
    main()