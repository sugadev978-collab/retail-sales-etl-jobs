import sys

from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.sql.functions import col, lit, current_timestamp, to_date
from pyspark.sql.functions import year, month
from pyspark.sql.types import DoubleType, IntegerType


# =========================
# PARAMETERS
# =========================

args = getResolvedOptions(
    sys.argv,
    ["JOB_NAME", "SOURCE_BUCKET"]
)

source_bucket = args["SOURCE_BUCKET"]

# =========================
# SPARK SESSION
# =========================

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

spark.conf.set("spark.sql.shuffle.partitions", "8")

print("===== JOB STARTED =====")


# =========================
# S3 PATHS
# =========================

customers_path    = f"s3://{source_bucket}/Customers/"
products_path     = f"s3://{source_bucket}/Products/"
stores_path       = f"s3://{source_bucket}/Stores/"
transactions_path = f"s3://{source_bucket}/Transactions/"

trusted_path  = f"s3://{source_bucket}/trusted/retail_sales/"
rejected_path = f"s3://{source_bucket}/rejected/retail_sales/"


print("Reading data from S3...")

# =========================
# READ DATA
# =========================

customers_df    = spark.read.option("header", True).option("inferSchema", True).csv(customers_path)
products_df     = spark.read.option("header", True).option("inferSchema", True).csv(products_path)
stores_df       = spark.read.option("header", True).option("inferSchema", True).csv(stores_path)
transactions_df = spark.read.option("header", True).option("inferSchema", True).csv(transactions_path)

print("Data Loaded")

print("Customers:", customers_df.count())
print("Products:", products_df.count())
print("Stores:", stores_df.count())
print("Transactions:", transactions_df.count())


# =========================
# REMOVE DUPLICATES
# =========================

customers_df    = customers_df.dropDuplicates(["customer_id"])
products_df     = products_df.dropDuplicates(["product_id"])
stores_df       = stores_df.dropDuplicates(["store_id"])
transactions_df = transactions_df.dropDuplicates(["transaction_id"])


# =========================
# TYPE CASTING
# =========================

transactions_df = transactions_df.withColumn("quantity", col("quantity").cast(IntegerType()))
transactions_df = transactions_df.withColumn("price", col("price").cast(DoubleType()))
transactions_df = transactions_df.withColumn("transaction_date", to_date(col("transaction_date")))


# =========================
# BASIC VALIDATION
# =========================

valid_df = transactions_df.filter(
    (col("transaction_id").isNotNull()) &
    (col("customer_id").isNotNull()) &
    (col("product_id").isNotNull()) &
    (col("store_id").isNotNull()) &
    (col("quantity") > 0) &
    (col("price") > 0) &
    (col("transaction_date").isNotNull())
)

rejected_basic_df = transactions_df.filter(
    (col("transaction_id").isNull()) |
    (col("customer_id").isNull()) |
    (col("product_id").isNull()) |
    (col("store_id").isNull()) |
    (col("quantity") <= 0) |
    (col("price") <= 0) |
    (col("transaction_date").isNull())
).withColumn("rejection_reason", lit("Basic validation failed"))


# =========================
# JOIN DIMENSIONS
# =========================

# Rename price in products to avoid ambiguity with transactions price
products_df = products_df.withColumnRenamed("price", "product_price")

joined_df = valid_df \
    .join(customers_df, "customer_id", "left") \
    .join(products_df, "product_id", "left") \
    .join(stores_df, "store_id", "left")


# =========================
# FINAL VALIDATION
# =========================

final_valid_df = joined_df.filter(
    (col("customer_name").isNotNull()) &
    (col("product_name").isNotNull()) &
    (col("store_region").isNotNull())
)

rejected_join_df = joined_df.filter(
    (col("customer_name").isNull()) |
    (col("product_name").isNull()) |
    (col("store_region").isNull())
).withColumn("rejection_reason", lit("Dimension lookup failed"))


# =========================
# FINAL TRANSFORM
# =========================

final_df = final_valid_df \
    .withColumn("revenue", col("quantity") * col("price")) \
    .withColumn("order_year", year(col("transaction_date"))) \
    .withColumn("order_month", month(col("transaction_date"))) \
    .withColumn("etl_inserted_time", current_timestamp()) \
    .select(
        "transaction_id",
        "customer_id",
        "customer_name",
        "city",
        "product_id",
        "product_name",
        "category",
        "product_price",
        "store_id",
        "store_region",
        "quantity",
        "price",
        "revenue",
        "transaction_date",
        "order_year",
        "order_month",
        "etl_inserted_time"
    )


# =========================
# REJECTED FINAL
# =========================

rejected_final_df = rejected_basic_df.select(
    "transaction_id",
    "customer_id",
    "product_id",
    "store_id",
    "quantity",
    "price",
    "transaction_date",
    "rejection_reason"
).unionByName(
    rejected_join_df.select(
        "transaction_id",
        "customer_id",
        "product_id",
        "store_id",
        "quantity",
        "price",
        "transaction_date",
        "rejection_reason"
    )
)


# =========================
# COUNTS
# =========================

print("Valid records:", final_df.count())
print("Rejected records:", rejected_final_df.count())


# =========================
# WRITE OUTPUT
# =========================

print("Writing Trusted Data...")

final_df.write \
    .mode("overwrite") \
    .partitionBy("order_year", "order_month") \
    .parquet(trusted_path)

print("Writing Rejected Data...")

rejected_final_df.write \
    .mode("overwrite") \
    .parquet(rejected_path)

print("===== JOB COMPLETED SUCCESSFULLY =====")

job.commit()
#changes done