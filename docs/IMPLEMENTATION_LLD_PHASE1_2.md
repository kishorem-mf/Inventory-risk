# Implementation LLD: Phase 1 & 2 - Delta Share Provisioning & Feature Engineering

| Document Info | |
|---------------|---|
| **Version** | 1.0 |
| **Last Updated** | 2025-01-29 |
| **Status** | Implementation Ready |
| **Author** | Data Engineering Team |
| **Parent Document** | [LLD_Databricks_ML.md](../LLD_Databricks_ML.md) |

---

## Table of Contents

1. [Overview](#1-overview)
2. [Phase 1: Delta Share Provisioning](#2-phase-1-delta-share-provisioning)
3. [Phase 2: Feature Engineering](#3-phase-2-feature-engineering)
4. [Validation Framework](#4-validation-framework)
5. [Implementation Checklist](#5-implementation-checklist)

---

## 1. Overview

### 1.1 Scope

This document provides implementation details for:
- **Phase 1**: Provisioning and validating source tables from SAP BDC via Delta Sharing
- **Phase 2**: Implementing feature engineering aligned with existing `reasoning_agent_pipeline.py` logic

### 1.2 Design Principle

> **All features must map directly to existing application logic** - No theoretical features. Every feature derives from L1/L2 reasoning rules in `reasoning_agent_pipeline.py`.

### 1.3 Source Systems

| System | Schema | Access Method |
|--------|--------|---------------|
| SAP HANA | `CURRENT_INVT` | Current app (direct) |
| SAP HANA | `INVT_HISTORICAL_DATA` | Current app (direct) |
| SAP BDC | `sap_bdc.inventory_risk` | ML pipeline (Delta Sharing) |

---

## 2. Phase 1: Delta Share Provisioning

### 2.1 Source Table Inventory

#### 2.1.1 Master Data Tables (from CURRENT_INVT)

| Table | Delta Share Name | Row Count | Update Freq | Purpose |
|-------|-----------------|-----------|-------------|---------|
| `STOCK_STATUS_V2` | `stock_status_v2` | ~500K | Weekly | Primary ML source |
| `LOCATION_SOURCE` | `location_source` | ~10K | Static | Transportation lead times |
| `PRODUCTION_SOURCE_HEADER` | `production_source_header` | ~5K | Static | Production lead times |
| `PRODUCT` | `product` | ~5K | Static | Product master |
| `LOCATION` | `location` | ~1K | Static | Location master |

#### 2.1.2 Transactional Tables (from INVT_HISTORICAL_DATA)

| Table | Delta Share Name | Row Count | Update Freq | Purpose |
|-------|-----------------|-----------|-------------|---------|
| `REVIEW_DC_HISTORY` | `review_dc_history` | ~200K | Weekly | DC metrics |
| `REVIEW_PLANT_HISTORY` | `review_plant_history` | ~100K | Weekly | Plant metrics |
| `REVIEW_VENDORS_HISTORY` | `review_vendors_history` | ~50K | Weekly | Vendor supply |
| `REVIEW_CAPACITY_HISTORY` | `review_capacity_history` | ~30K | Weekly | Capacity metrics |
| `REVIEW_COMPONENT_HISTORY` | `review_component_history` | ~80K | Weekly | Component inventory |

### 2.2 Delta Share Configuration

#### 2.2.1 Share Definition

```sql
-- Create share in SAP BDC
CREATE SHARE inventory_risk_ml_source
COMMENT 'Source tables for Inventory Risk ML pipeline';

-- Add tables to share
ALTER SHARE inventory_risk_ml_source ADD TABLE sap_bdc.inventory_risk.stock_status_v2;
ALTER SHARE inventory_risk_ml_source ADD TABLE sap_bdc.inventory_risk.location_source;
ALTER SHARE inventory_risk_ml_source ADD TABLE sap_bdc.inventory_risk.production_source_header;
ALTER SHARE inventory_risk_ml_source ADD TABLE sap_bdc.inventory_risk.review_dc_history;
ALTER SHARE inventory_risk_ml_source ADD TABLE sap_bdc.inventory_risk.review_plant_history;
ALTER SHARE inventory_risk_ml_source ADD TABLE sap_bdc.inventory_risk.review_vendors_history;
ALTER SHARE inventory_risk_ml_source ADD TABLE sap_bdc.inventory_risk.review_capacity_history;

-- Grant access to Databricks recipient
GRANT SELECT ON SHARE inventory_risk_ml_source TO RECIPIENT databricks_ml_workspace;
```

#### 2.2.2 Databricks Access Configuration

```python
# File: notebooks/01_delta_share_setup.py

# Configure Delta Sharing credentials
spark.conf.set("spark.databricks.delta.sharing.profile", "/dbfs/delta-sharing/sap-bdc-profile.json")

# Create catalog reference
spark.sql("""
    CREATE CATALOG IF NOT EXISTS sap_bdc
    USING SHARE `sap-bdc-provider`.inventory_risk_ml_source
""")

# Verify access
spark.sql("SHOW TABLES IN sap_bdc.inventory_risk").show()
```

### 2.3 Table Access Code

```python
# File: notebooks/02_data_access.py

from pyspark.sql import SparkSession

def load_source_tables():
    """Load all source tables from Delta Share."""

    tables = {
        # Primary table
        "stock_status": spark.read.table("sap_bdc.inventory_risk.stock_status_v2"),

        # Master tables
        "location_source": spark.read.table("sap_bdc.inventory_risk.location_source"),
        "production_source": spark.read.table("sap_bdc.inventory_risk.production_source_header"),

        # History tables (for capacity and vendor features)
        "review_capacity": spark.read.table("sap_bdc.inventory_risk.review_capacity_history"),
        "review_vendors": spark.read.table("sap_bdc.inventory_risk.review_vendors_history"),
    }

    return tables
```

### 2.4 Phase 1 Validation

#### 2.4.1 Schema Validation

```python
# File: notebooks/03_schema_validation.py

EXPECTED_STOCK_STATUS_COLUMNS = [
    "product_id", "location_id", "week_num", "year",
    "projected_stock", "safety_stock", "stock_on_hand",
    "incoming_receipts", "total_demand", "outgoing_supply",
    "supply_orders", "offset_stock", "stock_condition",
    "stock_status_warning", "location_type",
    "lag_1_dependent_demand", "lag_1_supply_orders", "lag_1_incoming_receipts",
    "transportation_lead_time", "production_lead_time"
]

def validate_schema(df, expected_columns, table_name):
    """Validate DataFrame has expected columns."""
    actual_columns = set(df.columns)
    expected_set = set(expected_columns)

    missing = expected_set - actual_columns
    if missing:
        raise ValueError(f"{table_name}: Missing columns: {missing}")

    print(f"✅ {table_name}: Schema validation passed ({len(expected_columns)} columns)")
    return True
```

#### 2.4.2 Data Quality Validation

```python
# File: notebooks/03_schema_validation.py (continued)

from pyspark.sql import functions as F

def validate_data_quality(tables):
    """Run data quality checks on source tables."""

    results = {}

    # STOCK_STATUS_V2 checks
    stock = tables["stock_status"]
    results["stock_status"] = {
        "row_count": stock.count(),
        "null_product_id": stock.filter(F.col("product_id").isNull()).count(),
        "null_location_id": stock.filter(F.col("location_id").isNull()).count(),
        "null_safety_stock": stock.filter(F.col("safety_stock").isNull()).count(),
        "negative_safety_stock": stock.filter(F.col("safety_stock") < 0).count(),
        "distinct_weeks": stock.select("year", "week_num").distinct().count(),
        "location_types": stock.select("location_type").distinct().collect(),
    }

    # Validation rules
    assert results["stock_status"]["row_count"] > 0, "STOCK_STATUS_V2 is empty"
    assert results["stock_status"]["null_product_id"] == 0, "Null product_id found"
    assert results["stock_status"]["null_location_id"] == 0, "Null location_id found"
    assert results["stock_status"]["negative_safety_stock"] == 0, "Negative safety_stock found"

    print("✅ Data quality validation passed")
    return results
```

#### 2.4.3 Freshness Validation

```python
def validate_freshness(tables, max_lag_days=7):
    """Ensure data is recent enough for ML training."""

    stock = tables["stock_status"]

    # Get latest week in data
    latest = stock.agg(
        F.max(F.struct("year", "week_num")).alias("latest_week")
    ).collect()[0]["latest_week"]

    # Compare to current week
    from datetime import datetime
    current_week = datetime.now().isocalendar()[1]
    current_year = datetime.now().year

    if latest["year"] < current_year or \
       (latest["year"] == current_year and latest["week_num"] < current_week - 1):
        raise ValueError(f"Data is stale. Latest: {latest}, Current: {current_year}-W{current_week}")

    print(f"✅ Freshness validation passed. Latest data: {latest['year']}-W{latest['week_num']}")
    return True
```

---

## 3. Phase 2: Feature Engineering

### 3.1 Feature Mapping to Application Logic

> **Source**: `reasoning_agent_pipeline.py` L1/L2 reasoning rules

#### 3.1.1 Core Risk Features

| Feature | Formula | L1/L2 Mapping | Code Reference |
|---------|---------|---------------|----------------|
| `stock_to_safety_ratio` | `projected_stock / safety_stock` | All L1 detection | Line 1006: "stock > safety_stock" |
| `soh_to_safety_ratio` | `stock_on_hand / safety_stock` | Overstock L1: "High Stock on Hand at Start" | Line 1010 |
| `offset_stock` | `projected_stock - safety_stock` | Already in table | - |
| `consecutive_deficit_weeks` | Streak where condition='deficit' | 4+ weeks triggers understock | Line 64-88 |
| `consecutive_excess_weeks` | Streak where condition='excess' | 4+ weeks triggers overstock | Line 64-88 |

#### 3.1.2 Demand/Forecast Features

| Feature | Formula | L1/L2 Mapping | Code Reference |
|---------|---------|---------------|----------------|
| `forecast_accuracy_ratio` | `lag_1_dependent_demand / total_demand` | Overstock L1: "Overforecasting", Understock L1: "Underforecasting" | Line 1015, 1180 |
| `demand_wow_change` | `(total_demand - lag_1_dependent_demand) / lag_1_dependent_demand` | Understock L2: "Demand Spike" | Line 1185 |
| `demand_spike_flag` | `1 if total_demand > lag_1 * 1.5` | Understock L1: Demand spike detection | Line 1190 |

#### 3.1.3 Supply Features

| Feature | Formula | L1/L2 Mapping | Code Reference |
|---------|---------|---------------|----------------|
| `receipts_to_demand_ratio` | `incoming_receipts / total_demand` | Overstock L1: "Larger Lot Size" | Line 1012 |
| `zero_receipt_weeks_flag` | `1 if incoming_receipts == 0` | Understock L1: "Longer Lead Time" | Line 1175 |
| `supply_order_change` | `lag_1_supply_orders - supply_orders` | Understock L1: "Delayed Purchase Orders" | Line 1195 |
| `supply_reliability_idx` | `incoming_receipts / lag_1_supply_orders` | Supply reliability measure | Line 1200 |

#### 3.1.4 Lead Time Features

| Feature | Formula | L1/L2 Mapping | Code Reference |
|---------|---------|---------------|----------------|
| `total_lead_time` | `transportation_lead_time + production_lead_time` | BOM tracing logic | Line 758-773 |
| `long_transport_lead_flag` | `1 if transportation_lead_time > 3` | Understock L2: "Longer Transportation Lead" | Line 1205 |
| `long_production_lead_flag` | `1 if production_lead_time > 3` | Understock L2: "Longer Production Lead" | Line 1210 |

#### 3.1.5 Capacity Features (from REVIEW_CAPACITY_HISTORY)

| Feature | Formula | L1/L2 Mapping | Code Reference |
|---------|---------|---------------|----------------|
| `zero_capacity_weeks` | Count where capacity_usage == 0 | Understock L1: "Production Delays" | Line 850: gather_capacity_data() |
| `capacity_utilization_avg` | AVG(capacity_usage) over 4 weeks | Production efficiency indicator | Line 855 |

#### 3.1.6 Vendor Features (from REVIEW_VENDORS_HISTORY)

| Feature | Formula | L1/L2 Mapping | Code Reference |
|---------|---------|---------------|----------------|
| `vendor_supply_flag` | `1 if vendor_max_receipt > 0` | Understock L1: "Supplier Delays" | Line 830: gather_vendor_direct_data() |
| `vendor_zero_supply_weeks` | Count where vendor_supply == 0 | Vendor reliability | Line 835 |

#### 3.1.7 Location Features

| Feature | Formula | L1/L2 Mapping | Code Reference |
|---------|---------|---------------|----------------|
| `is_dc` | `1 if location_type == 'DC'` | Location-specific L1 reasons | Line 70 |
| `is_rdc` | `1 if location_type == 'RDC'` | RDCs have transport, no production | Line 70 |
| `is_plant` | `1 if location_type in ['PL', 'P']` | Plants have production L1 reasons | Line 70 |

### 3.2 Feature Engineering Implementation

```python
# File: notebooks/04_feature_engineering.py

from pyspark.sql import functions as F
from pyspark.sql.window import Window

def calculate_consecutive_weeks(df, condition_col, condition_value, output_col):
    """
    Calculate streak of consecutive weeks meeting a condition.
    Maps to: 4+ weeks rule for understock/overstock detection.
    Source: reasoning_agent_pipeline.py Line 64-88
    """
    w = Window.partitionBy("product_id", "location_id").orderBy("year", "week_num")

    return df \
        .withColumn("_cond", F.when(F.col(condition_col) == condition_value, 1).otherwise(0)) \
        .withColumn("_grp", F.sum(F.when(F.col("_cond") == 0, 1).otherwise(0)).over(w)) \
        .withColumn(output_col, F.when(
            F.col("_cond") == 1,
            F.row_number().over(
                Window.partitionBy("product_id", "location_id", "_grp")
                .orderBy("year", "week_num")
            )
        ).otherwise(0)) \
        .drop("_cond", "_grp")


def generate_features(stock_df, location_source_df, capacity_df=None, vendor_df=None):
    """
    Generate ML features aligned with reasoning_agent_pipeline.py L1/L2 logic.

    All features are derived from existing application logic - no theoretical additions.
    """

    w = Window.partitionBy("product_id", "location_id").orderBy("year", "week_num")

    # Join lead time from LOCATION_SOURCE
    df = stock_df.join(
        location_source_df.select("location_id", "product_id", "transportation_lead_time"),
        on=["location_id", "product_id"],
        how="left"
    )

    # ========== CORE RISK FEATURES ==========
    # Source: Basic detection logic
    df = df \
        .withColumn("stock_to_safety_ratio",
            F.when(F.col("safety_stock") > 0,
                   F.col("projected_stock") / F.col("safety_stock"))
            .otherwise(F.lit(1.0))) \
        .withColumn("soh_to_safety_ratio",
            F.when(F.col("safety_stock") > 0,
                   F.col("stock_on_hand") / F.col("safety_stock"))
            .otherwise(F.lit(1.0)))

    # ========== DEMAND/FORECAST FEATURES ==========
    # Source: Overstock L1 "Overforecasting", Understock L1 "Underforecasting"
    df = df \
        .withColumn("forecast_accuracy_ratio",
            F.when(F.col("total_demand") > 0,
                   F.col("lag_1_dependent_demand") / F.col("total_demand"))
            .otherwise(F.lit(1.0))) \
        .withColumn("demand_wow_change",
            F.when(F.col("lag_1_dependent_demand") > 0,
                   (F.col("total_demand") - F.col("lag_1_dependent_demand")) /
                   F.col("lag_1_dependent_demand"))
            .otherwise(F.lit(0.0))) \
        .withColumn("demand_spike_flag",
            F.when(F.col("total_demand") > F.col("lag_1_dependent_demand") * 1.5, 1)
            .otherwise(0))

    # ========== SUPPLY FEATURES ==========
    # Source: Overstock L1 "Larger Lot Size", Understock L1 "Delayed POs"
    df = df \
        .withColumn("receipts_to_demand_ratio",
            F.when(F.col("total_demand") > 0,
                   F.col("incoming_receipts") / F.col("total_demand"))
            .otherwise(F.lit(1.0))) \
        .withColumn("zero_receipt_weeks_flag",
            F.when(F.col("incoming_receipts") == 0, 1).otherwise(0)) \
        .withColumn("supply_order_change",
            F.col("lag_1_supply_orders") - F.col("supply_orders")) \
        .withColumn("supply_reliability_idx",
            F.when(F.col("lag_1_supply_orders") > 0,
                   F.col("incoming_receipts") / F.col("lag_1_supply_orders"))
            .otherwise(F.lit(1.0)))

    # ========== LEAD TIME FEATURES ==========
    # Source: BOM tracing, Understock L2 "Longer Lead Time"
    df = df \
        .withColumn("total_lead_time",
            F.coalesce(F.col("transportation_lead_time"), F.lit(0)) +
            F.coalesce(F.col("production_lead_time"), F.lit(0))) \
        .withColumn("long_transport_lead_flag",
            F.when(F.col("transportation_lead_time") > 3, 1).otherwise(0)) \
        .withColumn("long_production_lead_flag",
            F.when(F.col("production_lead_time") > 3, 1).otherwise(0))

    # ========== PATTERN FEATURES ==========
    # Source: 4+ consecutive weeks rule
    df = calculate_consecutive_weeks(df, "stock_condition", "deficit", "consecutive_deficit_weeks")
    df = calculate_consecutive_weeks(df, "stock_condition", "excess", "consecutive_excess_weeks")

    # ========== LOCATION FEATURES ==========
    # Source: Location-specific L1 reasons
    df = df \
        .withColumn("is_dc", F.when(F.col("location_type") == "DC", 1).otherwise(0)) \
        .withColumn("is_rdc", F.when(F.col("location_type") == "RDC", 1).otherwise(0)) \
        .withColumn("is_plant", F.when(F.col("location_type").isin(["PL", "P"]), 1).otherwise(0))

    # ========== CAPACITY FEATURES (Optional) ==========
    # Source: gather_capacity_data() - Understock L1 "Production Delays"
    if capacity_df is not None:
        cap_agg = capacity_df.groupBy("product_id", "location_id", "year", "week_num").agg(
            F.sum(F.when(F.col("capacity_usage") == 0, 1).otherwise(0)).alias("zero_capacity_weeks"),
            F.avg("capacity_usage").alias("capacity_utilization_avg")
        )
        df = df.join(cap_agg, on=["product_id", "location_id", "year", "week_num"], how="left")
        df = df.fillna(0, subset=["zero_capacity_weeks", "capacity_utilization_avg"])

    # ========== VENDOR FEATURES (Optional) ==========
    # Source: gather_vendor_direct_data() - Understock L1 "Supplier Delays"
    if vendor_df is not None:
        ven_agg = vendor_df.groupBy("product_id", "location_id", "year", "week_num").agg(
            F.max(F.when(F.col("vendor_max_external_receipt") > 0, 1).otherwise(0)).alias("vendor_supply_flag"),
            F.sum(F.when(F.col("vendor_max_external_receipt") == 0, 1).otherwise(0)).alias("vendor_zero_supply_weeks")
        )
        df = df.join(ven_agg, on=["product_id", "location_id", "year", "week_num"], how="left")
        df = df.fillna(0, subset=["vendor_supply_flag", "vendor_zero_supply_weeks"])

    # ========== TARGET VARIABLE ==========
    df = df.withColumn("risk_label",
        F.when(F.col("stock_status_warning").contains("understock"), 1)
        .when(F.col("stock_status_warning").contains("overstock"), 2)
        .otherwise(0))

    return df


# Feature columns list
FEATURE_COLS = [
    # Core Risk (5)
    "stock_to_safety_ratio", "soh_to_safety_ratio", "offset_stock",
    "consecutive_deficit_weeks", "consecutive_excess_weeks",
    # Demand/Forecast (3)
    "forecast_accuracy_ratio", "demand_wow_change", "demand_spike_flag",
    # Supply (4)
    "receipts_to_demand_ratio", "zero_receipt_weeks_flag",
    "supply_order_change", "supply_reliability_idx",
    # Lead Time (3)
    "total_lead_time", "long_transport_lead_flag", "long_production_lead_flag",
    # Capacity (2)
    "zero_capacity_weeks", "capacity_utilization_avg",
    # Vendor (2)
    "vendor_supply_flag", "vendor_zero_supply_weeks",
    # Location (3)
    "is_dc", "is_rdc", "is_plant"
]

PRIMARY_KEYS = ["product_id", "location_id", "year", "week_num"]
```

### 3.3 Feature Store Registration

```python
# File: notebooks/05_feature_store_setup.py

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Create feature table
fe.create_table(
    name="sap_bdc.inventory_risk.ml_features",
    primary_keys=["product_id", "location_id", "year", "week_num"],
    timestamp_keys=["feature_timestamp"],
    description="Inventory risk ML features aligned with reasoning_agent_pipeline.py L1/L2 logic",
    tags={
        "team": "data-engineering",
        "domain": "supply-chain",
        "source": "reasoning_agent_pipeline.py"
    }
)

# Write features
fe.write_table(
    name="sap_bdc.inventory_risk.ml_features",
    df=features_df.withColumn("feature_timestamp", F.current_timestamp()),
    mode="merge"
)
```

---

## 4. Validation Framework

### 4.1 Feature Validation Rules

| Feature | Valid Range | Null Handling | Validation Rule |
|---------|-------------|---------------|-----------------|
| `stock_to_safety_ratio` | [0, 10] | Fill with 1.0 | Cap at boundaries |
| `soh_to_safety_ratio` | [0, 10] | Fill with 1.0 | Cap at boundaries |
| `forecast_accuracy_ratio` | [0, 5] | Fill with 1.0 | Cap at boundaries |
| `consecutive_deficit_weeks` | [0, 52] | Fill with 0 | Integer only |
| `consecutive_excess_weeks` | [0, 52] | Fill with 0 | Integer only |
| `total_lead_time` | [0, 26] | Fill with median | Cap at 26 weeks |
| `demand_spike_flag` | {0, 1} | Fill with 0 | Binary only |

### 4.2 Feature Validation Code

```python
# File: notebooks/06_feature_validation.py

def validate_features(df):
    """Validate and clean feature values."""

    # Range capping
    df = df \
        .withColumn("stock_to_safety_ratio",
            F.when(F.col("stock_to_safety_ratio").isNull(), 1.0)
            .when(F.col("stock_to_safety_ratio") > 10, 10.0)
            .when(F.col("stock_to_safety_ratio") < 0, 0.0)
            .otherwise(F.col("stock_to_safety_ratio"))) \
        .withColumn("soh_to_safety_ratio",
            F.when(F.col("soh_to_safety_ratio").isNull(), 1.0)
            .when(F.col("soh_to_safety_ratio") > 10, 10.0)
            .when(F.col("soh_to_safety_ratio") < 0, 0.0)
            .otherwise(F.col("soh_to_safety_ratio"))) \
        .withColumn("forecast_accuracy_ratio",
            F.when(F.col("forecast_accuracy_ratio").isNull(), 1.0)
            .when(F.col("forecast_accuracy_ratio") > 5, 5.0)
            .when(F.col("forecast_accuracy_ratio") < 0, 0.0)
            .otherwise(F.col("forecast_accuracy_ratio")))

    # Fill nulls for streak features
    df = df.fillna(0, subset=[
        "consecutive_deficit_weeks", "consecutive_excess_weeks",
        "zero_capacity_weeks", "vendor_zero_supply_weeks"
    ])

    # Fill nulls for flag features
    df = df.fillna(0, subset=[
        "demand_spike_flag", "zero_receipt_weeks_flag",
        "long_transport_lead_flag", "long_production_lead_flag",
        "vendor_supply_flag", "is_dc", "is_rdc", "is_plant"
    ])

    return df


def run_feature_quality_checks(df):
    """Run quality checks on generated features."""

    checks = {}

    # Null checks
    for col in FEATURE_COLS:
        null_count = df.filter(F.col(col).isNull()).count()
        checks[f"null_{col}"] = null_count

    # Range checks
    checks["ratio_out_of_range"] = df.filter(
        (F.col("stock_to_safety_ratio") < 0) | (F.col("stock_to_safety_ratio") > 10)
    ).count()

    checks["streak_negative"] = df.filter(
        (F.col("consecutive_deficit_weeks") < 0) | (F.col("consecutive_excess_weeks") < 0)
    ).count()

    # Distribution checks
    checks["risk_label_distribution"] = df.groupBy("risk_label").count().collect()

    # Validate all checks pass
    total_nulls = sum(v for k, v in checks.items() if k.startswith("null_"))
    assert total_nulls == 0, f"Features contain {total_nulls} null values"
    assert checks["ratio_out_of_range"] == 0, "Ratio features out of range"
    assert checks["streak_negative"] == 0, "Negative streak values found"

    print("✅ Feature quality checks passed")
    return checks
```

### 4.3 L1/L2 Alignment Validation

```python
# File: notebooks/07_alignment_validation.py

def validate_l1_l2_alignment(features_df, stock_status_df):
    """
    Validate that ML features correctly identify L1/L2 conditions.
    Compare ML feature flags with existing stock_status_warning labels.
    """

    # Join features with original labels
    validation_df = features_df.join(
        stock_status_df.select("product_id", "location_id", "year", "week_num", "stock_status_warning"),
        on=["product_id", "location_id", "year", "week_num"]
    )

    # Check: consecutive_deficit_weeks >= 4 should align with understock warnings
    understock_check = validation_df.filter(
        F.col("stock_status_warning").contains("understock")
    ).agg(
        F.avg(F.when(F.col("consecutive_deficit_weeks") >= 4, 1).otherwise(0)).alias("alignment_rate")
    ).collect()[0]["alignment_rate"]

    # Check: consecutive_excess_weeks >= 4 should align with overstock warnings
    overstock_check = validation_df.filter(
        F.col("stock_status_warning").contains("overstock")
    ).agg(
        F.avg(F.when(F.col("consecutive_excess_weeks") >= 4, 1).otherwise(0)).alias("alignment_rate")
    ).collect()[0]["alignment_rate"]

    print(f"Understock alignment rate: {understock_check:.2%}")
    print(f"Overstock alignment rate: {overstock_check:.2%}")

    # Threshold: at least 90% alignment expected
    assert understock_check >= 0.90, f"Understock alignment too low: {understock_check:.2%}"
    assert overstock_check >= 0.90, f"Overstock alignment too low: {overstock_check:.2%}"

    print("✅ L1/L2 alignment validation passed")
    return {"understock": understock_check, "overstock": overstock_check}
```

---

## 5. Implementation Checklist

### 5.1 Phase 1: Delta Share Provisioning

| # | Task | Owner | Status | Validation |
|---|------|-------|--------|------------|
| 1.1 | Create Delta Share in SAP BDC | Data Eng | 🔲 | Share visible in catalog |
| 1.2 | Add source tables to share | Data Eng | 🔲 | All 7 tables added |
| 1.3 | Configure Databricks recipient | Data Eng | 🔲 | Access granted |
| 1.4 | Create catalog reference in Databricks | Data Eng | 🔲 | `SHOW TABLES` works |
| 1.5 | Run schema validation | Data Eng | 🔲 | All columns present |
| 1.6 | Run data quality validation | Data Eng | 🔲 | No nulls in PKs |
| 1.7 | Run freshness validation | Data Eng | 🔲 | Data < 7 days old |

### 5.2 Phase 2: Feature Engineering

| # | Task | Owner | Status | Validation |
|---|------|-------|--------|------------|
| 2.1 | Create `04_feature_engineering.py` notebook | Data Eng | 🔲 | Notebook runs |
| 2.2 | Implement `calculate_consecutive_weeks()` | Data Eng | 🔲 | Unit test passes |
| 2.3 | Implement `generate_features()` | Data Eng | 🔲 | 22 features created |
| 2.4 | Join LOCATION_SOURCE for lead times | Data Eng | 🔲 | No null lead times |
| 2.5 | Join REVIEW_CAPACITY_HISTORY | Data Eng | 🔲 | Capacity features populated |
| 2.6 | Join REVIEW_VENDORS_HISTORY | Data Eng | 🔲 | Vendor features populated |
| 2.7 | Implement `validate_features()` | Data Eng | 🔲 | All ranges valid |
| 2.8 | Run feature quality checks | Data Eng | 🔲 | 0 nulls, valid ranges |
| 2.9 | Run L1/L2 alignment validation | Data Eng | 🔲 | ≥90% alignment |
| 2.10 | Create Feature Store table | Data Eng | 🔲 | Table in catalog |
| 2.11 | Write features to Feature Store | Data Eng | 🔲 | ~500K rows written |

### 5.3 Sign-Off Criteria

| Criteria | Threshold | Status |
|----------|-----------|--------|
| All source tables accessible via Delta Share | 7/7 tables | 🔲 |
| Schema validation passes | 100% columns match | 🔲 |
| Data freshness | < 7 days old | 🔲 |
| Feature null rate | 0% | 🔲 |
| Feature range compliance | 100% | 🔲 |
| L1/L2 alignment (understock) | ≥ 90% | 🔲 |
| L1/L2 alignment (overstock) | ≥ 90% | 🔲 |
| Feature Store table created | Yes | 🔲 |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-29 | Data Eng | Initial implementation LLD for Phase 1 & 2 |
