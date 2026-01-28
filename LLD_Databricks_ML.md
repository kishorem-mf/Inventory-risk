# Low-Level Design: Inventory Risk ML on Databricks

| Document Info | |
|---------------|---|
| **Version** | 1.1 |
| **Last Updated** | 2025-01-28 |
| **Status** | Draft |
| **Author** | ML Engineering Team |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Context](#2-system-context)
3. [Source Data Specification](#3-source-data-specification)
4. [Phase 2: Feature Engineering](#4-phase-2-feature-engineering)
5. [Phase 3: ML Models](#5-phase-3-ml-models)
6. [Phase 4: Write-Back to SAP BDC](#6-phase-4-write-back-to-sap-bdc)
7. [Phase 5: Model Serving & Integration](#7-phase-5-model-serving--integration)
8. [MLOps & Orchestration](#8-mlops--orchestration)
9. [Monitoring & Alerting](#9-monitoring--alerting)
10. [Testing Strategy](#10-testing-strategy)
11. [Appendix](#11-appendix)

---

## 1. Executive Summary

### 1.1 Objective

Enhance the existing Inventory Risk Explainability application with ML-powered capabilities:
- **Predictive Risk Classification**: Classify product-location combinations as understock/overstock/normal
- **Severity Scoring**: Quantify risk severity (0-100) for prioritization
- **Early Warning**: Predict risk occurrence 4 weeks ahead
- **Explainability**: SHAP-based feature importance for root cause transparency

### 1.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                 SAP BDC (Source)                                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │stock_status │ │  review_dc  │ │review_plant │ │review_vendor│ │ lag_tables  │   │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘   │
└─────────┼───────────────┼───────────────┼───────────────┼───────────────┼──────────┘
          │               │               │               │               │
          └───────────────┴───────────────┴───────────────┴───────────────┘
                                          │
                                Delta Sharing (zero-copy read)
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            SAP DATABRICKS (ML Processing)                            │
│                                                                                      │
│  ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐        │
│  │  FEATURE STORE     │    │   MODEL REGISTRY   │    │  MODEL SERVING     │        │
│  │  ───────────────   │    │  ───────────────   │    │  ───────────────   │        │
│  │  ml_features       │───►│  XGBoost Classifier│───►│  /invocations      │        │
│  │  (Delta Table)     │    │  LSTM Early Warning│    │  (REST API)        │        │
│  │                    │    │  LightGBM Severity │    │                    │        │
│  └────────────────────┘    └────────────────────┘    └────────────────────┘        │
│            │                         │                         │                    │
│            ▼                         ▼                         │                    │
│  ┌─────────────────────────────────────────────┐               │                    │
│  │           BATCH SCORING JOB                  │               │                    │
│  │  ml_features → models → ml_predictions       │               │                    │
│  └─────────────────────────────────────────────┘               │                    │
│                          │                                      │                    │
└──────────────────────────┼──────────────────────────────────────┼────────────────────┘
                           │                                      │
                  Delta Sharing (write-back)              REST API Call
                           │                                      │
                           ▼                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              SAP BDC (ML Results)                                    │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  ml_predictions (Data Product)                                               │   │
│  │  - risk_class, risk_probability, severity_score, early_warning_weeks        │   │
│  │  - top_factor_1, top_factor_2, top_factor_3 (SHAP-based)                    │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         SAP BTP (Existing Application)                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  Flask API (main.py)                                                         │   │
│  │  - Supervisor Agent → Information Retrieval / Reasoning Agent               │   │
│  │  - NEW: ML Prediction Integration                                            │   │
│  │  - Enhanced response with severity_score + top_factors                      │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Project Timeline

| Phase | Deliverable | Duration | Status |
|-------|-------------|----------|--------|
| 1 | Data Layer (Delta Sharing from SAP BDC) | - | ✅ Done |
| 2 | Feature Engineering + Feature Store | 1 week | 🔲 Pending |
| 3 | ML Models (XGBoost + LSTM + LightGBM) | 2 weeks | 🔲 Pending |
| 4 | Write-Back to SAP BDC as Data Product | 1 week | 🔲 Pending |
| 5 | Model Serving + Integration | 1 week | 🔲 Pending |

**Total Remaining: ~5 weeks**

---

## 2. System Context

### 2.1 Current Application (As-Is)

The existing SAP BTP Flask application provides:

| Component | Function |
|-----------|----------|
| **Supervisor Agent** | Routes queries to appropriate sub-agents |
| **Information Retrieval Agent** | Converts NL to SQL, queries SAP HANA |
| **Reasoning Agent** | Analyzes understock/overstock root causes |
| **Memory Retention** | Session and long-term conversation storage |

**Current Limitation**: Risk detection is rule-based (4+ consecutive weeks of deficit/excess).

### 2.2 Enhanced Application (To-Be)

| Enhancement | Benefit |
|-------------|---------|
| **ML Risk Classification** | Probabilistic risk prediction vs. binary rule |
| **Severity Scoring** | Prioritization of alerts (0-100 scale) |
| **Early Warning** | 4-week forward prediction |
| **SHAP Explainability** | Quantified feature importance for root cause |

### 2.3 Integration Points

| System | Integration Method | Data Direction |
|--------|-------------------|----------------|
| SAP BDC | Delta Sharing | Read (source data) |
| SAP Databricks | Native | Processing |
| SAP BDC | Delta Sharing | Write (predictions) |
| SAP BTP Flask | REST API / Delta table read | Consume predictions |
| SAP Analytics Cloud | Data Product subscription | Dashboards |

---

## 3. Source Data Specification

### 3.1 Data Products from SAP BDC

All source tables are accessed via Delta Sharing from SAP BDC catalog: `sap_bdc.inventory_risk`

```python
# Databricks notebook - Data Access
stock_status = spark.read.table("sap_bdc.inventory_risk.stock_status_v2")
review_dc = spark.read.table("sap_bdc.inventory_risk.review_dc")
review_plant = spark.read.table("sap_bdc.inventory_risk.review_plant")
review_vendors = spark.read.table("sap_bdc.inventory_risk.review_vendors")
lag_1_review_dc = spark.read.table("sap_bdc.inventory_risk.lag_1_review_dc")
lag_1_review_plant = spark.read.table("sap_bdc.inventory_risk.lag_1_review_plant")
location_source = spark.read.table("sap_bdc.inventory_risk.location_source")
production_source = spark.read.table("sap_bdc.inventory_risk.production_source_header")
```

### 3.2 Primary Source: `stock_status_v2`

This is the main table for ML training and scoring.

| Column | Type | Description | ML Usage |
|--------|------|-------------|----------|
| `product_id` | STRING | Product identifier (e.g., FG-100-001) | Primary key |
| `location_id` | STRING | Location identifier (e.g., DC1000) | Primary key |
| `week_num` | INT | ISO week number (1-52) | Primary key |
| `year` | INT | Calendar year | Primary key |
| `projected_stock` | DOUBLE | Expected stock after receipts/issues | Feature input |
| `safety_stock` | DOUBLE | Buffer stock level | Feature input |
| `stock_on_hand` | DOUBLE | Current available inventory | Feature input |
| `incoming_receipts` | DOUBLE | Confirmed inbound supply | Feature input |
| `total_demand` | DOUBLE | Total demand for the week | Feature input |
| `outgoing_supply` | DOUBLE | Outbound shipments | Feature input |
| `supply_orders` | DOUBLE | Open purchase/transport orders | Feature input |
| `location_type` | STRING | PL/DC/RDC/VEN | Feature input |
| `lag_1_dependent_demand` | DOUBLE | Previous week's demand | Feature input |
| `lag_1_supply_orders` | DOUBLE | Previous week's supply orders | Feature input |
| `lag_1_incoming_receipts` | DOUBLE | Previous week's receipts | Feature input |
| `transportation_lead_time` | DOUBLE | Weeks for RDC→DC transport | Feature input |
| `production_lead_time` | DOUBLE | Weeks for production | Feature input |
| `offset_stock` | DOUBLE | projected_stock - safety_stock | Derived |
| `stock_condition` | STRING | excess/deficit/in-stock | **Target (current)** |
| `stock_status_warning` | STRING | normal/understock_instance_N/overstock_instance_N | **Target (label)** |

### 3.3 Supporting Tables

#### `review_dc` / `review_plant` (Pivoted Key Figures)

| Key Figure | Description |
|------------|-------------|
| Dependent Demand | Downstream demand |
| Stock on Hand | Current inventory |
| Total Open PO+STO | Open orders |
| Incoming Transport Receipts | Inbound supply |
| Outgoing Supply | Outbound shipments |
| Safety Stock (SOP) | Buffer stock |
| Inventory Target | Target stock level |
| Projected Stock - Calculated | Expected stock |

#### `location_source` (Transportation Rules)

| Column | ML Usage |
|--------|----------|
| `transportation_lead_time` | Lead time feature |
| `minimum_transportation_lot_size` | Supply constraint feature |
| `incremental_transportation_lot_size` | Supply flexibility feature |

### 3.4 Data Volume Estimates

| Table | Rows (est.) | Update Frequency |
|-------|-------------|------------------|
| stock_status_v2 | ~500K | Weekly |
| review_dc | ~200K | Weekly |
| review_plant | ~100K | Weekly |
| ml_features | ~500K | Daily |
| ml_predictions | ~500K | Daily |

---

## 4. Phase 2: Feature Engineering

### 4.1 Feature Catalog

#### 4.1.1 Stock Features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `stock_to_safety_ratio` | `projected_stock / safety_stock` | Core risk indicator; <1 = understock risk, >1.5 = overstock risk |
| `stock_coverage_weeks` | `projected_stock / demand_rolling_mean_4w` | How many weeks of demand can current stock cover |
| `offset_stock_pct` | `offset_stock / safety_stock` | Normalized deviation from safety stock |
| `stock_velocity` | `(stock_t - stock_t-1) / stock_t-1` | Week-over-week stock change rate |

#### 4.1.2 Demand Features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `demand_rolling_mean_4w` | `AVG(total_demand) OVER 4 weeks` | Baseline demand level |
| `demand_rolling_std_4w` | `STDDEV(total_demand) OVER 4 weeks` | Demand variability |
| `demand_rolling_cv_4w` | `demand_rolling_std_4w / demand_rolling_mean_4w` | Coefficient of variation; high CV = unpredictable demand |
| `demand_wow_change` | `(demand_t - demand_t-1) / demand_t-1` | Week-over-week demand change |
| `demand_trend_4w` | `LINEAR_SLOPE(total_demand) OVER 4 weeks` | Demand trajectory (+ve = increasing) |
| `demand_spike_flag` | `1 if demand > mean + 2*std else 0` | Binary flag for unusual demand |

#### 4.1.3 Supply Features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `supply_demand_ratio` | `incoming_receipts / total_demand` | Supply adequacy; <1 = potential shortage |
| `supply_reliability_idx` | `actual_receipts / lag_1_supply_orders` | Supplier performance |
| `supply_rolling_mean_4w` | `AVG(incoming_receipts) OVER 4 weeks` | Baseline supply level |
| `supply_volatility` | `STDDEV(incoming_receipts) / AVG(incoming_receipts)` | Supply consistency |
| `order_fulfillment_gap` | `supply_orders - incoming_receipts` | Unfulfilled orders |

#### 4.1.4 Lead Time Features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `total_lead_time` | `transportation_lead_time + production_lead_time` | Total replenishment time |
| `lead_time_coverage` | `projected_stock / (demand_rolling_mean_4w * total_lead_time)` | Can stock last through lead time? |
| `lead_time_demand_exposure` | `demand_rolling_mean_4w * total_lead_time` | Demand during lead time window |

#### 4.1.5 Pattern Features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `consecutive_deficit_weeks` | Streak count where `stock_condition = 'deficit'` | Persistence of understock |
| `consecutive_excess_weeks` | Streak count where `stock_condition = 'excess'` | Persistence of overstock |
| `weeks_since_last_normal` | Count since last `stock_condition = 'in-stock'` | Duration of abnormal state |
| `risk_trend_direction` | `+1` if worsening, `-1` if improving, `0` if stable | Risk trajectory |

#### 4.1.6 Location Features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `location_type_encoded` | One-hot: `is_dc`, `is_rdc`, `is_plant` | Location-specific behavior |
| `location_avg_demand` | `AVG(demand) for location over 12 weeks` | Location demand profile |
| `location_risk_history` | Count of past warning instances at location | Historical risk pattern |

### 4.2 Feature Engineering Pipeline

```python
# File: notebooks/feature_engineering.py

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, DoubleType

def calculate_consecutive_weeks(df, condition_col, condition_value, output_col):
    """Calculate streak of consecutive weeks meeting a condition."""
    w = Window.partitionBy("product_id", "location_id").orderBy("year", "week_num")

    return df \
        .withColumn("condition_met", F.when(F.col(condition_col) == condition_value, 1).otherwise(0)) \
        .withColumn("group_id", F.sum(
            F.when(F.col("condition_met") == 0, 1).otherwise(0)
        ).over(w)) \
        .withColumn(output_col, F.when(
            F.col("condition_met") == 1,
            F.row_number().over(Window.partitionBy("product_id", "location_id", "group_id").orderBy("year", "week_num"))
        ).otherwise(0)) \
        .drop("condition_met", "group_id")


def generate_features(stock_status_df, location_source_df):
    """
    Generate ML features from stock_status_v2 and supporting tables.

    Args:
        stock_status_df: spark DataFrame from stock_status_v2
        location_source_df: spark DataFrame from location_source

    Returns:
        spark DataFrame with all engineered features
    """

    # Window specifications
    w = Window.partitionBy("product_id", "location_id").orderBy("year", "week_num")
    w_4w = w.rowsBetween(-3, 0)  # Current week + 3 prior weeks
    w_12w = w.rowsBetween(-11, 0)  # 12-week lookback

    # Join lead time data
    df = stock_status_df.join(
        location_source_df.select("location_id", "product_id", "transportation_lead_time"),
        on=["location_id", "product_id"],
        how="left"
    )

    # ==================== STOCK FEATURES ====================
    df = df \
        .withColumn("stock_to_safety_ratio",
            F.when(F.col("safety_stock") > 0,
                   F.col("projected_stock") / F.col("safety_stock"))
            .otherwise(F.lit(None))) \
        .withColumn("offset_stock_pct",
            F.when(F.col("safety_stock") > 0,
                   F.col("offset_stock") / F.col("safety_stock"))
            .otherwise(F.lit(None))) \
        .withColumn("stock_velocity",
            (F.col("projected_stock") - F.lag("projected_stock", 1).over(w)) /
            F.lag("projected_stock", 1).over(w))

    # ==================== DEMAND FEATURES ====================
    df = df \
        .withColumn("demand_rolling_mean_4w", F.avg("total_demand").over(w_4w)) \
        .withColumn("demand_rolling_std_4w", F.stddev("total_demand").over(w_4w)) \
        .withColumn("demand_rolling_cv_4w",
            F.when(F.col("demand_rolling_mean_4w") > 0,
                   F.col("demand_rolling_std_4w") / F.col("demand_rolling_mean_4w"))
            .otherwise(F.lit(0))) \
        .withColumn("demand_wow_change",
            F.when(F.lag("total_demand", 1).over(w) > 0,
                   (F.col("total_demand") - F.lag("total_demand", 1).over(w)) /
                   F.lag("total_demand", 1).over(w))
            .otherwise(F.lit(0))) \
        .withColumn("demand_spike_flag",
            F.when(F.col("total_demand") >
                   F.col("demand_rolling_mean_4w") + 2 * F.col("demand_rolling_std_4w"), 1)
            .otherwise(0))

    # Stock coverage
    df = df.withColumn("stock_coverage_weeks",
        F.when(F.col("demand_rolling_mean_4w") > 0,
               F.col("projected_stock") / F.col("demand_rolling_mean_4w"))
        .otherwise(F.lit(None)))

    # ==================== SUPPLY FEATURES ====================
    df = df \
        .withColumn("supply_demand_ratio",
            F.when(F.col("total_demand") > 0,
                   F.col("incoming_receipts") / F.col("total_demand"))
            .otherwise(F.lit(None))) \
        .withColumn("supply_reliability_idx",
            F.when(F.col("lag_1_supply_orders") > 0,
                   F.col("incoming_receipts") / F.col("lag_1_supply_orders"))
            .otherwise(F.lit(1.0))) \
        .withColumn("supply_rolling_mean_4w", F.avg("incoming_receipts").over(w_4w)) \
        .withColumn("order_fulfillment_gap",
            F.col("supply_orders") - F.col("incoming_receipts"))

    # ==================== LEAD TIME FEATURES ====================
    df = df \
        .withColumn("total_lead_time",
            F.coalesce(F.col("transportation_lead_time"), F.lit(0)) +
            F.coalesce(F.col("production_lead_time"), F.lit(0))) \
        .withColumn("lead_time_coverage",
            F.when((F.col("demand_rolling_mean_4w") > 0) & (F.col("total_lead_time") > 0),
                   F.col("projected_stock") / (F.col("demand_rolling_mean_4w") * F.col("total_lead_time")))
            .otherwise(F.lit(None))) \
        .withColumn("lead_time_demand_exposure",
            F.col("demand_rolling_mean_4w") * F.col("total_lead_time"))

    # ==================== PATTERN FEATURES ====================
    df = calculate_consecutive_weeks(df, "stock_condition", "deficit", "consecutive_deficit_weeks")
    df = calculate_consecutive_weeks(df, "stock_condition", "excess", "consecutive_excess_weeks")

    # ==================== LOCATION FEATURES ====================
    df = df \
        .withColumn("is_dc", F.when(F.col("location_type") == "DC", 1).otherwise(0)) \
        .withColumn("is_rdc", F.when(F.col("location_type") == "RDC", 1).otherwise(0)) \
        .withColumn("is_plant", F.when(F.col("location_type").isin(["PL", "P"]), 1).otherwise(0))

    # Location historical demand
    location_demand = df.groupBy("location_id").agg(
        F.avg("total_demand").alias("location_avg_demand")
    )
    df = df.join(location_demand, on="location_id", how="left")

    # ==================== TARGET VARIABLE ====================
    df = df.withColumn("risk_label",
        F.when(F.col("stock_status_warning").contains("understock"), 1)
        .when(F.col("stock_status_warning").contains("overstock"), 2)
        .otherwise(0))  # 0 = normal, 1 = understock, 2 = overstock

    return df


# Feature columns for ML models
FEATURE_COLS = [
    # Stock
    "stock_to_safety_ratio", "stock_coverage_weeks", "offset_stock_pct", "stock_velocity",
    # Demand
    "demand_rolling_mean_4w", "demand_rolling_cv_4w", "demand_wow_change", "demand_spike_flag",
    # Supply
    "supply_demand_ratio", "supply_reliability_idx", "supply_rolling_mean_4w", "order_fulfillment_gap",
    # Lead Time
    "total_lead_time", "lead_time_coverage", "lead_time_demand_exposure",
    # Pattern
    "consecutive_deficit_weeks", "consecutive_excess_weeks",
    # Location
    "is_dc", "is_rdc", "is_plant", "location_avg_demand"
]

PRIMARY_KEYS = ["product_id", "location_id", "year", "week_num"]
```

### 4.3 Feature Store Registration

```python
# File: notebooks/feature_store_setup.py

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

fe = FeatureEngineeringClient()

# Create feature table
fe.create_table(
    name="sap_bdc.inventory_risk.ml_features",
    primary_keys=["product_id", "location_id", "year", "week_num"],
    timestamp_keys=["feature_timestamp"],
    description="Inventory risk ML features - daily refresh",
    tags={"team": "ml-engineering", "domain": "supply-chain"}
)

# Write features
fe.write_table(
    name="sap_bdc.inventory_risk.ml_features",
    df=features_df,
    mode="merge"  # Upsert based on primary keys
)
```

### 4.4 Feature Validation Rules

| Feature | Valid Range | Null Handling |
|---------|-------------|---------------|
| `stock_to_safety_ratio` | [0, 10] | Fill with 1.0 |
| `demand_rolling_cv_4w` | [0, 5] | Fill with 0.0 |
| `supply_demand_ratio` | [0, 10] | Fill with 1.0 |
| `consecutive_deficit_weeks` | [0, 52] | Fill with 0 |
| `total_lead_time` | [0, 26] | Fill with median |

```python
def validate_features(df):
    """Apply validation rules and handle nulls/outliers."""
    return df \
        .withColumn("stock_to_safety_ratio",
            F.when(F.col("stock_to_safety_ratio").isNull(), 1.0)
            .when(F.col("stock_to_safety_ratio") > 10, 10.0)
            .when(F.col("stock_to_safety_ratio") < 0, 0.0)
            .otherwise(F.col("stock_to_safety_ratio"))) \
        .withColumn("demand_rolling_cv_4w",
            F.coalesce(F.col("demand_rolling_cv_4w"), F.lit(0.0))) \
        .fillna(0, subset=["consecutive_deficit_weeks", "consecutive_excess_weeks"])
```

---

## 5. Phase 3: ML Models

### 5.1 Model 1: Risk Classification (XGBoost)

#### 5.1.1 Objective
Classify each product-location-week combination into: `normal` (0), `understock` (1), `overstock` (2)

#### 5.1.2 Training Configuration

```python
# File: notebooks/train_risk_classifier.py

import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, f1_score
import shap

# MLflow experiment setup
mlflow.set_experiment("/inventory-risk/risk-classifier")

CLASSIFIER_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.05,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "early_stopping_rounds": 50,
    "random_state": 42,
    "n_jobs": -1
}

# Class weights for imbalanced data (normal >> understock/overstock)
CLASS_WEIGHTS = {0: 1.0, 1: 2.5, 2: 2.0}

def train_risk_classifier(features_df):
    """Train XGBoost risk classifier with MLflow tracking."""

    # Prepare data
    X = features_df[FEATURE_COLS].values
    y = features_df["risk_label"].values

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Apply class weights
    sample_weights = [CLASS_WEIGHTS[label] for label in y_train]

    with mlflow.start_run(run_name="xgboost_risk_classifier"):
        # Log parameters
        mlflow.log_params(CLASSIFIER_PARAMS)
        mlflow.log_param("class_weights", CLASS_WEIGHTS)
        mlflow.log_param("features", FEATURE_COLS)

        # Train
        model = XGBClassifier(**CLASSIFIER_PARAMS)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test)],
            verbose=100
        )

        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Metrics
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_understock = f1_score(y_test, y_pred, labels=[1], average='micro')
        f1_overstock = f1_score(y_test, y_pred, labels=[2], average='micro')

        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("f1_understock", f1_understock)
        mlflow.log_metric("f1_overstock", f1_overstock)

        # Log classification report
        report = classification_report(y_test, y_pred,
                                       target_names=["normal", "understock", "overstock"])
        mlflow.log_text(report, "classification_report.txt")

        # SHAP explainability
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:1000])
        mlflow.log_artifact(shap.summary_plot(shap_values, X_test[:1000],
                                               feature_names=FEATURE_COLS, show=False))

        # Log model
        mlflow.xgboost.log_model(
            model,
            "risk_classifier",
            registered_model_name="inventory_risk_classifier"
        )

        return model, explainer
```

#### 5.1.3 Model Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| F1 (macro) | ≥ 0.70 | Balanced performance across classes |
| F1 (understock) | ≥ 0.75 | Higher priority - stockout risk |
| F1 (overstock) | ≥ 0.65 | Important but less critical |
| Precision (understock) | ≥ 0.80 | Minimize false alarms |
| Recall (understock) | ≥ 0.70 | Don't miss real risks |

### 5.2 Model 2: Early Warning (LSTM)

#### 5.2.1 Objective
Predict risk occurrence probability for next 4 weeks given 12 weeks of history.

#### 5.2.2 Architecture

```python
# File: notebooks/train_early_warning.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import mlflow.tensorflow

SEQUENCE_LENGTH = 12  # 12 weeks history
FORECAST_HORIZON = 4  # Predict 4 weeks ahead

LSTM_CONFIG = {
    "units_layer1": 64,
    "units_layer2": 32,
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    "batch_size": 256,
    "epochs": 100
}

def create_lstm_model(n_features):
    """Create LSTM model for early warning prediction."""

    model = Sequential([
        # First LSTM layer
        LSTM(
            units=LSTM_CONFIG["units_layer1"],
            input_shape=(SEQUENCE_LENGTH, n_features),
            return_sequences=True,
            name="lstm_1"
        ),
        BatchNormalization(),
        Dropout(LSTM_CONFIG["dropout_rate"]),

        # Second LSTM layer
        LSTM(
            units=LSTM_CONFIG["units_layer2"],
            return_sequences=False,
            name="lstm_2"
        ),
        BatchNormalization(),
        Dropout(LSTM_CONFIG["dropout_rate"]),

        # Output layer - 4 weeks prediction
        Dense(FORECAST_HORIZON, activation='sigmoid', name="output")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LSTM_CONFIG["learning_rate"]),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model


def prepare_sequences(df, feature_cols):
    """Convert DataFrame to sequences for LSTM."""
    sequences = []
    targets = []

    for (product_id, location_id), group in df.groupby(["product_id", "location_id"]):
        group = group.sort_values(["year", "week_num"])
        features = group[feature_cols].values
        labels = group["risk_label"].apply(lambda x: 1 if x > 0 else 0).values

        for i in range(SEQUENCE_LENGTH, len(features) - FORECAST_HORIZON):
            seq = features[i - SEQUENCE_LENGTH:i]
            target = labels[i:i + FORECAST_HORIZON]
            sequences.append(seq)
            targets.append(target)

    return np.array(sequences), np.array(targets)


def train_early_warning_model(features_df):
    """Train LSTM early warning model."""

    mlflow.set_experiment("/inventory-risk/early-warning-lstm")

    X, y = prepare_sequences(features_df, FEATURE_COLS)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="lstm_early_warning"):
        mlflow.log_params(LSTM_CONFIG)

        model = create_lstm_model(len(FEATURE_COLS))

        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint("best_lstm.h5", save_best_only=True)
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=LSTM_CONFIG["epochs"],
            batch_size=LSTM_CONFIG["batch_size"],
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = np.mean((y_pred > 0.5) == y_test)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.tensorflow.log_model(model, "early_warning_lstm",
                                     registered_model_name="inventory_early_warning")

        return model
```

#### 5.2.3 Model Performance Targets

| Metric | Target |
|--------|--------|
| Accuracy (any risk in 4 weeks) | ≥ 70% |
| AUC | ≥ 0.75 |
| Lead time accuracy (which week) | ≥ 60% |

### 5.3 Model 3: Severity Scoring (LightGBM)

#### 5.3.1 Objective
Predict a continuous severity score (0-100) for prioritization.

#### 5.3.2 Severity Score Definition

| Score Range | Severity | Description |
|-------------|----------|-------------|
| 0-30 | Low | Minor deviation, self-correcting |
| 30-60 | Medium | Attention needed, plan action |
| 60-80 | High | Immediate action required |
| 80-100 | Critical | Emergency intervention |

#### 5.3.3 Training Configuration

```python
# File: notebooks/train_severity_scorer.py

import lightgbm as lgb
import mlflow.lightgbm

SEVERITY_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1
}

def calculate_severity_target(row):
    """
    Calculate target severity score based on multiple factors.

    Factors:
    - Consecutive weeks of risk (0-40 points)
    - Magnitude of offset (0-30 points)
    - Demand volatility (0-15 points)
    - Supply reliability (0-15 points)
    """
    score = 0

    # Consecutive weeks (max 40 points)
    weeks = max(row["consecutive_deficit_weeks"], row["consecutive_excess_weeks"])
    score += min(weeks * 10, 40)

    # Offset magnitude (max 30 points)
    if row["stock_to_safety_ratio"] < 0.5:  # Severe understock
        score += 30
    elif row["stock_to_safety_ratio"] < 0.8:  # Moderate understock
        score += 20
    elif row["stock_to_safety_ratio"] > 2.0:  # Severe overstock
        score += 25
    elif row["stock_to_safety_ratio"] > 1.5:  # Moderate overstock
        score += 15

    # Demand volatility (max 15 points)
    if row["demand_rolling_cv_4w"] > 0.5:
        score += 15
    elif row["demand_rolling_cv_4w"] > 0.3:
        score += 10

    # Supply reliability (max 15 points)
    if row["supply_reliability_idx"] < 0.7:
        score += 15
    elif row["supply_reliability_idx"] < 0.9:
        score += 8

    return min(score, 100)


def train_severity_scorer(features_df):
    """Train LightGBM severity scoring model."""

    mlflow.set_experiment("/inventory-risk/severity-scorer")

    # Calculate target
    features_df["severity_target"] = features_df.apply(calculate_severity_target, axis=1)

    X = features_df[FEATURE_COLS].values
    y = features_df["severity_target"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="lightgbm_severity"):
        mlflow.log_params(SEVERITY_PARAMS)

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        model = lgb.train(
            SEVERITY_PARAMS,
            train_data,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(50)]
        )

        # Evaluate
        y_pred = model.predict(X_test)
        mae = np.mean(np.abs(y_pred - y_test))

        mlflow.log_metric("mae", mae)
        mlflow.lightgbm.log_model(model, "severity_scorer",
                                   registered_model_name="inventory_severity_scorer")

        return model
```

### 5.4 SHAP Explainability Module

```python
# File: notebooks/shap_explainability.py

import shap
import json

def get_top_factors(model, features, feature_names, top_n=3):
    """
    Get top N contributing factors for a prediction using SHAP.

    Returns:
        List of tuples: [(feature_name, shap_value, direction), ...]
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # For multi-class, get SHAP values for predicted class
    if isinstance(shap_values, list):
        pred_class = model.predict(features)[0]
        shap_vals = shap_values[pred_class][0]
    else:
        shap_vals = shap_values[0]

    # Get absolute values and sort
    importance = [(feature_names[i], shap_vals[i], "+" if shap_vals[i] > 0 else "-")
                  for i in range(len(shap_vals))]
    importance.sort(key=lambda x: abs(x[1]), reverse=True)

    return importance[:top_n]


def format_factor_explanation(factor_name, shap_value, direction):
    """Convert SHAP factor to human-readable explanation."""

    FACTOR_EXPLANATIONS = {
        "stock_to_safety_ratio": {
            "+": "Stock level above safety threshold",
            "-": "Stock level below safety threshold"
        },
        "demand_rolling_cv_4w": {
            "+": "High demand variability",
            "-": "Stable demand pattern"
        },
        "supply_demand_ratio": {
            "+": "Supply exceeds demand",
            "-": "Supply shortage relative to demand"
        },
        "consecutive_deficit_weeks": {
            "+": "Prolonged understock condition",
            "-": "Improving stock condition"
        },
        "consecutive_excess_weeks": {
            "+": "Prolonged overstock condition",
            "-": "Stock normalizing"
        },
        "supply_reliability_idx": {
            "+": "Reliable supplier delivery",
            "-": "Unreliable supplier delivery"
        },
        "lead_time_coverage": {
            "+": "Sufficient stock for lead time",
            "-": "Insufficient stock for lead time"
        },
        "demand_wow_change": {
            "+": "Increasing demand trend",
            "-": "Decreasing demand trend"
        }
    }

    explanation = FACTOR_EXPLANATIONS.get(factor_name, {}).get(direction, factor_name)
    return f"{explanation} (impact: {abs(shap_value):.2f})"
```

---

## 6. Phase 4: Write-Back to SAP BDC

### 6.1 Output Table Schema

```sql
-- File: sql/create_ml_predictions.sql

CREATE TABLE IF NOT EXISTS sap_bdc.inventory_risk.ml_predictions (
    -- Primary Keys
    product_id STRING NOT NULL COMMENT 'Product identifier (e.g., FG-100-001)',
    location_id STRING NOT NULL COMMENT 'Location identifier (e.g., DC1000)',
    week_num INT NOT NULL COMMENT 'ISO week number (1-52)',
    year INT NOT NULL COMMENT 'Calendar year',

    -- ML Classification Output
    risk_class STRING COMMENT 'Predicted risk class: normal, understock, overstock',
    risk_probability DOUBLE COMMENT 'Probability of predicted class (0.0 - 1.0)',

    -- Severity Score
    severity_score DOUBLE COMMENT 'Risk severity score (0-100)',
    severity_level STRING COMMENT 'Severity level: Low, Medium, High, Critical',

    -- Early Warning
    early_warning_weeks INT COMMENT 'Predicted weeks until risk occurrence (0-4)',
    early_warning_probability DOUBLE COMMENT 'Probability of risk in next 4 weeks',

    -- SHAP Explainability
    top_factor_1 STRING COMMENT 'Primary contributing factor with explanation',
    top_factor_1_impact DOUBLE COMMENT 'SHAP impact value for factor 1',
    top_factor_2 STRING COMMENT 'Secondary contributing factor',
    top_factor_2_impact DOUBLE COMMENT 'SHAP impact value for factor 2',
    top_factor_3 STRING COMMENT 'Tertiary contributing factor',
    top_factor_3_impact DOUBLE COMMENT 'SHAP impact value for factor 3',

    -- Feature Snapshot (for debugging/audit)
    stock_to_safety_ratio DOUBLE,
    demand_rolling_cv_4w DOUBLE,
    supply_demand_ratio DOUBLE,
    consecutive_deficit_weeks INT,
    consecutive_excess_weeks INT,

    -- Metadata
    model_version STRING COMMENT 'Version of ML models used',
    prediction_timestamp TIMESTAMP COMMENT 'When prediction was generated',
    feature_timestamp TIMESTAMP COMMENT 'Timestamp of features used',

    -- Partitioning
    prediction_date DATE COMMENT 'Date of prediction for partitioning'
)
USING DELTA
PARTITIONED BY (prediction_date)
COMMENT 'ML predictions for inventory risk - refreshed daily'
TBLPROPERTIES (
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact' = 'true'
);
```

### 6.2 Batch Scoring Pipeline

```python
# File: notebooks/batch_scoring.py

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, DoubleType, IntegerType, ArrayType, StructType, StructField
import mlflow
from datetime import datetime

def load_models():
    """Load production models from MLflow registry."""

    risk_classifier = mlflow.xgboost.load_model("models:/inventory_risk_classifier/Production")
    severity_scorer = mlflow.lightgbm.load_model("models:/inventory_severity_scorer/Production")
    early_warning = mlflow.tensorflow.load_model("models:/inventory_early_warning/Production")

    return risk_classifier, severity_scorer, early_warning


def create_scoring_udfs(risk_classifier, severity_scorer, shap_explainer):
    """Create Spark UDFs for model scoring."""

    # Risk classification UDF
    @F.udf(returnType=StringType())
    def classify_risk(features):
        pred = risk_classifier.predict([features])[0]
        return ["normal", "understock", "overstock"][pred]

    @F.udf(returnType=DoubleType())
    def risk_probability(features):
        proba = risk_classifier.predict_proba([features])[0]
        return float(max(proba))

    # Severity scoring UDF
    @F.udf(returnType=DoubleType())
    def score_severity(features):
        return float(severity_scorer.predict([features])[0])

    @F.udf(returnType=StringType())
    def severity_level(score):
        if score < 30:
            return "Low"
        elif score < 60:
            return "Medium"
        elif score < 80:
            return "High"
        else:
            return "Critical"

    # Top factors UDF
    @F.udf(returnType=ArrayType(StructType([
        StructField("factor", StringType()),
        StructField("impact", DoubleType())
    ])))
    def get_top_factors(features):
        shap_values = shap_explainer.shap_values([features])
        # Return top 3 factors
        importance = sorted(enumerate(shap_values[0]), key=lambda x: abs(x[1]), reverse=True)[:3]
        return [(FEATURE_COLS[i], float(v)) for i, v in importance]

    return classify_risk, risk_probability, score_severity, severity_level, get_top_factors


def run_batch_scoring():
    """Execute daily batch scoring pipeline."""

    # Load models
    risk_classifier, severity_scorer, early_warning = load_models()
    shap_explainer = shap.TreeExplainer(risk_classifier)

    # Create UDFs
    classify_risk, risk_probability, score_severity, severity_level, get_top_factors = \
        create_scoring_udfs(risk_classifier, severity_scorer, shap_explainer)

    # Load features
    features_df = spark.read.table("sap_bdc.inventory_risk.ml_features")

    # Get latest week's features
    latest_week = features_df.agg(
        F.max(F.struct("year", "week_num")).alias("max_week")
    ).collect()[0]["max_week"]

    current_features = features_df.filter(
        (F.col("year") == latest_week["year"]) &
        (F.col("week_num") == latest_week["week_num"])
    )

    # Create feature vector column
    feature_vector = F.array([F.col(c) for c in FEATURE_COLS])

    # Score
    predictions = current_features \
        .withColumn("feature_vector", feature_vector) \
        .withColumn("risk_class", classify_risk(F.col("feature_vector"))) \
        .withColumn("risk_probability", risk_probability(F.col("feature_vector"))) \
        .withColumn("severity_score", score_severity(F.col("feature_vector"))) \
        .withColumn("severity_level", severity_level(F.col("severity_score"))) \
        .withColumn("top_factors", get_top_factors(F.col("feature_vector"))) \
        .withColumn("top_factor_1", F.col("top_factors")[0]["factor"]) \
        .withColumn("top_factor_1_impact", F.col("top_factors")[0]["impact"]) \
        .withColumn("top_factor_2", F.col("top_factors")[1]["factor"]) \
        .withColumn("top_factor_2_impact", F.col("top_factors")[1]["impact"]) \
        .withColumn("top_factor_3", F.col("top_factors")[2]["factor"]) \
        .withColumn("top_factor_3_impact", F.col("top_factors")[2]["impact"]) \
        .withColumn("model_version", F.lit("v1.0.0")) \
        .withColumn("prediction_timestamp", F.current_timestamp()) \
        .withColumn("feature_timestamp", F.col("feature_timestamp")) \
        .withColumn("prediction_date", F.current_date())

    # Select final columns
    output_cols = [
        "product_id", "location_id", "week_num", "year",
        "risk_class", "risk_probability",
        "severity_score", "severity_level",
        "early_warning_weeks", "early_warning_probability",
        "top_factor_1", "top_factor_1_impact",
        "top_factor_2", "top_factor_2_impact",
        "top_factor_3", "top_factor_3_impact",
        "stock_to_safety_ratio", "demand_rolling_cv_4w", "supply_demand_ratio",
        "consecutive_deficit_weeks", "consecutive_excess_weeks",
        "model_version", "prediction_timestamp", "feature_timestamp", "prediction_date"
    ]

    # Write to Delta table
    predictions.select(output_cols) \
        .write \
        .format("delta") \
        .mode("overwrite") \
        .option("replaceWhere", f"prediction_date = '{datetime.now().date()}'") \
        .saveAsTable("sap_bdc.inventory_risk.ml_predictions")

    return predictions.count()
```

### 6.3 Delta Sharing Configuration

```python
# File: notebooks/delta_sharing_setup.py

# The ml_predictions table is automatically available via Delta Sharing
# Configuration in SAP BDC:

delta_sharing_config = {
    "share_name": "inventory_risk_ml",
    "schema": "sap_bdc.inventory_risk",
    "tables": [
        {
            "name": "ml_predictions",
            "shared_as": "ml_predictions",
            "partitions": ["prediction_date"],
            "cdf_enabled": True  # Change Data Feed for incremental sync
        },
        {
            "name": "ml_features",
            "shared_as": "ml_features",
            "partitions": ["year", "week_num"]
        }
    ],
    "recipients": [
        "sap_analytics_cloud",
        "sap_btp_flask_app",
        "sap_s4hana"
    ]
}
```

---

## 7. Phase 5: Model Serving & Integration

### 7.1 Databricks Model Serving Endpoint

```python
# File: notebooks/model_serving_setup.py

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()

# Create serving endpoint
endpoint_config = EndpointCoreConfigInput(
    name="inventory-risk-endpoint",
    served_entities=[
        ServedEntityInput(
            name="risk-classifier",
            entity_name="inventory_risk_classifier",
            entity_version="1",  # or "Production" stage
            workload_size="Small",
            scale_to_zero_enabled=True
        ),
        ServedEntityInput(
            name="severity-scorer",
            entity_name="inventory_severity_scorer",
            entity_version="1",
            workload_size="Small",
            scale_to_zero_enabled=True
        )
    ]
)

w.serving_endpoints.create(
    name="inventory-risk-endpoint",
    config=endpoint_config
)
```

### 7.2 API Contract

#### Request

```yaml
POST https://<databricks-workspace>/serving-endpoints/inventory-risk-endpoint/invocations

Headers:
  Authorization: Bearer <DATABRICKS_TOKEN>
  Content-Type: application/json

Request Body:
{
  "dataframe_records": [
    {
      "product_id": "FG-100-001",
      "location_id": "DC1000",
      "stock_to_safety_ratio": 0.75,
      "demand_rolling_cv_4w": 0.32,
      "demand_wow_change": 0.15,
      "supply_demand_ratio": 0.85,
      "supply_reliability_idx": 0.92,
      "total_lead_time": 3,
      "consecutive_deficit_weeks": 2,
      "consecutive_excess_weeks": 0,
      "is_dc": 1,
      "is_rdc": 0,
      "is_plant": 0,
      "location_avg_demand": 5000
    }
  ]
}
```

#### Response

```json
{
  "predictions": [
    {
      "risk_class": "understock",
      "risk_probability": 0.78,
      "severity_score": 62.5,
      "severity_level": "High",
      "early_warning_weeks": 2,
      "top_factors": [
        {"factor": "stock_to_safety_ratio", "impact": -0.45, "explanation": "Stock level below safety threshold"},
        {"factor": "supply_demand_ratio", "impact": -0.28, "explanation": "Supply shortage relative to demand"},
        {"factor": "consecutive_deficit_weeks", "impact": 0.22, "explanation": "Prolonged understock condition"}
      ]
    }
  ]
}
```

### 7.3 SAP BTP Flask App Integration

```python
# File: Production-files/InventoryRiskMasterAgentAPI_production_v1/ml_integration.py

import os
import requests
from functools import lru_cache

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
DATABRICKS_ENDPOINT = f"https://{DATABRICKS_HOST}/serving-endpoints/inventory-risk-endpoint/invocations"

# Alternative: Read from Delta table via SAP BDC
HANA_ML_PREDICTIONS_TABLE = "CURRENT_INVT.ML_PREDICTIONS"


class MLPredictionClient:
    """Client for fetching ML predictions."""

    def __init__(self, mode="table"):
        """
        Args:
            mode: "api" for real-time serving, "table" for batch predictions
        """
        self.mode = mode

    def get_prediction_from_api(self, product_id, location_id, features):
        """Real-time prediction via Databricks Model Serving."""

        payload = {
            "dataframe_records": [{
                "product_id": product_id,
                "location_id": location_id,
                **features
            }]
        }

        response = requests.post(
            DATABRICKS_ENDPOINT,
            headers={
                "Authorization": f"Bearer {DATABRICKS_TOKEN}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=5
        )

        if response.status_code == 200:
            return response.json()["predictions"][0]
        else:
            raise Exception(f"ML API error: {response.status_code} - {response.text}")

    def get_prediction_from_table(self, engine, product_id, location_id):
        """Fetch pre-computed prediction from Delta table (via SAP BDC)."""

        query = f"""
        SELECT
            risk_class, risk_probability, severity_score, severity_level,
            early_warning_weeks, early_warning_probability,
            top_factor_1, top_factor_1_impact,
            top_factor_2, top_factor_2_impact,
            top_factor_3, top_factor_3_impact,
            prediction_timestamp
        FROM {HANA_ML_PREDICTIONS_TABLE}
        WHERE product_id = '{product_id}'
          AND location_id = '{location_id}'
          AND prediction_date = CURRENT_DATE
        """

        with engine.connect() as conn:
            result = conn.execute(text(query)).fetchone()

        if result:
            return {
                "risk_class": result[0],
                "risk_probability": result[1],
                "severity_score": result[2],
                "severity_level": result[3],
                "early_warning_weeks": result[4],
                "top_factors": [
                    {"factor": result[6], "impact": result[7]},
                    {"factor": result[8], "impact": result[9]},
                    {"factor": result[10], "impact": result[11]}
                ],
                "prediction_timestamp": result[12]
            }
        return None

    def get_prediction(self, engine, product_id, location_id, features=None):
        """Get prediction using configured mode."""

        if self.mode == "api" and features:
            return self.get_prediction_from_api(product_id, location_id, features)
        else:
            return self.get_prediction_from_table(engine, product_id, location_id)


# Initialize client
ml_client = MLPredictionClient(mode="table")  # Use batch predictions by default
```

### 7.4 Enhanced Reasoning Agent Response

```python
# File: Production-files/InventoryRiskMasterAgentAPI_production_v1/main.py (modifications)

from ml_integration import ml_client

def enhanced_reasoning_response(query, product_id, location_id, reasoning_result):
    """
    Enhance reasoning agent response with ML predictions.
    """

    # Get ML prediction
    ml_prediction = ml_client.get_prediction(engine, product_id, location_id)

    if ml_prediction:
        # Format ML insights
        ml_summary = f"""

**ML Risk Assessment:**
- **Risk Classification**: {ml_prediction['risk_class'].upper()} (Confidence: {ml_prediction['risk_probability']*100:.1f}%)
- **Severity**: {ml_prediction['severity_level']} ({ml_prediction['severity_score']:.0f}/100)
- **Early Warning**: {'Risk predicted in ' + str(ml_prediction['early_warning_weeks']) + ' weeks' if ml_prediction['early_warning_weeks'] > 0 else 'No immediate risk predicted'}

**Key Contributing Factors (ML-identified):**
1. {ml_prediction['top_factors'][0]['factor'].replace('_', ' ').title()}: Impact {ml_prediction['top_factors'][0]['impact']:.2f}
2. {ml_prediction['top_factors'][1]['factor'].replace('_', ' ').title()}: Impact {ml_prediction['top_factors'][1]['impact']:.2f}
3. {ml_prediction['top_factors'][2]['factor'].replace('_', ' ').title()}: Impact {ml_prediction['top_factors'][2]['impact']:.2f}
"""

        # Combine with reasoning result
        combined_response = reasoning_result + ml_summary

        return {
            "response": combined_response,
            "ml_prediction": ml_prediction,
            "reasoning": reasoning_result
        }

    return {
        "response": reasoning_result,
        "ml_prediction": None,
        "reasoning": reasoning_result
    }
```

---

## 8. MLOps & Orchestration

### 8.1 Databricks Workflow Definition

```yaml
# File: workflows/inventory_risk_ml_workflow.yaml

name: inventory_risk_ml_daily

schedule:
  quartz_cron_expression: "0 0 3 * * ?"  # 3 AM daily
  timezone_id: "UTC"

tasks:
  - task_key: feature_engineering
    description: "Generate ML features from SAP BDC data"
    notebook_task:
      notebook_path: /Repos/inventory-risk-ml/notebooks/feature_engineering
      base_parameters:
        run_date: "{{job.start_time.date}}"
    new_cluster:
      spark_version: "13.3.x-scala2.12"
      node_type_id: "Standard_DS3_v2"
      num_workers: 2
      spark_conf:
        spark.databricks.delta.preview.enabled: "true"
    timeout_seconds: 3600
    max_retries: 2

  - task_key: batch_scoring
    description: "Score all product-location combinations"
    depends_on:
      - task_key: feature_engineering
    notebook_task:
      notebook_path: /Repos/inventory-risk-ml/notebooks/batch_scoring
      base_parameters:
        run_date: "{{job.start_time.date}}"
    new_cluster:
      spark_version: "13.3.x-scala2.12"
      node_type_id: "Standard_DS4_v2"
      num_workers: 4
    timeout_seconds: 1800
    max_retries: 2

  - task_key: data_quality_check
    description: "Validate predictions quality"
    depends_on:
      - task_key: batch_scoring
    notebook_task:
      notebook_path: /Repos/inventory-risk-ml/notebooks/data_quality_check
    existing_cluster_id: "0123-456789-abcdef"
    timeout_seconds: 600

email_notifications:
  on_failure:
    - ml-team@company.com
  on_success:
    - ml-team@company.com
```

### 8.2 Weekly Retraining Workflow

```yaml
# File: workflows/inventory_risk_ml_retrain.yaml

name: inventory_risk_ml_weekly_retrain

schedule:
  quartz_cron_expression: "0 0 2 ? * SUN"  # 2 AM every Sunday
  timezone_id: "UTC"

tasks:
  - task_key: prepare_training_data
    notebook_task:
      notebook_path: /Repos/inventory-risk-ml/notebooks/prepare_training_data
    new_cluster:
      spark_version: "13.3.x-scala2.12"
      num_workers: 4
    timeout_seconds: 3600

  - task_key: train_risk_classifier
    depends_on:
      - task_key: prepare_training_data
    notebook_task:
      notebook_path: /Repos/inventory-risk-ml/notebooks/train_risk_classifier
    new_cluster:
      spark_version: "13.3.x-ml-scala2.12"
      num_workers: 2
      node_type_id: "Standard_DS4_v2"
    timeout_seconds: 7200

  - task_key: train_severity_scorer
    depends_on:
      - task_key: prepare_training_data
    notebook_task:
      notebook_path: /Repos/inventory-risk-ml/notebooks/train_severity_scorer
    new_cluster:
      spark_version: "13.3.x-ml-scala2.12"
      num_workers: 2
    timeout_seconds: 3600

  - task_key: train_early_warning
    depends_on:
      - task_key: prepare_training_data
    notebook_task:
      notebook_path: /Repos/inventory-risk-ml/notebooks/train_early_warning
    new_cluster:
      spark_version: "13.3.x-ml-scala2.12"
      num_workers: 2
      node_type_id: "Standard_NC6"  # GPU for LSTM
    timeout_seconds: 7200

  - task_key: model_validation
    depends_on:
      - task_key: train_risk_classifier
      - task_key: train_severity_scorer
      - task_key: train_early_warning
    notebook_task:
      notebook_path: /Repos/inventory-risk-ml/notebooks/model_validation
    timeout_seconds: 1800

  - task_key: promote_models
    depends_on:
      - task_key: model_validation
    condition_task:
      condition: "{{tasks.model_validation.result.validation_passed}}"
    notebook_task:
      notebook_path: /Repos/inventory-risk-ml/notebooks/promote_models
    timeout_seconds: 600
```

### 8.3 Pipeline Schedule Summary

| Pipeline | Schedule | Trigger | Duration | Compute |
|----------|----------|---------|----------|---------|
| Feature Engineering | 3:00 AM daily | Cron | ~20 min | 2 workers |
| Batch Scoring | 3:30 AM daily | After features | ~15 min | 4 workers |
| Data Quality Check | 4:00 AM daily | After scoring | ~5 min | Shared cluster |
| Model Retraining | 2:00 AM Sunday | Cron | ~2 hrs | 2 workers + GPU |
| Model Validation | After training | Dependency | ~15 min | Shared cluster |

---

## 9. Monitoring & Alerting

### 9.1 Data Quality Metrics

```python
# File: notebooks/data_quality_check.py

def run_data_quality_checks():
    """Run data quality checks on ML predictions."""

    predictions = spark.read.table("sap_bdc.inventory_risk.ml_predictions") \
        .filter(F.col("prediction_date") == F.current_date())

    checks = {
        "row_count": predictions.count(),
        "null_risk_class": predictions.filter(F.col("risk_class").isNull()).count(),
        "null_severity": predictions.filter(F.col("severity_score").isNull()).count(),
        "invalid_probability": predictions.filter(
            (F.col("risk_probability") < 0) | (F.col("risk_probability") > 1)
        ).count(),
        "severity_out_of_range": predictions.filter(
            (F.col("severity_score") < 0) | (F.col("severity_score") > 100)
        ).count()
    }

    # Risk distribution (should not be >80% any single class)
    risk_distribution = predictions.groupBy("risk_class").count().collect()
    total = sum([r["count"] for r in risk_distribution])
    max_pct = max([r["count"]/total for r in risk_distribution]) if total > 0 else 0
    checks["max_class_concentration"] = max_pct

    # Validation
    passed = (
        checks["row_count"] > 0 and
        checks["null_risk_class"] == 0 and
        checks["null_severity"] == 0 and
        checks["invalid_probability"] == 0 and
        checks["severity_out_of_range"] == 0 and
        checks["max_class_concentration"] < 0.8
    )

    return {"checks": checks, "passed": passed}
```

### 9.2 Model Performance Monitoring

```python
# File: notebooks/model_monitoring.py

def monitor_model_performance():
    """Track model performance metrics over time."""

    # Compare predictions to actuals (with 1-week lag)
    predictions = spark.read.table("sap_bdc.inventory_risk.ml_predictions") \
        .filter(F.col("prediction_date") == F.date_sub(F.current_date(), 7))

    actuals = spark.read.table("sap_bdc.inventory_risk.stock_status_v2") \
        .filter(F.col("week_end_date") == F.current_date())

    # Join and calculate metrics
    comparison = predictions.join(
        actuals.select("product_id", "location_id", "stock_status_warning"),
        on=["product_id", "location_id"],
        how="inner"
    )

    # Convert actuals to same format
    comparison = comparison.withColumn("actual_class",
        F.when(F.col("stock_status_warning").contains("understock"), "understock")
        .when(F.col("stock_status_warning").contains("overstock"), "overstock")
        .otherwise("normal"))

    # Calculate accuracy
    correct = comparison.filter(F.col("risk_class") == F.col("actual_class")).count()
    total = comparison.count()
    accuracy = correct / total if total > 0 else 0

    # Log to MLflow
    with mlflow.start_run(run_name="model_monitoring"):
        mlflow.log_metric("weekly_accuracy", accuracy)
        mlflow.log_metric("total_predictions", total)

    # Alert if accuracy drops
    if accuracy < 0.65:
        send_alert(f"Model accuracy dropped to {accuracy:.2%}. Consider retraining.")

    return {"accuracy": accuracy, "total": total}
```

### 9.3 Alerting Configuration

| Metric | Threshold | Action |
|--------|-----------|--------|
| Daily prediction count | < 90% of expected | Alert |
| Null predictions | > 0 | Alert + block |
| Model accuracy (weekly) | < 65% | Alert + trigger retrain |
| Serving endpoint latency | > 500ms p99 | Alert |
| Feature freshness | > 24 hours | Alert |

---

## 10. Testing Strategy

### 10.1 Unit Tests

```python
# File: tests/test_feature_engineering.py

import pytest
from pyspark.sql import SparkSession
from notebooks.feature_engineering import generate_features, calculate_consecutive_weeks

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder.master("local[2]").getOrCreate()

def test_stock_to_safety_ratio(spark):
    data = [("P1", "L1", 100, 50, 1, 2024)]
    df = spark.createDataFrame(data, ["product_id", "location_id", "projected_stock", "safety_stock", "week_num", "year"])
    result = generate_features(df, spark.createDataFrame([], schema="location_id STRING, product_id STRING, transportation_lead_time DOUBLE"))
    assert result.select("stock_to_safety_ratio").collect()[0][0] == 2.0

def test_consecutive_weeks_calculation(spark):
    data = [
        ("P1", "L1", "deficit", 1, 2024),
        ("P1", "L1", "deficit", 2, 2024),
        ("P1", "L1", "deficit", 3, 2024),
        ("P1", "L1", "normal", 4, 2024),
    ]
    df = spark.createDataFrame(data, ["product_id", "location_id", "stock_condition", "week_num", "year"])
    result = calculate_consecutive_weeks(df, "stock_condition", "deficit", "consecutive_deficit_weeks")
    weeks = [r["consecutive_deficit_weeks"] for r in result.orderBy("week_num").collect()]
    assert weeks == [1, 2, 3, 0]
```

### 10.2 Integration Tests

```python
# File: tests/test_integration.py

def test_end_to_end_pipeline(spark):
    """Test full pipeline from features to predictions."""

    # Load sample data
    stock_status = spark.read.table("sap_bdc.inventory_risk.stock_status_v2").limit(1000)

    # Generate features
    features = generate_features(stock_status)
    assert features.count() > 0
    assert "stock_to_safety_ratio" in features.columns

    # Score with models
    predictions = run_batch_scoring(features)
    assert predictions.count() == features.count()
    assert all(col in predictions.columns for col in ["risk_class", "severity_score"])

def test_ml_client_table_mode():
    """Test ML client reading from predictions table."""

    client = MLPredictionClient(mode="table")
    prediction = client.get_prediction(engine, "FG-100-001", "DC1000")

    assert prediction is not None
    assert prediction["risk_class"] in ["normal", "understock", "overstock"]
    assert 0 <= prediction["severity_score"] <= 100
```

### 10.3 Model Validation Tests

| Test | Criteria | Threshold |
|------|----------|-----------|
| F1 Score (understock) | Holdout set | ≥ 0.70 |
| F1 Score (overstock) | Holdout set | ≥ 0.65 |
| Severity MAE | Holdout set | ≤ 15 |
| Early Warning Accuracy | 4-week lookahead | ≥ 65% |
| Prediction Latency | p99 | ≤ 500ms |
| Feature Drift | PSI | ≤ 0.2 |

---

## 11. Appendix

### 11.1 Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABRICKS_HOST` | Databricks workspace URL | `adb-xxx.azuredatabricks.net` |
| `DATABRICKS_TOKEN` | PAT for API access | `dapi...` |
| `HANA_HOST` | SAP HANA host | `xxx.hana.cloud.sap` |
| `HANA_USER` | HANA username | `ML_SERVICE` |
| `HANA_PASSWORD` | HANA password | (secret) |
| `MLFLOW_TRACKING_URI` | MLflow server | `databricks` |

### 11.2 Databricks Cluster Configuration

```json
{
  "spark_version": "13.3.x-ml-scala2.12",
  "node_type_id": "Standard_DS4_v2",
  "num_workers": 4,
  "spark_conf": {
    "spark.databricks.delta.preview.enabled": "true",
    "spark.sql.shuffle.partitions": "200"
  },
  "spark_env_vars": {
    "MLFLOW_TRACKING_URI": "databricks"
  },
  "libraries": [
    {"pypi": {"package": "xgboost==1.7.6"}},
    {"pypi": {"package": "lightgbm==4.0.0"}},
    {"pypi": {"package": "shap==0.42.1"}},
    {"pypi": {"package": "tensorflow==2.13.0"}}
  ]
}
```

### 11.3 Verification Checklist

| # | Checkpoint | Owner | Status |
|---|------------|-------|--------|
| 1 | Delta Sharing from SAP BDC working | Data Eng | ✅ |
| 2 | Feature engineering notebook validated | ML Eng | 🔲 |
| 3 | Feature store table created | ML Eng | 🔲 |
| 4 | XGBoost F1 ≥ 0.70 on holdout | ML Eng | 🔲 |
| 5 | LSTM accuracy ≥ 70% | ML Eng | 🔲 |
| 6 | Severity MAE ≤ 15 | ML Eng | 🔲 |
| 7 | Predictions table schema validated | Data Eng | 🔲 |
| 8 | Delta Sharing write-back configured | Data Eng | 🔲 |
| 9 | Model serving endpoint live | ML Eng | 🔲 |
| 10 | Endpoint latency < 500ms | ML Eng | 🔲 |
| 11 | SAP BTP integration tested | App Dev | 🔲 |
| 12 | Daily workflow running | MLOps | 🔲 |
| 13 | Weekly retrain workflow running | MLOps | 🔲 |
| 14 | Monitoring dashboards created | MLOps | 🔲 |
| 15 | Alerting configured | MLOps | 🔲 |

### 11.4 Rollback Procedure

1. **Model Rollback**: In MLflow, transition previous version to "Production"
2. **Predictions Rollback**: Restore previous partition from Delta time travel
   ```sql
   RESTORE TABLE sap_bdc.inventory_risk.ml_predictions TO VERSION AS OF <version>
   ```
3. **Feature Rollback**: Re-run feature engineering with previous date
4. **Endpoint Rollback**: Update serving endpoint to previous model version

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-15 | ML Team | Initial draft |
| 1.1 | 2025-01-28 | ML Team | Added source data specs, SHAP module, integration details |
