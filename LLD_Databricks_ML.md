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

### 2.4 SAP BDC Prerequisites

> **Critical**: Data engineering in SAP Databricks requires source datasets to be published as Data Products in SAP Datasphere and registered in SAP Business Data Cloud (BDC) before development can begin.

#### 2.4.1 Architecture Principles

| Principle | Description |
|-----------|-------------|
| **BDC as Central Catalog** | SAP BDC acts as the governance layer, automatically exposing Datasphere data products to Databricks |
| **Auto-provisioned Delta Sharing** | No manual Delta Share creation required; share, recipient, security, and metadata are managed by BDC |
| **Read-Only Source Access** | Published data products appear as read-only Delta tables in BDC-managed Unity Catalog namespace |
| **Customer-Owned Derived Tables** | All transformations (feature engineering, ML prep) produce derived tables in customer-owned schemas |
| **Publish-Back Requirement** | Derived datasets for enterprise reuse must be published back to BDC as derived data products |

#### 2.4.2 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SAP DATASPHERE                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Source Tables → Published as Data Products (with business metadata)     │   │
│  │  • STOCK_STATUS_V2, REVIEW_DC_HISTORY, REVIEW_PLANT_HISTORY, etc.       │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │ Publish
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      SAP BUSINESS DATA CLOUD (BDC)                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Central Catalog & Governance Layer                                      │   │
│  │  • Auto-provisions Delta Sharing to Databricks                          │   │
│  │  • Manages security, lineage, and metadata (ORD/CSN)                    │   │
│  │  • No manual share/recipient configuration required                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │ BDC-Managed Delta Sharing (automatic)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SAP DATABRICKS                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  BDC-Managed Unity Catalog Namespace (READ-ONLY)                        │   │
│  │  • sap_bdc.inventory_risk.stock_status_v2                               │   │
│  │  • sap_bdc.inventory_risk.review_dc_history                             │   │
│  │  • sap_bdc.inventory_risk.review_plant_history                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                            │
│                                     ▼ Transform                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Customer-Owned Unity Catalog Schema (READ-WRITE)                       │   │
│  │  • inventory_risk_ml.ml_features (derived)                              │   │
│  │  • inventory_risk_ml.ml_predictions (derived)                           │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │ Publish as Derived Data Products
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      SAP BUSINESS DATA CLOUD (BDC)                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Derived Data Products (with ORD + CSN metadata)                        │   │
│  │  • ml_predictions → Available to Datasphere, SAC, BTP apps              │   │
│  │  • ml_features → Available for downstream ML consumers                  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### 2.4.3 Prerequisites Checklist

| # | Prerequisite | Owner | Status |
|---|--------------|-------|--------|
| 1 | Source tables published as Data Products in SAP Datasphere | Data Steward | 🔲 |
| 2 | Data Products registered in SAP BDC with business metadata | Data Steward | 🔲 |
| 3 | BDC-managed Unity Catalog namespace accessible in Databricks | Platform Admin | 🔲 |
| 4 | Customer-owned schema created for derived tables (`inventory_risk_ml`) | Data Eng | 🔲 |
| 5 | ORD/CSN metadata templates prepared for publish-back | Data Eng | 🔲 |

#### 2.4.4 Key Terminology

| Term | Definition |
|------|------------|
| **Data Product** | A governed, reusable dataset published with business and technical metadata |
| **ORD Metadata** | Open Resource Discovery - business metadata (description, owner, tags) |
| **CSN Metadata** | Core Schema Notation - technical schema definition |
| **BDC-Managed Namespace** | Unity Catalog namespace automatically created by BDC for shared data products |
| **Derived Data Product** | Dataset created in Databricks and published back to BDC for enterprise consumption |

---

## 3. Source Data Specification

### 3.1 Data Products from SAP BDC

All source tables are accessed via **BDC-managed Delta Sharing** from catalog: `sap_bdc.inventory_risk`

> **Important - BDC-Managed Access**:
> - Source tables are published as Data Products in SAP Datasphere
> - SAP BDC automatically provisions Delta Sharing to Databricks (no manual share creation)
> - Tables appear as read-only in BDC-managed Unity Catalog namespace
> - The current application uses SAP HANA schemas (`CURRENT_INVT`, `INVT_HISTORICAL_DATA`); these are mirrored as Data Products in BDC

#### 3.1.1 Schema Mapping

| HANA Schema | Table Type | Delta Sharing Catalog |
|-------------|------------|----------------------|
| `CURRENT_INVT` | Master data, Stock status | `sap_bdc.inventory_risk` |
| `INVT_HISTORICAL_DATA` | Review history, Cost, Demand | `sap_bdc.inventory_risk` |

#### 3.1.2 Master Data Tables

| Table | HANA Schema | Description |
|-------|-------------|-------------|
| `PRODUCT` | CURRENT_INVT | Product catalog |
| `LOCATION` | CURRENT_INVT | Supply chain nodes (PL/DC/RDC/VEN) |
| `LOCATION_PRODUCT` | CURRENT_INVT | Product-location mapping with safety stock |
| `LOCATION_SOURCE` | CURRENT_INVT | Transportation rules and lead times |
| `PRODUCTION_SOURCE_HEADER` | CURRENT_INVT | Production configuration and lead times |
| `PRODUCTION_SOURCE_ITEM` | CURRENT_INVT | Bill of Materials (BOM) components |
| `PRODUCTION_SOURCE_RESOURCE` | CURRENT_INVT | Resource assignments |
| `CUSTOMER_SOURCE` | INVT_HISTORICAL_DATA | Customer sourcing rules |

#### 3.1.3 Transactional Tables

| Table | HANA Schema | Description | Update Frequency |
|-------|-------------|-------------|------------------|
| `STOCK_STATUS_V2` | CURRENT_INVT | **Primary** - Stock levels, warnings, lag columns | Weekly |
| `REVIEW_DC_HISTORY` | INVT_HISTORICAL_DATA | DC pivoted key figures | Weekly |
| `REVIEW_PLANT_HISTORY` | INVT_HISTORICAL_DATA | Plant pivoted key figures | Weekly |
| `REVIEW_VENDORS_HISTORY` | INVT_HISTORICAL_DATA | Vendor material flows | Weekly |
| `REVIEW_CAPACITY_HISTORY` | INVT_HISTORICAL_DATA | Production capacity metrics | Weekly |
| `REVIEW_COMPONENT_HISTORY` | INVT_HISTORICAL_DATA | Component inventory metrics | Weekly |
| `DEMAND_FULFILLMENT_HISTORY` | INVT_HISTORICAL_DATA | Demand metrics | Weekly |
| `PROFIT_MARGIN_HISTORY` | INVT_HISTORICAL_DATA | Profitability metrics | Weekly |
| `COST` | INVT_HISTORICAL_DATA | Cost parameters | Weekly |

```python
# Databricks notebook - Data Access
# NOTE: These tables are auto-provisioned by BDC - no manual Delta Share setup required
# Tables are READ-ONLY in the BDC-managed Unity Catalog namespace

# Master Data (from CURRENT_INVT → BDC Data Products)
stock_status = spark.read.table("sap_bdc.inventory_risk.stock_status_v2")
location_source = spark.read.table("sap_bdc.inventory_risk.location_source")
production_source = spark.read.table("sap_bdc.inventory_risk.production_source_header")

# Review History Tables (from INVT_HISTORICAL_DATA → BDC Data Products)
review_dc = spark.read.table("sap_bdc.inventory_risk.review_dc_history")
review_plant = spark.read.table("sap_bdc.inventory_risk.review_plant_history")
review_vendors = spark.read.table("sap_bdc.inventory_risk.review_vendors_history")
review_capacity = spark.read.table("sap_bdc.inventory_risk.review_capacity_history")
review_component = spark.read.table("sap_bdc.inventory_risk.review_component_history")

# Analysis Tables (from INVT_HISTORICAL_DATA → BDC Data Products)
demand_fulfillment = spark.read.table("sap_bdc.inventory_risk.demand_fulfillment_history")
profit_margin = spark.read.table("sap_bdc.inventory_risk.profit_margin_history")
cost = spark.read.table("sap_bdc.inventory_risk.cost")
```

### 3.2 Primary Source: `STOCK_STATUS_V2`

This is the main table for ML training and scoring. Located in schema `CURRENT_INVT`.

> **Important**: Lag values (previous week's data) are stored as **columns within this table**, not as separate lag tables. This eliminates the need for separate `lag_1_review_dc` or `lag_1_review_plant` tables.

#### Core Columns

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
| `transportation_lead_time` | DOUBLE | Weeks for RDC→DC transport | Feature input |
| `production_lead_time` | DOUBLE | Weeks for production | Feature input |
| `offset_stock` | DOUBLE | projected_stock - safety_stock | Derived |
| `stock_condition` | STRING | excess/deficit/in-stock | **Target (current)** |
| `stock_status_warning` | STRING | normal/understock_instance_N/overstock_instance_N | **Target (label)** |

#### Lag Columns (Embedded in STOCK_STATUS_V2)

| Column | Type | Description | ML Usage |
|--------|------|-------------|----------|
| `lag_1_projected_stock` | DOUBLE | Previous week's projected stock | Feature input |
| `lag_1_stock_on_hand` | DOUBLE | Previous week's stock on hand | Feature input |
| `lag_1_dependent_demand` | DOUBLE | Previous week's demand | Feature input |
| `lag_1_supply_orders` | DOUBLE | Previous week's supply orders | Feature input |
| `lag_1_incoming_receipts` | DOUBLE | Previous week's receipts | Feature input |

### 3.3 Supporting Tables

All review tables are in schema `INVT_HISTORICAL_DATA` and use weekly bucket columns (e.g., `w30_2025`, `w31_2025`... `w29_2026`).

#### `REVIEW_DC_HISTORY` / `REVIEW_PLANT_HISTORY` (Pivoted Key Figures)

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

#### `REVIEW_VENDORS_HISTORY` (Vendor Material Flows)

| Key Figure | Description |
|------------|-------------|
| Dependent Demand | Downstream demand from vendors |
| Stock on Hand | Vendor inventory |
| Outgoing Transport Supply | Outbound shipments to locations |

#### `REVIEW_CAPACITY_HISTORY` / `REVIEW_COMPONENT_HISTORY`

| Table | Description |
|-------|-------------|
| `REVIEW_CAPACITY_HISTORY` | Production capacity metrics by location |
| `REVIEW_COMPONENT_HISTORY` | Component inventory levels for BOM |

#### `LOCATION_SOURCE` (Transportation Rules) - Schema: `CURRENT_INVT`

| Column | ML Usage |
|--------|----------|
| `transportation_lead_time` | Lead time feature |
| `minimum_transportation_lot_size` | Supply constraint feature |
| `incremental_transportation_lot_size` | Supply flexibility feature |

#### `PRODUCTION_SOURCE_HEADER` (Production Config) - Schema: `CURRENT_INVT`

| Column | ML Usage |
|--------|----------|
| `production_lead_time` | Production lead time feature |
| `source_id` | Production source identifier |

### 3.4 Data Volume Estimates

#### Source Tables (from SAP HANA)

| Table | Schema | Rows (est.) | Update Frequency |
|-------|--------|-------------|------------------|
| `STOCK_STATUS_V2` | CURRENT_INVT | ~500K | Weekly |
| `REVIEW_DC_HISTORY` | INVT_HISTORICAL_DATA | ~200K | Weekly |
| `REVIEW_PLANT_HISTORY` | INVT_HISTORICAL_DATA | ~100K | Weekly |
| `REVIEW_VENDORS_HISTORY` | INVT_HISTORICAL_DATA | ~50K | Weekly |
| `REVIEW_CAPACITY_HISTORY` | INVT_HISTORICAL_DATA | ~30K | Weekly |
| `REVIEW_COMPONENT_HISTORY` | INVT_HISTORICAL_DATA | ~80K | Weekly |
| `LOCATION_SOURCE` | CURRENT_INVT | ~10K | Static |
| `PRODUCTION_SOURCE_HEADER` | CURRENT_INVT | ~5K | Static |

#### ML Output Tables (to Databricks)

| Table | Rows (est.) | Update Frequency |
|-------|-------------|------------------|
| `ml_features` | ~500K | Daily |
| `ml_predictions` | ~500K | Daily |

---

## 4. Phase 2: Feature Engineering

> **Design Principle**: Features are derived from the existing `reasoning_agent_pipeline.py` L1/L2 reasoning logic to ensure ML predictions align with current business rules.

### 4.1 Feature Catalog (Code-Aligned)

#### 4.1.1 Core Risk Features (from STOCK_STATUS_V2)

These features directly map to the current rule-based detection logic.

| Feature | Formula | Source Logic | L1/L2 Mapping |
|---------|---------|--------------|---------------|
| `stock_to_safety_ratio` | `projected_stock / safety_stock` | Core detection: <1 = deficit, >1 = excess | All L1 reasons |
| `soh_to_safety_ratio` | `stock_on_hand / safety_stock` | Overstock L1: "High Stock on Hand at Start" | Overstock L1 |
| `offset_stock` | `projected_stock - safety_stock` | Already in table, used for condition calc | All L1 reasons |
| `consecutive_deficit_weeks` | Streak count where `stock_condition = 'deficit'` | 4+ weeks triggers understock_instance | Pattern detection |
| `consecutive_excess_weeks` | Streak count where `stock_condition = 'excess'` | 4+ weeks triggers overstock_instance | Pattern detection |

#### 4.1.2 Demand & Forecast Features (from STOCK_STATUS_V2)

These features detect demand-related L1 reasons (overforecasting, underforecasting, demand spike).

| Feature | Formula | Source Logic | L1/L2 Mapping |
|---------|---------|--------------|---------------|
| `forecast_accuracy_ratio` | `lag_1_dependent_demand / total_demand` | Overstock: >1 = overforecasting; Understock: <1 = underforecasting | Overstock L1, Understock L1 |
| `demand_wow_change` | `(total_demand - lag_1_dependent_demand) / lag_1_dependent_demand` | Detects demand spike or drop | Understock L2: "Demand Spike" |
| `demand_spike_flag` | `1 if total_demand > lag_1_dependent_demand * 1.5 else 0` | Binary flag for unusual demand increase | Understock L1 |

#### 4.1.3 Supply Features (from STOCK_STATUS_V2)

These features detect supply-related L1 reasons (larger lot size, supplier delays, delayed POs).

| Feature | Formula | Source Logic | L1/L2 Mapping |
|---------|---------|--------------|---------------|
| `receipts_to_demand_ratio` | `incoming_receipts / total_demand` | Overstock L1: >1 = "Larger Lot Size" | Overstock L1 |
| `zero_receipt_weeks_flag` | `1 if incoming_receipts == 0 else 0` | Understock L1: "Longer Lead Time" (no receipts during instance) | Understock L1 |
| `supply_order_change` | `lag_1_supply_orders - supply_orders` | Understock L1: >0 = "Delayed Purchase Orders" (orders pushed ahead) | Understock L1 |
| `supply_reliability_idx` | `incoming_receipts / lag_1_supply_orders` | Measures if planned supply arrived | Understock L1 |

#### 4.1.4 Lead Time Features (from STOCK_STATUS_V2 + LOCATION_SOURCE)

These features detect lead time-related L2 reasons.

| Feature | Formula | Source Logic | L1/L2 Mapping |
|---------|---------|--------------|---------------|
| `total_lead_time` | `transportation_lead_time + production_lead_time` | BOM tracing sums these for pre-adjustment | Understock L2, Overstock L2 |
| `long_transport_lead_flag` | `1 if transportation_lead_time > 3 else 0` | Understock L2: "Longer Transportation Lead Time" | Understock L2 |
| `long_production_lead_flag` | `1 if production_lead_time > 3 else 0` | Understock L2: "Longer Production Lead Time" | Understock L2 |

#### 4.1.5 Capacity Features (from REVIEW_CAPACITY_HISTORY)

These features detect production-related L1/L2 reasons (production delays, resource unavailable).

| Feature | Formula | Source Logic | L1/L2 Mapping |
|---------|---------|--------------|---------------|
| `zero_capacity_weeks` | Count of weeks where `capacity_usage == 0` | Understock L1: "Production Delays" (4+ weeks) | Understock L1 |
| `capacity_utilization_avg` | `AVG(capacity_usage) OVER 4 weeks` | Low utilization indicates production issues | Understock L2 |

#### 4.1.6 Vendor Features (from REVIEW_VENDORS_HISTORY)

These features detect vendor-related L1 reasons (supplier delays, vendor capacity constraints).

| Feature | Formula | Source Logic | L1/L2 Mapping |
|---------|---------|--------------|---------------|
| `vendor_supply_flag` | `1 if vendor_max_external_receipt > 0 else 0` | Understock L1: 0 = "Supplier Delays" | Understock L1 |
| `vendor_zero_supply_weeks` | Count of weeks where vendor supply = 0 | Sustained vendor issues | Understock L2 |

#### 4.1.7 Location Features (from STOCK_STATUS_V2)

| Feature | Formula | Source Logic | L1/L2 Mapping |
|---------|---------|--------------|---------------|
| `is_dc` | `1 if location_type == 'DC' else 0` | Location type affects which L1/L2 reasons apply | All |
| `is_rdc` | `1 if location_type == 'RDC' else 0` | RDCs have transport but no production | All |
| `is_plant` | `1 if location_type in ['PL', 'P'] else 0` | Plants have production-specific L1 reasons | Understock L1: Production |

### 4.1.8 Feature Summary

| Category | Count | Source Tables |
|----------|-------|---------------|
| Core Risk | 5 | STOCK_STATUS_V2 |
| Demand/Forecast | 3 | STOCK_STATUS_V2 |
| Supply | 4 | STOCK_STATUS_V2 |
| Lead Time | 3 | STOCK_STATUS_V2, LOCATION_SOURCE |
| Capacity | 2 | REVIEW_CAPACITY_HISTORY |
| Vendor | 2 | REVIEW_VENDORS_HISTORY |
| Location | 3 | STOCK_STATUS_V2 |
| **Total** | **22** | |

### 4.2 Feature Engineering Pipeline

```python
# File: notebooks/feature_engineering.py
# Features aligned with reasoning_agent_pipeline.py L1/L2 logic

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, DoubleType

def calculate_consecutive_weeks(df, condition_col, condition_value, output_col):
    """
    Calculate streak of consecutive weeks meeting a condition.
    Maps to: 4+ weeks rule for understock/overstock detection.
    """
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


def count_zero_weeks(df, col_name, output_col, window_size=4):
    """Count weeks where a column is zero within a rolling window."""
    w = Window.partitionBy("product_id", "location_id").orderBy("year", "week_num").rowsBetween(-(window_size-1), 0)
    return df.withColumn(output_col,
        F.sum(F.when(F.col(col_name) == 0, 1).otherwise(0)).over(w))


def generate_features(stock_status_df, location_source_df, capacity_df=None, vendor_df=None):
    """
    Generate ML features aligned with reasoning_agent_pipeline.py L1/L2 logic.

    Args:
        stock_status_df: STOCK_STATUS_V2 from CURRENT_INVT
        location_source_df: LOCATION_SOURCE from CURRENT_INVT
        capacity_df: REVIEW_CAPACITY_HISTORY from INVT_HISTORICAL_DATA (optional)
        vendor_df: REVIEW_VENDORS_HISTORY from INVT_HISTORICAL_DATA (optional)

    Returns:
        spark DataFrame with code-aligned features
    """

    # Window specifications
    w = Window.partitionBy("product_id", "location_id").orderBy("year", "week_num")
    w_4w = w.rowsBetween(-3, 0)  # 4-week lookback (matches business rule)

    # Join lead time data from LOCATION_SOURCE
    df = stock_status_df.join(
        location_source_df.select("location_id", "product_id", "transportation_lead_time"),
        on=["location_id", "product_id"],
        how="left"
    )

    # ==================== CORE RISK FEATURES ====================
    # Maps to: Basic risk detection logic
    df = df \
        .withColumn("stock_to_safety_ratio",
            F.when(F.col("safety_stock") > 0,
                   F.col("projected_stock") / F.col("safety_stock"))
            .otherwise(F.lit(1.0))) \
        .withColumn("soh_to_safety_ratio",
            F.when(F.col("safety_stock") > 0,
                   F.col("stock_on_hand") / F.col("safety_stock"))
            .otherwise(F.lit(1.0)))
    # Note: offset_stock already exists in STOCK_STATUS_V2

    # ==================== DEMAND/FORECAST FEATURES ====================
    # Maps to: Overstock L1 "Overforecasting", Understock L1 "Underforecasting"
    df = df \
        .withColumn("forecast_accuracy_ratio",
            F.when(F.col("total_demand") > 0,
                   F.col("lag_1_dependent_demand") / F.col("total_demand"))
            .otherwise(F.lit(1.0))) \
        .withColumn("demand_wow_change",
            F.when(F.col("lag_1_dependent_demand") > 0,
                   (F.col("total_demand") - F.col("lag_1_dependent_demand")) / F.col("lag_1_dependent_demand"))
            .otherwise(F.lit(0.0))) \
        .withColumn("demand_spike_flag",
            F.when(F.col("total_demand") > F.col("lag_1_dependent_demand") * 1.5, 1)
            .otherwise(0))

    # ==================== SUPPLY FEATURES ====================
    # Maps to: Overstock L1 "Larger Lot Size", Understock L1 "Delayed POs"
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

    # ==================== LEAD TIME FEATURES ====================
    # Maps to: Understock L2 "Longer Lead Time", Overstock L2 "Longer Lead Time"
    df = df \
        .withColumn("total_lead_time",
            F.coalesce(F.col("transportation_lead_time"), F.lit(0)) +
            F.coalesce(F.col("production_lead_time"), F.lit(0))) \
        .withColumn("long_transport_lead_flag",
            F.when(F.col("transportation_lead_time") > 3, 1).otherwise(0)) \
        .withColumn("long_production_lead_flag",
            F.when(F.col("production_lead_time") > 3, 1).otherwise(0))

    # ==================== PATTERN FEATURES ====================
    # Maps to: 4+ consecutive weeks rule for instance detection
    df = calculate_consecutive_weeks(df, "stock_condition", "deficit", "consecutive_deficit_weeks")
    df = calculate_consecutive_weeks(df, "stock_condition", "excess", "consecutive_excess_weeks")

    # ==================== LOCATION FEATURES ====================
    # Maps to: Location-specific L1 reasons (plants have production logic)
    df = df \
        .withColumn("is_dc", F.when(F.col("location_type") == "DC", 1).otherwise(0)) \
        .withColumn("is_rdc", F.when(F.col("location_type") == "RDC", 1).otherwise(0)) \
        .withColumn("is_plant", F.when(F.col("location_type").isin(["PL", "P"]), 1).otherwise(0))

    # ==================== CAPACITY FEATURES (if available) ====================
    # Maps to: Understock L1 "Production Delays", L2 "Resource Capacity"
    if capacity_df is not None:
        capacity_agg = capacity_df.groupBy("product_id", "location_id", "year", "week_num").agg(
            F.sum(F.when(F.col("capacity_usage") == 0, 1).otherwise(0)).alias("zero_capacity_weeks"),
            F.avg("capacity_usage").alias("capacity_utilization_avg")
        )
        df = df.join(capacity_agg, on=["product_id", "location_id", "year", "week_num"], how="left")
        df = df.fillna(0, subset=["zero_capacity_weeks", "capacity_utilization_avg"])

    # ==================== VENDOR FEATURES (if available) ====================
    # Maps to: Understock L1 "Supplier Delays", L2 "Vendor Capacity Constraints"
    if vendor_df is not None:
        vendor_agg = vendor_df.groupBy("product_id", "location_id", "year", "week_num").agg(
            F.max(F.when(F.col("vendor_max_external_receipt") > 0, 1).otherwise(0)).alias("vendor_supply_flag"),
            F.sum(F.when(F.col("vendor_max_external_receipt") == 0, 1).otherwise(0)).alias("vendor_zero_supply_weeks")
        )
        df = df.join(vendor_agg, on=["product_id", "location_id", "year", "week_num"], how="left")
        df = df.fillna(0, subset=["vendor_supply_flag", "vendor_zero_supply_weeks"])

    # ==================== TARGET VARIABLE ====================
    df = df.withColumn("risk_label",
        F.when(F.col("stock_status_warning").contains("understock"), 1)
        .when(F.col("stock_status_warning").contains("overstock"), 2)
        .otherwise(0))  # 0 = normal, 1 = understock, 2 = overstock

    return df


# Feature columns for ML models (code-aligned)
FEATURE_COLS = [
    # Core Risk (5)
    "stock_to_safety_ratio", "soh_to_safety_ratio", "offset_stock",
    "consecutive_deficit_weeks", "consecutive_excess_weeks",
    # Demand/Forecast (3)
    "forecast_accuracy_ratio", "demand_wow_change", "demand_spike_flag",
    # Supply (4)
    "receipts_to_demand_ratio", "zero_receipt_weeks_flag", "supply_order_change", "supply_reliability_idx",
    # Lead Time (3)
    "total_lead_time", "long_transport_lead_flag", "long_production_lead_flag",
    # Capacity (2) - optional
    "zero_capacity_weeks", "capacity_utilization_avg",
    # Vendor (2) - optional
    "vendor_supply_flag", "vendor_zero_supply_weeks",
    # Location (3)
    "is_dc", "is_rdc", "is_plant"
]

# Core features (always available from STOCK_STATUS_V2)
CORE_FEATURE_COLS = [
    "stock_to_safety_ratio", "soh_to_safety_ratio", "offset_stock",
    "consecutive_deficit_weeks", "consecutive_excess_weeks",
    "forecast_accuracy_ratio", "demand_wow_change", "demand_spike_flag",
    "receipts_to_demand_ratio", "zero_receipt_weeks_flag", "supply_order_change", "supply_reliability_idx",
    "total_lead_time", "long_transport_lead_flag", "long_production_lead_flag",
    "is_dc", "is_rdc", "is_plant"
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

### 11.3 Data Engineering Task Checklist

#### Phase 1: Data Layer (✅ Complete)
| # | Task | Owner | Status |
|---|------|-------|--------|
| 1.1 | Configure Delta Sharing from SAP BDC | Data Eng | ✅ |
| 1.2 | Validate access to `sap_bdc.inventory_risk` catalog | Data Eng | ✅ |
| 1.3 | Test connectivity to HANA schemas | Data Eng | ✅ |

#### Phase 2: Feature Engineering Infrastructure
| # | Task | Owner | Status |
|---|------|-------|--------|
| 2.1 | Create `feature_engineering.py` notebook | Data Eng | 🔲 |
| 2.2 | Implement `calculate_consecutive_weeks()` function | Data Eng | 🔲 |
| 2.3 | Implement `generate_features()` with 22 code-aligned features | Data Eng | 🔲 |
| 2.4 | Join LOCATION_SOURCE for lead times | Data Eng | 🔲 |
| 2.5 | Join REVIEW_CAPACITY_HISTORY for capacity features | Data Eng | 🔲 |
| 2.6 | Join REVIEW_VENDORS_HISTORY for vendor features | Data Eng | 🔲 |
| 2.7 | Implement feature validation rules | Data Eng | 🔲 |
| 2.8 | Create Feature Store table `ml_features` | Data Eng | 🔲 |
| 2.9 | Configure merge/upsert mode for incremental updates | Data Eng | 🔲 |

#### Phase 4: Write-Back Infrastructure
| # | Task | Owner | Status |
|---|------|-------|--------|
| 4.1 | Create `ml_predictions` Delta table (Section 6.1 schema) | Data Eng | 🔲 |
| 4.2 | Configure partitioning by `prediction_date` | Data Eng | 🔲 |
| 4.3 | Enable autoOptimize and autoCompact | Data Eng | 🔲 |
| 4.4 | Set up Change Data Feed (CDF) for incremental sync | Data Eng | 🔲 |
| 4.5 | Create Delta Sharing share `inventory_risk_ml` | Data Eng | 🔲 |
| 4.6 | Add tables to share with recipients configured | Data Eng | 🔲 |

#### Phase 5: Batch Scoring Pipeline
| # | Task | Owner | Status |
|---|------|-------|--------|
| 5.1 | Create `batch_scoring.py` notebook | Data Eng | 🔲 |
| 5.2 | Implement model loading from MLflow registry | Data Eng | 🔲 |
| 5.3 | Create Spark UDFs for scoring | Data Eng | 🔲 |
| 5.4 | Write predictions to Delta with `replaceWhere` | Data Eng | 🔲 |

#### Phase 6: MLOps & Orchestration
| # | Task | Owner | Status |
|---|------|-------|--------|
| 6.1 | Create daily workflow `inventory_risk_ml_daily` | MLOps | 🔲 |
| 6.2 | Configure feature_engineering task (3:00 AM UTC) | MLOps | 🔲 |
| 6.3 | Configure batch_scoring task (depends on features) | MLOps | 🔲 |
| 6.4 | Configure data_quality_check task | MLOps | 🔲 |
| 6.5 | Create weekly workflow `inventory_risk_ml_weekly_retrain` | MLOps | 🔲 |
| 6.6 | Configure email notifications on failure | MLOps | 🔲 |

#### Phase 7: Monitoring & Quality
| # | Task | Owner | Status |
|---|------|-------|--------|
| 7.1 | Create `data_quality_check.py` notebook | Data Eng | 🔲 |
| 7.2 | Implement null/invalid prediction checks | Data Eng | 🔲 |
| 7.3 | Implement class concentration check (<80%) | Data Eng | 🔲 |
| 7.4 | Set up model performance tracking | Data Eng | 🔲 |
| 7.5 | Configure alerting thresholds | MLOps | 🔲 |

#### Phase 8: Testing
| # | Task | Owner | Status |
|---|------|-------|--------|
| 8.1 | Create unit tests for feature functions | QA | 🔲 |
| 8.2 | Create integration tests for pipeline | QA | 🔲 |
| 8.3 | Validate prediction schema compliance | QA | 🔲 |

### 11.4 ML Engineering Verification Checklist

| # | Checkpoint | Owner | Status |
|---|------------|-------|--------|
| 1 | Feature engineering notebook validated | ML Eng | 🔲 |
| 2 | XGBoost F1 ≥ 0.70 on holdout | ML Eng | 🔲 |
| 3 | LSTM accuracy ≥ 70% | ML Eng | 🔲 |
| 4 | Severity MAE ≤ 15 | ML Eng | 🔲 |
| 5 | Model serving endpoint live | ML Eng | 🔲 |
| 6 | Endpoint latency < 500ms | ML Eng | 🔲 |
| 7 | SAP BTP integration tested | App Dev | 🔲 |

### 11.5 Rollback Procedure

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
| 1.2 | 2025-01-29 | Data Eng | Updated Section 3: Corrected table names with _HISTORY suffix, added schema mapping (CURRENT_INVT, INVT_HISTORICAL_DATA), clarified lag columns are in STOCK_STATUS_V2 |
| 1.3 | 2025-01-29 | Data Eng | Updated Section 4: Replaced 21 theoretical features with 22 code-aligned features from reasoning_agent_pipeline.py L1/L2 logic |
| 1.4 | 2025-01-29 | Data Eng | Updated Section 11.3: Added comprehensive Data Engineering Task Checklist |
| 1.5 | 2025-01-29 | Data Eng | Added Section 2.4: SAP BDC Prerequisites - BDC-managed Delta Sharing architecture, data flow diagram, prerequisites checklist |
| 1.6 | 2025-01-29 | Data Eng | Updated Section 3.1: Clarified BDC auto-provisions Delta Sharing (no manual share creation) |
