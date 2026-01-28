# Inventory Risk ML Pipeline - Architecture Overview

| Document Info | |
|---------------|---|
| **Version** | 1.0 |
| **Last Updated** | 2025-01-28 |
| **Purpose** | Architecture diagram reference |

---

## 1. System Architecture

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    DATA SOURCES                                          │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                           SAP BDC (Business Data Cloud)                           │   │
│  │   • stock_status_v2    • review_dc       • review_plant                          │   │
│  │   • review_vendors     • location_source • production_source                     │   │
│  │   • lag_1_review_dc    • lag_1_review_plant                                      │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                              Delta Sharing (Zero-Copy Read)
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              SAP DATABRICKS (ML Platform)                                │
│                                                                                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                     │
│  │ FEATURE STORE   │    │ MODEL REGISTRY  │    │ MODEL SERVING   │                     │
│  │ ─────────────── │    │ ─────────────── │    │ ─────────────── │                     │
│  │ • ml_features   │───►│ • Risk Classifier│───►│ • REST API     │                     │
│  │ • Delta Table   │    │ • Early Warning  │    │ • /invocations │                     │
│  │ • Daily Refresh │    │ • Severity Scorer│    │ • Low latency  │                     │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘                     │
│           │                      │                      │                               │
│           ▼                      ▼                      │                               │
│  ┌────────────────────────────────────────┐             │                               │
│  │         BATCH SCORING JOB              │             │                               │
│  │  Features → Models → Predictions       │             │                               │
│  │  (Daily at 3 AM UTC)                   │             │                               │
│  └────────────────┬───────────────────────┘             │                               │
│                   │                                      │                               │
└───────────────────┼──────────────────────────────────────┼───────────────────────────────┘
                    │                                      │
         Delta Sharing (Write-Back)                  REST API
                    │                                      │
                    ▼                                      │
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              SAP BDC (ML Results)                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │  ml_predictions (Data Product)                                                    │   │
│  │  ─────────────────────────────                                                    │   │
│  │  • Risk Classification (normal/understock/overstock)                             │   │
│  │  • Severity Score (0-100)                                                         │   │
│  │  • Early Warning (weeks ahead)                                                    │   │
│  │  • SHAP Explainability (top 3 factors)                                           │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────┬────────────────────────────────────────────────┘
                                         │
                            Data Product Subscription
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
              ▼                          ▼                          ▼
┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐
│     SAP BTP App         │  │   SAP Analytics Cloud   │  │      SAP S/4HANA        │
│  (Flask API + Agents)   │  │    (Dashboards)         │  │   (ERP Integration)     │
└─────────────────────────┘  └─────────────────────────┘  └─────────────────────────┘
```

---

## 2. Component Architecture

### 2.1 Data Layer Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────┐    ┌────────────────────┐              │
│  │  MASTER DATA       │    │  TRANSACTIONAL     │              │
│  │  ──────────────    │    │  ──────────────    │              │
│  │  • Product         │    │  • Stock Status    │              │
│  │  • Location        │    │  • Review DC       │              │
│  │  • Location Product│    │  • Review Plant    │              │
│  │  • Location Source │    │  • Review Vendors  │              │
│  │  • Production Src  │    │  • Demand Fulfillment│            │
│  │  • Customer Source │    │  • Lag Tables      │              │
│  └────────────────────┘    └────────────────────┘              │
│                                                                  │
│              Data Flow: SAP BDC → Delta Sharing → Databricks    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Feature Engineering Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   FEATURE CATEGORIES                       │  │
│  ├───────────────┬───────────────┬───────────────────────────┤  │
│  │ STOCK         │ DEMAND        │ SUPPLY                    │  │
│  │ • Safety ratio│ • Rolling avg │ • Demand ratio            │  │
│  │ • Coverage    │ • Volatility  │ • Reliability index       │  │
│  │ • Velocity    │ • Trend       │ • Fulfillment gap         │  │
│  ├───────────────┼───────────────┼───────────────────────────┤  │
│  │ LEAD TIME     │ PATTERN       │ LOCATION                  │  │
│  │ • Total       │ • Consecutive │ • Type encoding           │  │
│  │ • Coverage    │   weeks       │ • Demand profile          │  │
│  │ • Exposure    │ • Risk trend  │ • Risk history            │  │
│  └───────────────┴───────────────┴───────────────────────────┘  │
│                                                                  │
│                  Output: 22 Engineered Features                  │
│                  Storage: Databricks Feature Store               │
│                  Refresh: Daily                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 ML Models Layer

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ML MODELS                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐  │
│  │   RISK CLASSIFIER    │  │   EARLY WARNING      │  │  SEVERITY SCORER │  │
│  │   ────────────────   │  │   ────────────────   │  │  ─────────────── │  │
│  │                      │  │                      │  │                  │  │
│  │   Model: XGBoost     │  │   Model: LSTM        │  │  Model: LightGBM │  │
│  │                      │  │                      │  │                  │  │
│  │   Input:             │  │   Input:             │  │  Input:          │  │
│  │   22 Features        │  │   12-week sequence   │  │  22 Features     │  │
│  │                      │  │                      │  │                  │  │
│  │   Output:            │  │   Output:            │  │  Output:         │  │
│  │   • normal           │  │   • Week 1 prob      │  │  • Score (0-100) │  │
│  │   • understock       │  │   • Week 2 prob      │  │  • Low/Med/High/ │  │
│  │   • overstock        │  │   • Week 3 prob      │  │    Critical      │  │
│  │   + probability      │  │   • Week 4 prob      │  │                  │  │
│  │                      │  │                      │  │                  │  │
│  │   Target: F1 ≥ 0.70  │  │   Target: Acc ≥ 70%  │  │  Target: MAE ≤15 │  │
│  └──────────────────────┘  └──────────────────────┘  └──────────────────┘  │
│                                                                              │
│                        Registry: MLflow Model Registry                       │
│                        Stages: Staging → Production                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Explainability Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                   SHAP EXPLAINABILITY                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│         ┌─────────────────────────────────────────┐             │
│         │        SHAP TreeExplainer               │             │
│         │                                         │             │
│         │   Model Prediction                      │             │
│         │         │                               │             │
│         │         ▼                               │             │
│         │   ┌─────────────┐                       │             │
│         │   │ SHAP Values │ → Feature Importance  │             │
│         │   └─────────────┘                       │             │
│         │         │                               │             │
│         │         ▼                               │             │
│         │   Top 3 Contributing Factors            │             │
│         │   • Factor name                         │             │
│         │   • Impact value                        │             │
│         │   • Direction (+/-)                     │             │
│         │   • Human-readable explanation          │             │
│         │                                         │             │
│         └─────────────────────────────────────────┘             │
│                                                                  │
│         Example Output:                                          │
│         1. "Stock level below safety threshold" (-0.45)          │
│         2. "Supply shortage relative to demand" (-0.28)          │
│         3. "Prolonged understock condition" (+0.22)              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow Architecture

### 3.1 Daily Batch Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DAILY BATCH PIPELINE (3 AM UTC)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STEP 1: Feature Engineering (20 min)                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ SAP BDC Tables ──► Delta Sharing ──► Feature Computation ──► Feature   │ │
│  │ (stock_status,                        (22 features)           Store    │ │
│  │  review_dc, etc.)                                                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                       │
│                                      ▼                                       │
│  STEP 2: Batch Scoring (15 min)                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Feature Store ──► Load Models ──► Score All ──► Generate ──► Write     │ │
│  │ (ml_features)     (MLflow)        Records      SHAP        Predictions │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                       │
│                                      ▼                                       │
│  STEP 3: Quality Check (5 min)                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Predictions ──► Null Checks ──► Range Checks ──► Distribution ──► Alert│ │
│  │ Table           (risk_class)    (severity)       Check          if fail│ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Weekly Retraining Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     WEEKLY RETRAINING PIPELINE (Sunday 2 AM)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────┐                                                      │
│  │ Prepare Training  │                                                      │
│  │ Data (Historical) │                                                      │
│  └─────────┬─────────┘                                                      │
│            │                                                                 │
│            ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                    PARALLEL TRAINING                         │            │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │            │
│  │  │   XGBoost    │  │    LSTM      │  │  LightGBM    │       │            │
│  │  │  Classifier  │  │ Early Warning│  │   Severity   │       │            │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │            │
│  │         │                 │                 │                │            │
│  └─────────┼─────────────────┼─────────────────┼────────────────┘            │
│            │                 │                 │                              │
│            └────────────────┬┴─────────────────┘                             │
│                             │                                                │
│                             ▼                                                │
│                    ┌─────────────────┐                                       │
│                    │ Model Validation │                                       │
│                    │ • F1 ≥ 0.70     │                                       │
│                    │ • Acc ≥ 70%     │                                       │
│                    │ • MAE ≤ 15      │                                       │
│                    └────────┬────────┘                                       │
│                             │                                                │
│                    ┌────────┴────────┐                                       │
│                    ▼                 ▼                                       │
│              [Pass]            [Fail]                                        │
│                │                  │                                          │
│                ▼                  ▼                                          │
│         ┌───────────┐      ┌───────────┐                                    │
│         │  Promote  │      │   Alert   │                                    │
│         │  Models   │      │   Team    │                                    │
│         └───────────┘      └───────────┘                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Integration Architecture

### 4.1 SAP BTP Application Integration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SAP BTP APPLICATION (Flask API)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       EXISTING COMPONENTS                            │    │
│  │                                                                      │    │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │    │
│  │  │  Supervisor  │───►│ Information  │───►│  Reasoning   │          │    │
│  │  │    Agent     │    │  Retrieval   │    │    Agent     │          │    │
│  │  │  (Router)    │    │    Agent     │    │ (Root Cause) │          │    │
│  │  └──────────────┘    └──────────────┘    └──────────────┘          │    │
│  │         │                                        │                   │    │
│  └─────────┼────────────────────────────────────────┼───────────────────┘    │
│            │                                        │                        │
│            │         ┌──────────────────────────────┘                        │
│            │         │                                                       │
│            ▼         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       NEW: ML INTEGRATION                            │    │
│  │                                                                      │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │                  ML Prediction Client                         │   │    │
│  │  │                                                               │   │    │
│  │  │   Mode: "table" ──► Read from ML_PREDICTIONS (via SAP BDC)   │   │    │
│  │  │   Mode: "api"   ──► Call Databricks Model Serving Endpoint   │   │    │
│  │  │                                                               │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  │                              │                                       │    │
│  │                              ▼                                       │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │              Enhanced Response Builder                        │   │    │
│  │  │                                                               │   │    │
│  │  │   Reasoning Output                                            │   │    │
│  │  │        +                                                      │   │    │
│  │  │   ML Prediction ──► Risk Class + Severity + Top Factors      │   │    │
│  │  │                                                               │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  │                                                                      │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Multi-Consumer Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA CONSUMERS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                    ┌────────────────────────┐                                │
│                    │    ML_PREDICTIONS      │                                │
│                    │    (Data Product)      │                                │
│                    └───────────┬────────────┘                                │
│                                │                                             │
│          Delta Sharing / Data Product Subscription                           │
│                                │                                             │
│     ┌──────────────────────────┼──────────────────────────┐                 │
│     │                          │                          │                  │
│     ▼                          ▼                          ▼                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   SAP BTP       │  │ SAP Analytics   │  │   SAP S/4HANA   │             │
│  │   Flask App     │  │    Cloud        │  │                 │             │
│  │                 │  │                 │  │                 │             │
│  │  Use Case:      │  │  Use Case:      │  │  Use Case:      │             │
│  │  • Chat UI      │  │  • Dashboards   │  │  • MRP Planning │             │
│  │  • Explain risk │  │  • KPIs         │  │  • Alerts       │             │
│  │  • Agent        │  │  • Trends       │  │  • Exceptions   │             │
│  │    responses    │  │  • Drill-down   │  │    workflow     │             │
│  │                 │  │                 │  │                 │             │
│  │  Access:        │  │  Access:        │  │  Access:        │             │
│  │  Table read     │  │  Subscription   │  │  API / Table    │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Output Schema

### 5.1 Predictions Data Product

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ML_PREDICTIONS TABLE STRUCTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PRIMARY KEYS                                                                │
│  ─────────────                                                               │
│  • product_id (STRING)                                                       │
│  • location_id (STRING)                                                      │
│  • week_num (INT)                                                            │
│  • year (INT)                                                                │
│                                                                              │
│  ML CLASSIFICATION                                                           │
│  ────────────────                                                            │
│  • risk_class: normal | understock | overstock                               │
│  • risk_probability: 0.0 - 1.0                                               │
│                                                                              │
│  SEVERITY SCORING                                                            │
│  ───────────────                                                             │
│  • severity_score: 0 - 100                                                   │
│  • severity_level: Low | Medium | High | Critical                            │
│                                                                              │
│  EARLY WARNING                                                               │
│  ─────────────                                                               │
│  • early_warning_weeks: 0 - 4                                                │
│  • early_warning_probability: 0.0 - 1.0                                      │
│                                                                              │
│  EXPLAINABILITY (SHAP)                                                       │
│  ────────────────────                                                        │
│  • top_factor_1 + impact                                                     │
│  • top_factor_2 + impact                                                     │
│  • top_factor_3 + impact                                                     │
│                                                                              │
│  METADATA                                                                    │
│  ────────                                                                    │
│  • model_version                                                             │
│  • prediction_timestamp                                                      │
│  • prediction_date (partition key)                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Deployment Architecture

### 6.1 Infrastructure Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DEPLOYMENT INFRASTRUCTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SAP BDC                              SAP DATABRICKS                         │
│  ─────────                            ──────────────                         │
│  ┌─────────────────┐                  ┌─────────────────┐                   │
│  │ Delta Lake      │◄────────────────►│ Unity Catalog   │                   │
│  │ Storage         │  Delta Sharing   │                 │                   │
│  │                 │                  │ ┌─────────────┐ │                   │
│  │ • Source tables │                  │ │ Workflows   │ │                   │
│  │ • ML predictions│                  │ │ (Jobs)      │ │                   │
│  └─────────────────┘                  │ └─────────────┘ │                   │
│                                       │                 │                   │
│                                       │ ┌─────────────┐ │                   │
│                                       │ │ MLflow      │ │                   │
│                                       │ │ Registry    │ │                   │
│                                       │ └─────────────┘ │                   │
│                                       │                 │                   │
│                                       │ ┌─────────────┐ │                   │
│                                       │ │ Model       │ │                   │
│                                       │ │ Serving     │ │                   │
│                                       │ └─────────────┘ │                   │
│                                       │                 │                   │
│                                       │ ┌─────────────┐ │                   │
│                                       │ │ Feature     │ │                   │
│                                       │ │ Store       │ │                   │
│                                       │ └─────────────┘ │                   │
│                                       └─────────────────┘                   │
│                                                                              │
│  SAP BTP                                                                     │
│  ────────                                                                    │
│  ┌─────────────────┐                                                        │
│  │ Cloud Foundry   │                                                        │
│  │                 │                                                        │
│  │ • Flask API     │                                                        │
│  │ • Agent System  │                                                        │
│  │ • SAP HANA      │                                                        │
│  │   Connection    │                                                        │
│  └─────────────────┘                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Compute Resources

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPUTE ALLOCATION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  DAILY JOBS                                                                  │
│  ──────────                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Feature Engineering  │  2 workers  │  Standard_DS3_v2  │  ~20 min  │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │ Batch Scoring        │  4 workers  │  Standard_DS4_v2  │  ~15 min  │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │ Quality Check        │  Shared     │  Existing cluster │  ~5 min   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  WEEKLY TRAINING                                                             │
│  ───────────────                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ XGBoost Training     │  2 workers  │  Standard_DS4_v2  │  ~1 hr    │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │ LSTM Training        │  2 workers  │  Standard_NC6 GPU │  ~1.5 hr  │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │ LightGBM Training    │  2 workers  │  Standard_DS4_v2  │  ~30 min  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  MODEL SERVING                                                               │
│  ─────────────                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Endpoint             │  Small      │  Scale-to-zero    │  <500ms   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Monitoring Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MONITORING & ALERTING                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     METRICS COLLECTION                               │    │
│  │                                                                      │    │
│  │  DATA QUALITY              MODEL PERFORMANCE         SYSTEM HEALTH  │    │
│  │  ────────────              ─────────────────         ───────────── │    │
│  │  • Row count               • Weekly accuracy          • Job duration│    │
│  │  • Null rate               • F1 scores               • Endpoint     │    │
│  │  • Range violations        • Prediction drift         latency      │    │
│  │  • Class distribution      • Feature drift           • Error rate   │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     ALERTING RULES                                   │    │
│  │                                                                      │    │
│  │  ┌─────────────────┬─────────────────┬─────────────────────────┐   │    │
│  │  │     METRIC      │    THRESHOLD    │        ACTION           │   │    │
│  │  ├─────────────────┼─────────────────┼─────────────────────────┤   │    │
│  │  │ Prediction count│ < 90% expected  │ Alert                   │   │    │
│  │  │ Null predictions│ > 0             │ Alert + Block           │   │    │
│  │  │ Model accuracy  │ < 65%           │ Alert + Trigger retrain │   │    │
│  │  │ Endpoint latency│ > 500ms p99     │ Alert                   │   │    │
│  │  │ Feature freshness│ > 24 hours     │ Alert                   │   │    │
│  │  └─────────────────┴─────────────────┴─────────────────────────┘   │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Implementation Phases

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      IMPLEMENTATION ROADMAP                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 1: Data Layer                              ✅ COMPLETED               │
│  ───────────────────                                                         │
│  • Delta Sharing from SAP BDC configured                                    │
│  • Source tables accessible in Databricks                                   │
│                                                                              │
│  PHASE 2: Feature Engineering                     🔲 1 WEEK                 │
│  ──────────────────────────                                                  │
│  • Implement 22 engineered features                                         │
│  • Register in Databricks Feature Store                                     │
│  • Set up daily refresh job                                                 │
│                                                                              │
│  PHASE 3: ML Models                               🔲 2 WEEKS                │
│  ─────────────────                                                           │
│  • Train XGBoost risk classifier                                            │
│  • Train LSTM early warning model                                           │
│  • Train LightGBM severity scorer                                           │
│  • Implement SHAP explainability                                            │
│  • Register models in MLflow                                                │
│                                                                              │
│  PHASE 4: Write-Back                              🔲 1 WEEK                 │
│  ───────────────────                                                         │
│  • Create ml_predictions Delta table                                        │
│  • Implement batch scoring pipeline                                         │
│  • Configure Delta Sharing for consumers                                    │
│                                                                              │
│  PHASE 5: Integration                             🔲 1 WEEK                 │
│  ────────────────────                                                        │
│  • Deploy model serving endpoint                                            │
│  • Integrate with SAP BTP Flask app                                         │
│  • Set up monitoring and alerting                                           │
│                                                                              │
│  TOTAL: ~5 WEEKS                                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Feature Storage** | Databricks Feature Store | Native integration, time-travel, lineage |
| **Risk Classifier** | XGBoost | Best performance on tabular data, SHAP support |
| **Early Warning** | LSTM | Sequence modeling for temporal patterns |
| **Severity Scorer** | LightGBM | Fast inference, good regression performance |
| **Explainability** | SHAP TreeExplainer | Model-agnostic, quantified feature importance |
| **Data Transfer** | Delta Sharing | Zero-copy, secure, real-time sync |
| **Model Registry** | MLflow | Native Databricks, versioning, staging |
| **Primary Integration** | Batch (table read) | Lower latency requirements, cost efficient |
| **Fallback Integration** | REST API | Real-time needs, edge cases |

---

## 10. Security Considerations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SECURITY ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  DATA ACCESS                                                                 │
│  • Delta Sharing with recipient authentication                              │
│  • Unity Catalog row/column level security                                  │
│  • SAP BDC data product permissions                                         │
│                                                                              │
│  API SECURITY                                                                │
│  • Databricks PAT for model serving                                         │
│  • TLS 1.3 for all connections                                              │
│  • SAP BTP destination configuration                                        │
│                                                                              │
│  SECRETS MANAGEMENT                                                          │
│  • Databricks Secrets scope                                                 │
│  • SAP BTP credential store                                                 │
│  • No hardcoded credentials                                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

*This document is intended for architecture diagram creation. For implementation details, refer to `LLD_Databricks_ML.md`.*
