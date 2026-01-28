# Inventory Risk Application - Inputs & Outputs

## Overview

The Inventory Risk Explainability application is a Flask-based API deployed on SAP BTP that detects understock and overstock instances in supply chain data and uses LLM-powered agents to explain the root causes.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│                         (SAP BTP Frontend App)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FLASK API SERVER                                   │
│                         /handle_query endpoint                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌──────────────────────────────────────────────┐   │
│  │  Query Refiner  │───►│           SUPERVISOR AGENT                    │   │
│  │  (Context-aware)│    │  (GPT-4o based task delegation)              │   │
│  └─────────────────┘    └──────────────────────────────────────────────┘   │
│                                    │                                         │
│                    ┌───────────────┴───────────────┐                        │
│                    ▼                               ▼                        │
│         ┌──────────────────┐           ┌──────────────────┐                │
│         │   Information    │           │    Reasoning     │                │
│         │   Retrieval      │           │    Agent         │                │
│         │   Agent          │           │                  │                │
│         │  (SQL queries)   │           │  (Root cause     │                │
│         │                  │           │   analysis)      │                │
│         └──────────────────┘           └──────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SAP HANA DATABASE                                    │
│                      Schema: CURRENT_INVT                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## INPUTS

### 1. API Request Payload

**Endpoint**: `POST /handle_query`

```json
{
  "query": "string - Natural language question from user",
  "UserID": "string - Unique user identifier",
  "ChatID": "string - Conversation session identifier",
  "chat_type": "string - One of: new_chat | same_chat | past_chat",
  "past_history": "array - Previous conversation context (for past_chat type)"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Natural language question about inventory risk |
| `UserID` | string | Yes | User identifier for tracking and personalization |
| `ChatID` | string | Yes | Session identifier for conversation continuity |
| `chat_type` | string | Yes | `new_chat` (reset memory), `same_chat` (continue session), `past_chat` (load history) |
| `past_history` | array | No | Previous Q&A pairs for context restoration |

### 2. Data Sources (SAP HANA)

#### Master Data Tables

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `product` | Product catalog | `product_id`, `product_desc`, `material_type_id` (FG/RM/SFG) |
| `location` | Supply chain nodes | `location_id`, `location_type` (PL/DC/RDC/VEN), `location` |
| `location_product` | Product-location mapping | `location_id`, `product_id`, `subnetwork_id`, `safety_stock` |
| `location_source` | Transportation rules | `location_id`, `product_id`, `ship_from_loc_id`, `transportation_lead_time`, `minimum_transportation_lot_size` |
| `production_source_header` | Production configuration | `location_id`, `product_id`, `source_id`, `production_lead_time` |
| `production_source_item` | Bill of Materials (BOM) | `product_id`, `source_id`, `component_coefficient` |
| `production_source_resource` | Resource assignments | `resource_id`, `source_id`, `capacity_consumption_rate` |
| `customer_source` | Customer data | `customer_id`, `location_id`, `product_id` |

#### Transactional Data Tables (Weekly Buckets)

| Table | Purpose | Key Figures |
|-------|---------|-------------|
| `stock_status_v2` | **Warning instance detection** | `projected_stock`, `safety_stock`, `stock_condition`, `stock_status_warning`, `offset_stock` |
| `review_dc` | Distribution Center metrics | Dependent Demand, Stock on Hand, Incoming Transport Receipts, Projected Stock, Safety Stock |
| `review_plant` | Plant production metrics | Dependent Demand, Planned Production Receipts, Confirmed Production Receipts, Projected Stock |
| `review_vendors` | Vendor raw material flows | Dependent Demand, Stock on Hand, Outgoing Transport Supply |
| `review_capacity` | Production capacity | Capacity Supply, Capacity Utilization, Capacity Usage |
| `review_component` | Component inventory | Dependent Demand, Incoming Transport Receipts, Projected Stock |
| `demand_fulfillment` | Customer demand metrics | Consensus Demand, Customer Receipts, Demand Fulfillment % |
| `lag_1_review_dc` | Previous week DC values | Same as review_dc (1-week lag) |
| `lag_1_review_plant` | Previous week Plant values | Same as review_plant (1-week lag) |

#### Key Data Definitions

**Stock Status Warning Logic** (in `stock_status_v2`):
- `stock_condition`:
  - `excess` → projected_stock > safety_stock
  - `deficit` → projected_stock < safety_stock
  - `in-stock` → projected_stock == safety_stock
- `stock_status_warning`:
  - `normal` → No warning
  - `overstock_instance_N` → 4+ consecutive weeks of excess
  - `understock_instance_N` → 4+ consecutive weeks of deficit

**Supply Chain Flow**:
```
Vendor (VEN) → Plant (PL) → RDC → DC → Customer
```

---

## OUTPUTS

### API Response Payload

```json
[
  {
    "response": "string - Natural language explanation"
  },
  {
    "graph_base64": "string - Base64 encoded PNG chart"
  },
  {
    "json_data": {
      "row1": {"col1": "val1", "col2": "val2"},
      "row2": {"col1": "val1", "col2": "val2"}
    }
  },
  {
    "Tabular_Data": [
      {"col1": "val1", "col2": "val2"},
      {"col1": "val1", "col2": "val2"}
    ]
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `response` | string | LLM-generated natural language answer with insights and explanations |
| `graph_base64` | string | Base64-encoded matplotlib chart (empty string if not applicable) |
| `json_data` | object | Row-indexed dictionary for UI table rendering |
| `Tabular_Data` | array | Records-oriented array for flexible table display |

### Output Components

#### 1. Text Response
- Direct answer to user query
- Root cause analysis for understock/overstock instances
- Data-backed explanations with specific values
- Markdown-formatted tables when multiple records

#### 2. Visualization (graph_base64)
- Auto-generated matplotlib charts
- Chart type selected based on data characteristics
- Color scheme: `#FFFFFF` background, `#4BC884`/`#22A6BD` markers
- Data labels on peaks and lows

#### 3. Tabular Data
- Extracted structured data from LLM response
- Maximum 15 rows in json_data
- Full dataset in Tabular_Data

---

## Agent Routing Logic

### Supervisor Agent Decision Matrix

| Query Type | Routed To | Examples |
|------------|-----------|----------|
| "Why is there understock/overstock?" | Reasoning Agent | Root cause analysis |
| "What is the demand for product X?" | Information Retrieval Agent | SQL query execution |
| "Show me stock levels at DC1000" | Information Retrieval Agent | Data retrieval |
| "Explain the overstock at location Y" | Reasoning Agent | Causal reasoning |

### Information Retrieval Agent
- Converts natural language to SAP HANA SQL
- Executes queries against CURRENT_INVT schema
- Returns structured data with business context

### Reasoning Agent
- Analyzes `stock_status_v2` warning instances
- Investigates supply chain factors (demand spikes, supply delays, capacity constraints)
- Generates chain-of-thought explanations

---

## Memory & Session Management

### Short-term Memory (Session)
- Maintained within Flask app runtime
- Stores conversation context for `same_chat` queries
- Reset on `new_chat`

### Long-term Memory (Database)
- Stored in `QUERY_HISTORY` table
- Fields: `UserID`, `ChatID`, `QueryDateTime`, `UserQuery`, `LLMResponse`
- Retrieved for `past_chat` queries

---

## Environment Configuration

| Variable | Description |
|----------|-------------|
| `HANA_HOST` | SAP HANA database hostname |
| `HANA_USER` | Database username |
| `HANA_PASSWORD` | Database password |
| `PORT` | Flask server port (default: 8080) |

---

## Current Limitations

1. **Rule-based Detection**: Stock warnings based on fixed 4-week consecutive threshold
2. **No Predictive Capability**: Cannot forecast future risks
3. **No Severity Scoring**: All warnings treated equally
4. **No Explainability Metrics**: No quantified feature importance (SHAP values)
5. **Batch Processing**: No real-time streaming capability

---

## Proposed ML Enhancement

See [LLD_Databricks_ML.md](../LLD_Databricks_ML.md) for the planned ML layer that adds:
- XGBoost risk classification with severity scores
- LSTM-based forward-looking predictions
- SHAP-based explainability
- Daily automated refresh pipeline
