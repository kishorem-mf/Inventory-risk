# Inventory Risk Explainability - Project Understanding

## Overview

An AI-powered **Supply Chain Digital Assistant** that explains inventory risk situations (understock/overstock) using multi-agent architecture. The system queries a SAP HANA database and provides natural language explanations for inventory imbalances.

---

## Architecture

### Multi-Agent System (LangGraph)

```
User Query → Supervisor Agent → Reasoning Agent (WHY questions)
                             → Information Retrieval Agent (data queries)
```

| Agent | Purpose | Example Query |
|-------|---------|---------------|
| **Supervisor** | Routes to appropriate agent | All queries |
| **Reasoning** | Explains WHY understock/overstock | "Why is FG-1000 overstocked?" |
| **Info Retrieval** | SQL queries for metrics | "What is projected stock?" |

---

## Core Components

| File | Purpose |
|------|---------|
| `main.py` | Flask API + Supervisor orchestration |
| `reasoning_agent_pipeline.py` | BOM tracing + root cause analysis |
| `data_retrieval_agent.py` | NL-to-SQL conversion agent |

---

## Data Model

### Master Tables
| Table | Description |
|-------|-------------|
| `Product` | Product catalog (FG, RM, HALB) |
| `Location` | Plants, DCs, RDCs, Vendors |
| `Location Source` | Transport rules between locations |
| `Production Source Header` | Production configurations |
| `Production Source Item` | Raw material requirements (BOM) |

### Transaction Tables (Weekly Buckets: W22 2025, etc.)
| Table | Key Figures |
|-------|-------------|
| `Review DC` | Stock on Hand, Projected Stock, Safety Stock |
| `Review Plant` | Production Receipts, Outgoing Supply |
| `Review Vendors` | Vendor Shipments |
| `Review Capacity` | Capacity Supply, Utilization |
| `Review Component` | Raw Material Stock |
| `Demand Fulfillment` | Consensus Demand, Fulfillment % |

### Stock Status Detection
- **Understock**: Projected Stock < Safety Stock (3+ consecutive weeks)
- **Overstock**: Projected Stock > Inventory Target (3+ consecutive weeks)

---

## Supply Chain Flow

```
Vendor (RM) → Plant (produces FG) → RDC → DC → Customer
```

**Frozen Horizon**: First 3 weeks cannot be adjusted (production locked)

---

## Inventory Imbalance Scenarios

### Understock (19 patterns)
| Category | Examples |
|----------|----------|
| Demand | Spike, forecast underestimation |
| Supply | Supplier delay, transport delay |
| Production | Capacity constraint, raw material shortage |
| Planning | Frozen horizon, target misalignment |

### Overstock (27 patterns)
| Category | Examples |
|----------|----------|
| Ordering | Over-ordering, duplicate orders |
| Supply | Early shipment, supplier MOQ |
| Production | Overshoot, prebuild |
| Demand | Forecast drop, order cancellation |

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| LLM | GPT-4o (SAP AI Core) |
| Framework | LangChain, LangGraph |
| Database | SAP HANA Cloud |
| API | Flask |
| Deployment | SAP BTP Cloud Foundry |
| Monitoring | LangSmith |

---

## Project Structure

```
Inventory-Risk-Explainability-main/
├── Data/                    # CSV data files
├── Development-Files/       # R&D notebooks & APIs
│   ├── Master-Agent/
│   ├── knowledge_graph/
│   └── APIs/
└── Production-files/        # Deployed code
    ├── Supervisor_langgraph/    # Main API (current)
    ├── Digital_Assistant_v1/    # Legacy
    └── DA_memory_retention/     # Memory variant
```

---

## API Usage

```http
POST /handle_query
Content-Type: application/json

{"query": "Why is FG-1000 understocked at DC1000?"}
```

**Response:**
```json
[
  {"response": "Natural language explanation..."},
  {"graph_base64": "base64_chart"},
  {"json_data": {"row1": {...}}}
]
```

---

## Key Configuration Files

| File | Purpose |
|------|---------|
| `data_schema.txt` | Database schema |
| `Data_description.txt` | Data dictionary |
| `stock_status_scenarios.txt` | Imbalance patterns |
| `supply_chain_context.txt` | Domain whitepaper |
| `manifest.yml` | CF deployment config |

---

## Environment Variables

```
AICORE_AUTH_URL      # SAP AI Core auth
AICORE_CLIENT_ID     # Client credentials
AICORE_CLIENT_SECRET # Client secret
AICORE_BASE_URL      # AI Core endpoint
LANGSMITH_API_KEY    # Tracing
```

---

## Key Concepts

| Term | Definition |
|------|------------|
| **Projected Stock** | Future stock = SOH + Inbound - Outbound |
| **Safety Stock** | Buffer for demand/supply variability |
| **Inventory Target** | Desired stock level |
| **Dependent Demand** | Demand derived from higher-level plans |
| **Key Figure** | Named metric (e.g., "Stock on Hand") |
| **Bucket Week** | Time period column (W22 2025) |

---

## Reasoning Logic Flow

1. **Query stock_status_v2** → Find alerts (overstock/understock instances)
2. **Trace BOM upstream** → Identify resources, raw materials, vendors
3. **Analyze constraints** → Capacity, lead times, lot sizes
4. **Match scenarios** → Compare against known patterns
5. **Generate explanation** → Natural language with data evidence

---

## Future Enhancements

- Memory retention across sessions
- Knowledge graph integration (Neo4j)
- Real-time alert streaming
- What-if scenario simulation
