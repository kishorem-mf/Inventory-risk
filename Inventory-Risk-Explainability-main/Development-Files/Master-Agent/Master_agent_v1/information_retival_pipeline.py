import os
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, inspect, text
import pandas as pd
import re
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy.engine.url import URL
import json
import base64
from urllib.parse import quote_plus
from datetime import datetime


# =========================
# LLM Credentials
# =========================
os.environ["AICORE_AUTH_URL"] = "https://btp-ai-developments-sl2f9ys4.authentication.eu10.hana.ondemand.com"
os.environ["AICORE_CLIENT_ID"] = "sb-38176009-b499-470f-a3b8-9cf98daac1d0!b503699|aicore!b540"
os.environ["AICORE_CLIENT_SECRET"] = "1ac5c77f-d5ac-4e2d-8c19-6ffc47113ec8$52U4q9NYAN-GBm23a2lm_SFVrzmWNhuS7l_qFXs4s4A="
os.environ["AICORE_BASE_URL"] = "https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com"
os.environ["AICORE_RESOURCE_GROUP"] = "default"

llm_model = 'gpt-4o-mini'
llm = init_llm(llm_model, max_tokens=4096, temperature=0)

# =========================
# Database Config
# =========================
host = "cfe32093-429a-4e59-87dc-9f3e4da891bf.hna2.prod-eu10.hanacloud.ondemand.com"
port = "443"
# schema_name = "INVT_XAI_RAW"
schema_name = "INVT_HISTORICAL_DATA"
user = "DBADMIN"
password = "Bcone@1234567"

user_enc = quote_plus(user)
password_enc = quote_plus(password)



# =========================
# Connect to DB
# =========================
connection_str = f"hana://{user_enc}:{password_enc}@{host}:{port}/?currentSchema={schema_name}"
engine = create_engine(connection_str)
inspector = inspect(engine)
tables = inspector.get_table_names(schema=schema_name)
db = SQLDatabase(engine=engine, schema=schema_name, include_tables=tables)


COLUMNS_TO_DELETE = {"COST": ['customerid', 'shiptolocationid', 'sourceid']}

TABLE_DESCRIPTIONS = {
    "DEMANDFULFILLMENT": "Captures planned vs. actual customer demand and fulfillment KPIs by product and customer weekly. Includes consensus_demand, customer_receipts, Late Deliveries, and Fulfillment Percentage.",
    "REVIEWDC": "Weekly inventory and supply flow metrics at distribution centers by product. Key figures: Stock on Hand, Incoming/Outgoing Supply, Safety Stock, Inventory Target, Projected Stock.",
    "REVIEWPLANT": "Tracks production metrics for finished goods at plants on a weekly basis. Monitors Planned Production, Incoming Receipts, Outgoing Supply, Stock on Hand, Safety Stock, Inventory Target.",
    "REVIEWVENDORS": "Monitors raw material flows from vendors into plants or DCs weekly. Tracks Outgoing Transport Supply, Stock on Hand, and dependent_demand at the vendor location.",
    "REVIEWCAPACITY": "Monitors plant production capacity, utilization, and resource usage weekly. Captures Capacity Supply, Utilization %, and Actual Usage of resources against production sources.",
    "REVIEWCOMPONENT": "Tracks raw material demand, receipts, and projected stock at plants weekly. Key figures include dependent_demand, incoming_transport_receipts, Safety Stock, Inventory Target.",
    "PRODUCT": "Master product reference table with product_id, material type, unit of measure, and descriptive information. Used for linking production, inventory, and demand data.",
    "LOCATIONSOURCE": "Defines transport rules between source and target locations for products. Contains shipment lot sizes, transport ratios, and transportation lead times between connected nodes.",
    "LOCATIONPRODUCT": "Maps products to specific locations in the network, indicating where each product is produced, stored, or moved.",
    "PRODUCTIONSOURCEHEADER": "Configures production sources per product and plant. Includes source identifiers, freeze horizons, production ratios, rounding values, and source type indicators.",
    "PRODUCTIONSOURCEITEM": "Lists raw material components required per production source. Links each production source to its input components for BOM purposes.",
    "PRODUCTIONSOURCERESOURCE": "Links production sources to the resources (work centers) used at the plant. Supports capacity and constraint analysis during production planning.",
    "COST": "Captures cost parameters by product, location, customer, and source, including inventory holding cost, production cost rates, transportation cost rates, and penalties for late delivery.",
    "INVENTORY_STOCK_STATUS": "Tracks snapshot of inventory status per product and location. Provides current stock levels and movement summaries.",
    "STOCK_STATUS": "Similar to Inventory Stock Status but structured for detailed tracking of stock position and transactional movements.",
    "PROFITMARGIN": "Analyzes profitability of products by customer and location. Provides insight on margins, costs, and price-related metrics for financial analysis.",
    "QUERY_HISTORY": "Internal table used to store previous user queries, LLM-generated SQL, and timestamps for context retention and conversational memory."
}


TABLE_LINKS = """
Table Links:
All of these tables link primarily by the product_id and location_id keys (and, where relevant, Resource ID or Source ID). 
By joining on those identifiers plus the weekly “bucket” columns, you can generate end-to-end extraction queries.

- Product ↔ Location Product ↔ Location Source  
  (Defines which product is at which site and how it moves through the network.)

- Production & Resources  
  Location Product ↔ Production Source Header ↔ Location Resource  
  (Ties products at a plant to specific production processes and the resources that execute them.)

- DEMAND_FULFILLMENT_HISTORY, REVIEW_DC_HISTORY, Review Plant, Review Vendors, Review Capacity, Review Component, Profit Margin  
  Each references product_id and one or more location or customer keys to pull in weekly values.

- Refer to the Production Source Item table to see which raw materials are consumed to create a finished good.
"""



TABLE_COLUMN_DESCRIPTIONS = [

    {
        "table_name": "STOCK_STATUS",
        "description": "Details weekly stock levels, safety stock, incoming receipts, total demand, and outgoing supply by product and location.",
        "columns": {
            "product_id": "Unique identifier of the finished-goods product.",
            "location_id": "Identifier for the location (e.g., DC, Plant).",
            "projected_stock": "Expected stock level after accounting for incoming and outgoing flows.",
            "safety_stock": "The minimum stock level to be maintained for that week.",
            "stock_on_hand": "Actual inventory available at the location.",
            "incoming_receipts": "Confirmed quantity of supplies received.",
            "total_demand": "Total demand from customers or other locations.",
            "outgoing_supply": "Quantity of supplies shipped out.",
            "location_type": "The type of location (e.g., DC, Plant).",
            "week_num": "The week number.",
            "year": "The year.",
            "week_end_date": "The date when the week ends.",
            "quarter": "The quarter.",
            "month": "The month ('July', 'August', 'September', 'October', 'November', 'December', 'January', 'February', 'March', 'April', 'May', 'June')",
            "stock_condition": "The condition of the stock (e.g., healthy, at risk).",
            "stock_status_warning": "A warning flag for the stock status. ex: overstock_instance_2 - second overstock instance",
            "transporatation_lead_time": "Time in weeks for transportation.",
            "minimum_transportation": "Minimum allowed shipment size.",
            "incremental_transportation": "Shipment batch size increment.",
            "production_lead_time": "Time in weeks for production.",
            "offset_stock": "Stock that is offset due to lead times."
        }
    },

    {
        "table_name": "DEMAND_FULFILLMENT_HISTORY",
        "description": "Tracks planned vs. actual customer demand and fulfillment metrics by product and customer group on a weekly bucket basis.",
        "columns": {
            "product_id": "Unique identifier of the finished-goods product (e.g., FG-100-001).",
            "customer_id": "Identifier for the customer or customer group (e.g., Customer Group 1000).",
            "Key Figure": {
                "description": "Type of metric.",
                "values": {
                    "consensus_demand": "Forecasted Demand that is to be fulfilled.",
                    "customer_receipts": "Confirmed quantity to be shipped to customers.",
                    "customer_demand_delivered_late": "Quantity of demand shipped later",
                    "demand_fulfillment": "Fulfilled demand as a percentage of total demand."
                }
            },
            "Total": "Cumulative total across all weeks.",
            "week_year": "Value for Week 22 of 2025."
        }
    },
    {
        "table_name": "REVIEW_DC_HISTORY",
        "description": "Contains inventory and flow metrics for each distribution center by product and week.",
        "columns": {
            "product_id": "Finished-goods product_identifier (e.g., FG-1000).",
            "location_id": "Distribution center code (e.g., DC1000).",
            "Key Figure": {
                "description": "Metric type.",
                "values": {
                    "dependent_demand": "Demand received from customer or connected distribution center",
                    "stock_on_hand": "Inventory currently available at the DC.",
                    "incoming_transport_receipts": "Confirmed quantity of supplies received by the current location",
                    "outgoing_supply": "Actual outgoing quantity of supplies from the DC.",
                    "safety_stock_sop": "Planned Stock to be maintained for that week.",
                    "inventory_target": "Targeted inventory volume for that week.",
                    "projected_stock_calculated": "Expected stock level after netting incoming and outgoing flows."
                }
            },
            "Total": "Cumulative total across all weeks.",
            "week_year": "Value for Week 22 of 2025."
        }
    },
    {
        "table_name": "REVIEW_PLANT_HISTORY",
        "description": "Captures production-related metrics at each plant for finished goods on a weekly basis.",
        "columns": {
            "product_id": "Finished-goods product_identifier (e.g., FG-1000).",
            "location_id": "Plant location code (e.g., PL1000).",
            "Key Figure": {
                "description": "Production metric type.",
                "values": {
                    "dependent_demand": "Demand received from connected distribution center or regional distribution center",
                    "stock_on_hand": "Inventory currently available at the Plant.",
                    "planned_production_receipts": "Planned quantity of goods to be produced.",
                    "incoming_production_receipts": "Confirmed volume of shipment received by the current location",
                    "Confirmed Production Receipts": "Confirmed production quantities. In the freeze horizon period, this is the quantity of supplies that can be produced.",
                    "outgoing_supply": "Quantity planned or confirmed to be shipped from plant.",
                    "safety_stock_sop": "Planned volume to be maintained for that week.",
                    "inventory_target": "Targeted inventory volume for that week.",
                    "projected_stock_calculated": "Expected stock level after netting incoming and outgoing flows."
                }
            },
            "Total": "Cumulative total across all weeks.",
            "week_year": "Value for Week 22 of 2025."
        }
    },
    {
        "table_name": "REVIEW_VENDORS_HISTORY",
        "description": "Tracks raw-material flows from vendors into the network (DCs and plants) weekly.",
        "columns": {
            "product_id": "Raw-material code (e.g., RM-1000).",
            "location_id": "Vendor code or receiving location (e.g., VEN1000).",
            "ship_to_location_id": "Destination location for shipments (e.g., PL1000).",
            "Key Figure": {
                "description": "Metric type.",
                "values": {
                    "dependent_demand": "Demand received from customer or connected distribution center",
                    "stock_on_hand": "Inventory currently available at the Vendor.",
                    "outgoing_transport_supply": "Raw materials quantity planned or confirmed to be shipped from Vendor."
                }
            },
            "Total": "Cumulative total across all weeks.",
            "week_year": "Value for Week 22 of 2025."
        }
    },
    {
        "table_name": "REVIEW_CAPACITY_HISTORY",
        "description": "Monitors production resource capacity, utilization, and usage by product at each plant.",
        "columns": {
            "location_id": "Plant code where the resource resides (e.g., PL1000).",
            "resource_id": "Identifier for the resource (e.g., RES1000_001).",
            "source_id": "Context identifier for usage (e.g., production variant PL1000_FG1000_PV1).",
            "product_id": "Finished product measured against capacity usage (e.g., FG-1000).",
            "Key Figure": {
                "description": "Capacity metric type.",
                "values": {
                    "capacity_supply": "Planned capacity available.",
                    "capacity_utilization": "Percentage of available capacity used.",
                    "capacity_usage_of_production_resource": "Actual usage of the resource."
                }
            },
            "Total": "Cumulative total across all weeks.",
            "week_year": "Value for Week 22 of 2025."
        }
    },
    {
        "table_name": "REVIEW_COMPONENT_HISTORY",
        "description": "Details weekly demand, receipts, and projected stock for raw-material components at plants.",
        "columns": {
            "product_id": "Raw-material component code (e.g., RM-1000).",
            "location_id": "Plant code where component is consumed (e.g., PL1000).",
            "Key Figure": {
                "description": "Metric type.",
                "values": {
                    "dependent_demand": "Demand received from customer or connected distribution center",
                    "incoming_transport_receipts": "Inbound material receipts from vendors.",
                    "projected_stock_calculated": "Expected stock level after netting incoming and outgoing flows.",
                    "safety_stock_sop": "Planned volume to be maintained for that week.",
                    "inventory_target": "Targeted inventory volume for that week."
                }
            },
            "Total": "Sum of all weekly buckets.",
            "week_year": "Value for Week 22 of 2025."
        }
    },
    {
        "table_name": "PRODUCT",
        "description": "Master list of products with unit of measure and material classification.",
        "columns": {
            "product_id": "Unique product_identifier (e.g., FG-1000).",
            "base_uom2": "Base unit of measure (e.g., EA for Each).",
            "material_type_id": "Material classification code (FERT, ROH, HALB).",
            "product_desc": "Descriptive name for the product."
        }
    },
    {
        "table_name": "LOCATION_SOURCE",
        "description": "Defines transportation rules (lot sizes, ratios, lead times) between locations for each product.",
        "columns": {
            "location_id": "Target location code (e.g., DC1000).",
            "product_id": "Product code (e.g., FG-1000).",
            "ship_from_loc_id": "Origin or Source location for shipments (e.g., RDC1000).",
            "incremental_transportation_lot_s": "Shipment batch size increment (e.g., 250).",
            "location_transport_ratio": "Volume adjustment ratio per shipment (e.g., 1.00).",
            "minimum_transportation_lot_size": "Minimum allowed shipment size (e.g., 1000).",
            "transportation_lead_time": "Transit time in weeks between ship-from and receiving location."
        }
    },
    {
        "table_name": "LOCATION_PRODUCT",
        "description": "Maps which products are handled at which locations.",
        "columns": {
            "location_id": "Location code (e.g., DC1000).",
            "product_id": "Product code available at the location.",
            "subnetwork_id": "Identifier for the operational subnetwork (e.g., ELECTROLUX)."
        }
    },
    {
        "table_name": "PRODUCTION_SOURCE_HEADER",
        "description": "Configuration of production sources for different products as per plant.",
        "columns": {
            "location_id": "Location where Product gets produced (e.g., PL1000).",
            "product_id": "Product produced at this source (e.g., FG-1000).",
            "source_id": "Unique identifier for this production variant which produces the product",
            "output_product_coefficient": "Units of output per process cycle.",
            "production_freeze_horizon_days": "Days before which production changes are locked.",
            "production_ratio": "Proportional factor in planning calculations.",
            "production_rounding_value": "Rounding rule value for production output.",
            "production_source_type_indicator": "Indicator of source type (e.g., 'P')."
        }
    },
    {
        "table_name": "LOCATION_RESOURCE",
        "description": "Assignment of resources to plants with capacity indicators and constraint types.",
        "columns": {
            "location_id": "Plant code (e.g., PL1000).",
            "resource_id": "Resource identifier (e.g., RES1000_001).",
            "Capacity Supply Expansion Time Series Indicator": "Flag if expanded capacity time series is active (X).",
            "Capacity Supply Time Series Indicator": "Flag if base capacity time series is active (X).",
            "Constraint Type": "Type of capacity constraint (e.g., 'F' for finite).",
            "Resource Type": "Category or numeric code for the resource type."
        }
    },
    {
        "table_name": "CUSTOMER_SOURCE",
        "description": "Defines transportation rules (lot sizes, ratios, lead times) between locations for each product.",
        "columns": {
            "customer_id": "Identifier for the customer or customer group (e.g., Customer Group 1000).",
            "location_id": "Target location code (e.g., DC1000).",
            "product_id": "Product code (e.g., FG-1000).",
            "customer_source_balance_receipt": " ",
            "customer_source_invalid": " ",
            "customer_source_supply_priority": " ",
            "customer_sourcing_ratio": " ",
            "customer_transportation_leadtime": " ",
            "forecast_consumption_mode": " ",
            "lot_size_round_val_for_cust_rcpt": " ",
            "mandatory_attribute_source_custo": " ",
            "threshold_rounding_cust_source": " ",
            "time_series_property": " ",
            "ts_transportation_sourcing_ind": " ",
        }
    },
    {
        "table_name": "PRODUCTION_SOURCE_ITEM",
        "description": "Details weekly demand, receipts, and projected stock for raw-material components at plants.",
        "columns": {
            "product_id": "Raw-material component code (e.g., RM-1000).",
            "source_id": "Unique identifier for this production variant which produces the product",
            "component_coefficient": " ",
            "component_offset": " ",
            "is_substitute_component": " ",
            "mand_atrr_production_source_item": " ",
            "minimum_shelf_life_value_for_com": " ",
            "source_item_id": " ",
            "ts_component_coefficient_ind": " ",
        }
    },
    {
        "table_name": "PRODUCTION_SOURCE_RESOURCE",
        "description": "Assignment of resources to plants with capacity indicators and constraint types.",
        "columns": {
            "resource_id": "Resource identifier (e.g., RES1000_001).",
            "source_id": "Unique identifier for this production variant which produces the product",
            "capacity_consumption_rate": " ",
            "capacity_consumption_rate_time_s": " ",
            "end_period_of_capacity_consumpti": " ",
            "fixed_production_capaconsumption": " ",
            "is_alt_resource": " ",
            "mandatory_attribute_for_producti": " ",
            "start_period_of_capacity_consump": " ",
        }
    },
    {
        "table_name": "PROFIT_MARGIN_HISTORY",
        "description": "Details weekly demand, receipts, and projected stock for raw-material components at plants.",
        "columns": {
            "product_id": "Raw-material component code (e.g., RM-1000).",
            "customer_id": "Identifier for the customer or customer group (e.g., Customer Group 1000).",
            "week_year": "Value for Week 22 of 2025.",
            "constrained_cogs": " ",
            "constrained_demand_rev": " ",
            "customer_receipts": "Confirmed quantity to be shipped to customers.",
            "planned_cost_per_product": " ",
            "planned_price": " ",
            "refresh_date": " ",
        }
    },
    {
        "table_name": "LOCATION",
        "description": "Maps which products are handled at which locations.",
        "columns": {
            "location_id": "Location code (e.g., DC1000).",
            "buffer_profile_set_id_for_creati": " ",
            "geo_latitude": " ",
            "geo_longitude": " ",
            "holding_cost_percentage": " ",
            "location": " ",
            "location_business_partner_id": " ",
            "location_priority": " ",
            "location_region": " ",
            "location_relevant_for_external_r": " ",
            "location_type": " ",
            "location_valid": " ",
            "plant_priority": " ",
        }
    }
]


# =========================
# Load Schema & Sample Data
# =========================
def get_hana_tables_and_sample_data(engine, schema_name, sample_limit=5):
    int_db_local = {}
    connection_str = f"hana://{user_enc}:{password_enc}@{host}:{port}/?currentSchema={schema_name}"
    engine = create_engine(connection_str)
    inspector = inspect(engine)
    tables = inspector.get_table_names(schema=schema_name)
    
    for table in tables:
        table_upper = table.upper()
        columns = inspector.get_columns(table, schema=schema_name)
        schema_info = [(col['name'], str(col['type'])) for col in columns]
        try:
            sql_query = f'SELECT TOP {sample_limit} * FROM "{schema_name}"."{table_upper}"'
            conn = engine.connect()
            df = pd.read_sql(text(sql_query), con=conn)
            if table_upper in COLUMNS_TO_DELETE:
                df.drop(columns=[col for col in COLUMNS_TO_DELETE[table_upper] if col in df.columns], inplace=True)
        except Exception as e:
            print(f"Error fetching data for table {table_upper}: {e}")
            df = pd.DataFrame()
        int_db_local[table_upper] = {"schema": schema_info, "data": df}
    return int_db_local

int_db = get_hana_tables_and_sample_data(engine, schema_name, sample_limit=2)
print(int_db.keys())

# =========================
# SQL Query Generation
# =========================
def clean_sql_code(sql_code: str) -> str:
    sql_code = re.sub(r'```.*?\n', '', sql_code, flags=re.DOTALL)
    sql_code = re.sub(r'"""sql', '', sql_code, flags=re.IGNORECASE)
    sql_code = sql_code.replace('"""', '').strip()
    return sql_code

def Text_SQLquery(llm_model, table_schema_dict, user_input, schema_name):
    formatted_schema = ""
    for table, info in table_schema_dict.items():
        desc = TABLE_DESCRIPTIONS.get(table, "No specific description available.")
        filtered_columns = [col for col in info["schema"] if col[0] not in COLUMNS_TO_DELETE.get(table, [])]
        columns = ", ".join([col_name for col_name, col_type in filtered_columns])
        formatted_schema += f'- Table {table}: {desc}. Columns: {columns}\n'

    prompt = (
    "You are an expert SAP HANA SQL generator.\n"
    "Write plain SQL without using quotes around table or column names.\n"
    "Use only the exact columns and table names provided.\n"
    "Generate only SELECT statements. Do not include markdown or code blocks.\n\n"
    # "First search in 'STOCK_STATUS' table, it is created after merging with rest of the tables.\n" 
    # f"If the user question is regardig the overstock or understock or greater than or less than look in {STOCK_STATUS} table."
    f"Analyze the provided table and column descriptions {TABLE_COLUMN_DESCRIPTIONS} "
    f"and the user request {user_input} to determine the relevant tables and columns.\n\n"
    f"{TABLE_LINKS}\n\n"
    f"Schema Definition:\n{formatted_schema}\n\n"
    f"User Request:\n{user_input}\n\n"
    f"Generate a single SQL query using the schema {schema_name}.\n"
    "Ignore any date filters or conditions mentioned in the request."
    "If the question is about the source, refer to ship_from_loc_id as the source."
    # "Always retrieve data for the latest date available in the column named refresh_date for all tables used.\n"
    "Strictly generate sql queries from the tables mentioned in the schema only"
    "If the query is regarding REVIEW_VENDORS_HISTORY, replace ship_to_loc_id with ship_to_location_id "
)
    sql_query_generation_prompt = f"""
    given user query, data schema and data description write an sql query that on excution extracts the relevant data from the database as reuqested by user
    Instructions
    Understand the user query, use aggregation functions only when needed. 
    If the question is about the source, refer to ship_from_loc_id as the source.
    If the query is regarding REVIEW_VENDORS_HISTORY, replace ship_to_loc_id with ship_to_location_id 
    user input:
    {user_input}

    TABLE_COLUMN_DESCRIPTIONS:
    {TABLE_COLUMN_DESCRIPTIONS}
    TABLE_LINKS:
    {TABLE_LINKS}

    Schema Definition:\n{formatted_schema}

    Respond only with the sql query. Do not include any other additonal text or explanation in the response.
    Do not include markdown or code blocks or any backticks

    """


    try:
        response = llm_model.invoke(sql_query_generation_prompt)
        sql_query = response.content.strip() if hasattr(response, "content") else response.strip()
        sql_query_clean = clean_sql_code(sql_query)
        return sql_query_clean
    except Exception as e:
        print(f"Error generating SQL: {e}")
        return None

def execute_sql_and_fetch(sql_query, engine):
    try:
        print("Executing SQL Query:\n", sql_query)
        conn = engine.connect()
        df = pd.read_sql(text(sql_query), con = conn)
        print(df)
        print('Extracted df shape :', df.shape)
        return df
    except Exception as e:
        print(f"Error executing SQL:\n{e}\nQuery was:\n{sql_query}")
        return pd.DataFrame()


def df_to_chat_response(llm_model, df, sql_query, user_query):
    df_str = df.to_string(index=False) if not df.empty else "No data found for this query."
    date_analysis = ""
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'week' in col.lower()]
    today = pd.Timestamp(datetime.now().date())
    if df.empty:
        date_analysis += "No data rows found to evaluate time-based patterns.\n"
    elif date_cols:
        for col in date_cols:
            try:
                parsed_dates = pd.to_datetime(df[col], errors='coerce')
                past_dates = parsed_dates[parsed_dates < today].dropna()
                future_dates = parsed_dates[parsed_dates > today].dropna()
                today_dates = parsed_dates[parsed_dates == today].dropna()
                date_analysis += f"\nIn column '{col}':\n"
                if not past_dates.empty:
                    date_analysis += f"- {len(past_dates)} rows with past dates.\n"
                if not today_dates.empty:
                    date_analysis += f"- {len(today_dates)} rows with today's date ({today.date()}).\n"
                if not future_dates.empty:
                    date_analysis += f"- {len(future_dates)} rows with future dates.\n"
            except Exception:
                continue
    else:
        date_analysis += "No date or week-related columns found for analysis.\n"


######
    analysis_context = (
    "You are an expert in Supply Chain Planning.\n"
    "Your primary job is to summarize the extracted DataFrame(s) provided by the SQL agent, "
    "always keeping the user's query and intent front-and-center. Answer the question precisely; "
    "do NOT default to describing unrelated weekly trends.\n\n"

    "INTENT-FIRST BEHAVIOR:\n"
    "- If the user asks for components/BOM (e.g., 'what are components required to produce FG-100-001 in plant PL1000'),\n"
    "  • Use Production Source Item (BOM) and/or Review Component extracts.\n"
    "  • Filter by product_id and location_id (and Source ID if present for the plant/variant).\n"
    "  • Return a concise list of components with: component_id, (optional) component_desc, qty per finished unit, UOM,\n"
    "    and any available scrap/alt info. Include Source ID/production variant if available.\n"
    "  • If the DataFrame only has per-cycle quantities and the production output per cycle is available "
    "    (e.g., Output Product Coefficient), compute qty_per_unit = component_qty / output_per_cycle. "
    "    Use only available fields; do not assume.\n"
    "  • If required BOM fields are missing from the provided extract, state that explicitly and provide a minimal, targeted SQL "
    "    the agent should run (do NOT invent values, and do NOT switch to generic stock narration).\n"
    "- If the user asks a generic question like 'Tell me about the data', then and only then provide a descriptive overview of the dataset.\n"
    "- For lead-times/lot-sizes/transport rules questions, use Location Source; for production parameters, use Production Source Header.\n"
    "- For stock/alert/trend questions, use df_stock_status_alert and related fact tables.\n\n"

    "ALWAYS USE THE EXTRACTED DATA ONLY:\n"
    "- Base every conclusion strictly on the values present in the extracted DataFrame(s).\n"
    "- Do not make assumptions or bring in outside knowledge.\n"
    "- Use specific numbers. If time-series is relevant to the question, compare current vs. previous weeks; otherwise, avoid off-topic trends.\n\n"

    "STOCK-ALERT SCOPE (only when asked about stock status):\n"
    "- 'Excess' = projected_stock > safety_stock; 'Deficit' = projected_stock < safety_stock.\n"
    # "- 4+ consecutive weeks form an instance; mention the Nth occurrence once, in natural language, "
    "  always with location and precise week_end_date range.\n\n"

    "REFERENCE FIELDS (use when relevant to the user question):\n"
    "- LOCATIONSOURCE: transportationleadtime, minimumtransportationlotsize, transportationroundingvalue, "
    "  transpshipmentfrequency, iotransportationfrozenwindow.\n"
    "- PRODUCTIONSOURCEHEADER: productionleadtime, minimumproductionlotsize, productionroundingvalue, "
    "  productionfreezehorizondays, productionsourceinvalid, output product coefficient.\n"
    "- BOM/COMPONENTS: (Production Source Item / Review Component): component_id, component_desc, qty_per_cycle, UOM, "
    "  scrap/alt indicators, source_id.\n"
    "- Stock-related: projected stock, safety stock, on hand stock, forecasted demand, dependent_demand, incoming receipts.\n\n"
    "- If the question is regarding, what is the source look for ship_from_loc_id, this is the source "

    "- SFG refers to intermediate products created from raw materials (RM)."
    "- The production flow typically follows this sequence: RM → SFG → FG"

    "MISSING-DATA PROTOCOL (for precise answers without hallucination):\n"
    "- If the specific fields needed to answer the question are not present in the extract, respond:\n"
    "  'The provided extract does not include <missing_fields> for <product_id> at <location_id>, so the components list cannot be confirmed.'\n"
    "  Then provide a minimal SQL pattern the agent should run, e.g.:\n"
    "  SELECT location_id, product_id, source_id, component_id, component_desc, qty_per_cycle, uom\n"
    "  FROM Production_Source_Item\n"
    "  WHERE product_id = '<PRODUCT>' AND location_id = '<PLANT>';\n\n"

    "TONE & OUTPUT:\n"
    "- Do not mentioned the tablenames, sql query in response."
    "- Concise, business-friendly, fact-based, and directly answers the user's question.\n"
    "- For components/BOM questions, output a bullet list or compact table of components; include Source ID if available.\n"
    "- Supply Chain Flow context (VEN ➝ Plant ➝ RDC ➝ DC) may be referenced only if it clarifies the specific answer.\n"
    
    "- Do NOT justify the user's claim if the data does not support it.\n"
    "- Always base the explanation strictly on what the data shows, even if it contradicts the user's question.\n"
    "- Generate a response strictly based on the user's query, using only the relevant parts of the imported data. Ignore any extra or unrelated data while answering."""
    "NOTE:If the extracted data has an entry with a ship-to location of 0 in REVIEW_VENDORS_HISTORY, please do not consider that for explanation."
)
    
    messages = [
        {"role": "system", "content": analysis_context},
        {"role": "user", "content": f"User's Question:\n{user_query}"},
        {"role": "user", "content": f"Executed SQL Query:\n{sql_query}"},
        {"role": "user", "content": f"Query Result:\n{df_str}"},
        {"role": "user", "content": f"Time-based Data Analysis:\n{date_analysis}"},
        {"role": "user", "content": "Please explain the situation based on the above information."}
    ]
    try:
        response = llm_model.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"Error in getting explanation from LLM: {e}")
        return "LLM failed to generate an explanation."

# =========================
# Execute Pipeline
# =========================


def execute_pipeline(user_input):
    if not user_input:
        return {"error": "User query is missing."}

    MAX_RETRIES = 3  # Maximum times to retry SQL generation

    # 1. Detect general prompts
    general_prompts = [
        r"tell me about (the )?(data|dataset|table|tables)",
        r"describe (the )?(data|dataset|table|tables)",
        r"what (tables|data) (are|is) considered",
        r"give (me )?an overview of (the )?(data|tables|dataset)",
        r"list (all )?(tables|datasets)"
    ]
    for pattern in general_prompts:
        if re.search(pattern, user_input.strip().lower()):
            schema_summary = ""
            for table, info in int_db.items():
                desc = TABLE_DESCRIPTIONS.get(table, "No specific description available.")
                columns = ", ".join([f"{col_name} ({col_type})" for col_name, col_type in info["schema"]])
                sample_rows = info["data"].head(2).to_dict(orient="records") if not info["data"].empty else []
                schema_summary += (
                    f"Table: {table}\n"
                    f"Description: {desc}\n"
                    f"Columns: {columns}\n"
                    f"Sample Data: {json.dumps(sample_rows, default=str)}\n\n"
                )

            overview_prompt = (
                "You are a data analyst. The user is asking for an overview of the available data.\n"
                "Summarize the schema, important fields, and data contents in plain English.\n"
                "Do not list raw dictionaries or JSON. Instead, explain in a human-friendly way.\n"
                "Highlight the purpose of each table and relationships between them if possible.\n\n"
                f"Schema & Data Summary:\n{schema_summary}\n\n"
                "Now, provide a clear and concise description of the dataset."
            )

            try:
                llm_response = llm.invoke(overview_prompt)
                explanation = llm_response.content.strip() if hasattr(llm_response, "content") else str(llm_response).strip()
            except Exception as e:
                explanation = f"Error generating data overview: {e}"

            return {"sql_query": None, "data_preview": None, "explanation": explanation}

    # 2. Retry SQL generation if invalid
    sql_query, sql_df = None, None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Generate SQL
            sql_query = Text_SQLquery(llm, int_db, user_input, schema_name)
            if not sql_query:
                raise ValueError("Generated SQL query is empty.")

            # Try executing SQL
            sql_df = execute_sql_and_fetch(sql_query, engine)

            # If execution works without exception, break loop
            break

        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"[Retry {attempt}/{MAX_RETRIES}] SQL execution failed: {e}. Regenerating query...")
            else:
                return {"error": f"Failed to generate a valid SQL query after {MAX_RETRIES} attempts. Last error: {e}"}

    # 3. Generate Explanation
    try:
        response_text = df_to_chat_response(llm, sql_df.drop_duplicates(), sql_query, user_input)
    except Exception as e:
        response_text = f"Error generating explanation: {e}"

    return {
        "sql_query": sql_query,
        "data_preview": sql_df.to_dict(orient='records'),
        "explanation": response_text
    }

    #return sql_query, sql_df, response_text

