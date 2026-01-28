import os
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, inspect, text
import pandas as pd
import re
from gen_ai_hub.proxy.langchain.init_models import init_llm
from sqlalchemy.engine.url import URL
import json
import base64
from urllib.parse import quote_plus
from datetime import datetime
from datetime import datetime, timedelta

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



def filter_max_refresh_date(df):
    """Filter DataFrame by max refresh_date and drop the column."""
    if "refresh_date" not in df.columns:
        return df  # return unchanged if no such column
    
    df["refresh_date"] = pd.to_datetime(df["refresh_date"], dayfirst=True, errors="coerce")
    max_date = df["refresh_date"].max()
    filtered_df = df[df["refresh_date"] == max_date]  
    # filtered_df = filtered_df.drop(columns=["refresh_date"])
    return filtered_df

tables_with_refresh_date = [
    "DEMAND_FULFILLMENT_HISTORY",
    "PROFIT_MARGIN_HISTORY",
    "REVIEW_CAPACITY_HISTORY",
    "REVIEW_COMPONENT_HISTORY",
    "REVIEW_DC_HISTORY",
    "REVIEW_PLANT_HISTORY",
    "REVIEW_VENDORS_HISTORY"
]
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
            "week_year": "Value for Week 22 of 2025.",
            "Week_num": "Contains week number",
            "year_num": "Contains year number",
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
            "week_year": "Value for Week 22 of 2025.",
            "Week_num": "Contains week number",
            "year_num": "Contains year number",
        }
    },
    {
        "table_name": "REVIEW_VENDORS_HISTORY",
        "description": "Tracks raw-material flows from vendors into the network (DCs and plants) weekly.",
        "columns": {
            "product_id": "Raw-material code (e.g., RM-1000).",
            "location_id": "Vendors or Vendor code or receiving location (e.g., VEN1000).",
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
            "week_year": "Value for Week 22 of 2025.",
            "Week_num": "Contains week number",
            "year_num": "Contains year number",
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
            "week_year": "Value for Week 22 of 2025.",
            "Week_num": "Contains week number",
            "year_num": "Contains year number",
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
            "week_year": "Value for Week 22 of 2025.",
            "Week_num": "Contains week number",
            "year_num": "Contains year number",
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
        "description": "Contains Source Information data. Defines transportation rules (lot sizes, ratios, lead times) between locations for each product.",
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
        "description": "Contains components required to produce, Configuration of production sources for different products as per plant.",
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
            "Week_num": "Contains week number",
            "year_num": "Contains year number",
            
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


def get_current_week_year():
    today = datetime.today()
    year, week_num, _ = today.isocalendar()  # ISO calendar: (year, week_num, weekday)
    return week_num, year

def get_week_range(offset_weeks: int):
    """
    Returns (week_num, year) for a week relative to current week.
    offset_weeks: positive for future weeks, negative for past weeks
    """
    target_date = datetime.today() + timedelta(weeks=offset_weeks)
    year, week_num, _ = target_date.isocalendar()
    return week_num, year


# --- Example usage ---
current_week, current_year = get_current_week_year()
print(f"Today is week {current_week}, year {current_year}")

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
            df = pd.read_sql(text(sql_query), con = conn)
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
    # "First search in 'STOCK_STATUS' table, then look in other tables.\n" 
    "Import all columns data."
    # f"If the user question is regardig the overstock or understock or greater than or less than look in {STOCK_STATUS} table."
    f"Analyze the provided table and column descriptions {TABLE_COLUMN_DESCRIPTIONS} "
    f"and the user request {user_input} to determine the relevant tables and columns.\n\n"
    f"{TABLE_LINKS}\n\n"
    f"Schema Definition:\n{formatted_schema}\n\n"
    f"User Request:\n{user_input}\n\n"
    f"Generate a single SQL query using the schema {schema_name}.\n"
    f"Interpret the user's query, identify the intent, and search {TABLE_COLUMN_DESCRIPTIONS} to find the most relevant table and column that match the query."
    
    "Ignore any date filters or conditions mentioned in the request."
    "If the question is about the source, refer to ship_from_loc_id as the source."
    "Strictly generate sql queries from the tables mentioned in the schema only"
    # "If the query is regarding vendors in REVIEW_VENDORS_HISTORY, replace ship_to_loc_id with ship_to_location_id and don't include refresh_date column"
    "If date range is not mentioned in the query, fetch all data"
    "If the query is regarding warning instances or overstock or understock include week_num, year, location, product in the query"
    "If possible, The query should contain product, location, week_num, year"
    # "If the query is about date range, then incude week_num and year"
    "If the query is regarding 'source information' import data from LOCATION_SOURCE table"
    "Overstock- 'Excess' = projected_stock > safety_stock"
    "Understock- 'Deficit' = projected_stock < safety_stock."
    "In all tables except for STOCK_STATUS table, whenever the query involves a date range, it should reference the week_num field for week and year_num field for year. for example: W40 20225 or week 40 of 2025 ;then, week_num=40, year_num=2025."
    "If the query is regarding, COMPONENTS REQUIRED to produce then import data from PRODUCTION_SOURCE_HEADER table only."

    "If the query is regarding vendors, replace ship_to_loc_id with ship_to_location_id and don't import refresh_date column"
    "If the query is regarding vendors info. import only these columns product_id, location_id, ship_to_location_id."

    "If the query is regarding total demand and NO location in query, import data from DEMAND_FULFILLMENT_HISTORY table, total demand then interpret total demand as total consensus demand (consensus_demand) if location is not mentioned also import refresh_date, week_num, year_num, customer_id columns and DON'T import location_id column"
    "If the query is regarding total demand, import data from REVIEW_VENDORS_HISTORY table , total demand then interpret total demand as total dependent demand (dependent_demand) if location is mentioned, also import refresh_date column and DON'T import week_year column"
    
    "If the query is regarding total demand and query specifies a location,  import data from REVIEW_VENDORS_HISTORY table, Map 'total demand' to dependent demand (dependent_demand), also import refresh_date column and DON'T import week_year column"
    "If the query is regarding total demand and query does NOT specify a location, import data from DEMAND_FULFILLMENT_HISTORY table, Map 'total demand' to total consensus demand (consensus_demand), also import refresh_date column and DON'T import location_id column"
    
    "If the query is regarding total_demandDEMAND_FULFILLMENT_HISTORY, include refresh_date in the response else no."
    "If the query is regarding total_demandPROFIT_MARGIN_HISTORY, include refresh_date in the response else no."
    "If the query is regarding total_demandREVIEW_CAPACITY_HISTORY, include refresh_date in the response else no."
    "If the query is regarding total_demandREVIEW_COMPONENT_HISTORY, include refresh_date in the response else no."
    "If the query is regarding total_demandREVIEW_DC_HISTORY, include refresh_date in the response else no."
    "If the query is regarding total_demandREVIEW_PLANT_HISTORY, include refresh_date in the response else no."
    "If the query is regarding total_demandREVIEW_VENDORS_HISTORY include refresh_date in the response else no."
    "If the query is regarding total_demandDEMAND_FULFILLMENT_HISTORY, include refresh_date in the response else no."
    
    "Lot Size Rule: When the query is regarding Lot Size, always return the Minimum Lot Size and the Incremental Lot Size, along with the Ship-From Location and Ship-To Location."
    "If the query is regarding lot size we have between abc and xyz for all the products, ALWAYS import minimum_transportation_lot_size, incremental_transportation_lot_s, Ship-From Location and Ship-To Location"

    "If the query is regarding capacity supply available for those resources, AlWAYS import/include plant, location, week in sql query"

    "If the query is regarding External receipt, please include week_num, year_num."

    f"""
    When the user asks for data within a relative time range (e.g., next 6 weeks, past 2 weeks),
    first determine the current week number and year based on today's date.
    Today is week {current_week} of year {current_year}.
    Then calculate the target week range by adding or subtracting the requested number of weeks from the current week.
    Always filter results by comparing against the week_num and year columns, using the format week_num, year_num or year.
    Ensure the query translates the relative weeks into exact week_num and year values before retrieving data.
    """

"""NOTE: 

Date Filter Rules by Table:
- No week/year filter (no such columns):
CUSTOMER_SOURCE, LOCATION, LOCATION_PRODUCT, LOCATION_SOURCE, PRODUCT, PRODUCTION_SOURCE_HEADER, PRODUCTION_SOURCE_ITEM, PRODUCTION_SOURCE_RESOURCE.

- Use week_num + year_num:
DEMAND_FULFILLMENT_HISTORY, PROFIT_MARGIN_HISTORY, REVIEW_CAPACITY_HISTORY, REVIEW_COMPONENT_HISTORY, REVIEW_DC_HISTORY, REVIEW_PLANT_HISTORY, REVIEW_VENDORS_HISTORY.

- Use week_num + year:
STOCK_STATUS.

- ALWAYS IMPORT/INCLUDE refresh date (refresh_date) if the table name is DEMAND_FULFILLMENT_HISTORY, PROFIT_MARGIN_HISTORY, REVIEW_CAPACITY_HISTORY, REVIEW_COMPONENT_HISTORY, REVIEW_DC_HISTORY, REVIEW_PLANT_HISTORY, REVIEW_VENDORS_HISTORY.


Instruction to the LLM:
When a user query involves a week-based date range, apply week filters only to tables that have week fields:
- If the target table has week_num and year_num, filter on both.
- If it has week_num and year, filter on both.
- If the table lacks these fields, do not add week/year filters for that table.
- IMPORT week_num, year_num, year if exist in the table.

- IMPORTANT: IMPORT refresh date (refresh_date) if it exists in the table, else do not include it.
Tables containing refresh_date:[DEMAND_FULFILLMENT_HISTORY, PROFIT_MARGIN_HISTORY, REVIEW_CAPACITY_HISTORY, REVIEW_COMPONENT_HISTORY, REVIEW_DC_HISTORY, REVIEW_PLANT_HISTORY, REVIEW_VENDORS_HISTORY]
Tables not containing refresh_date:[CUSTOMER_SOURCE, LOCATION, LOCATION_PRODUCT, LOCATION_SOURCE, PRODUCT, PRODUCTION_SOURCE_HEADER, PRODUCTION_SOURCE_ITEM, PRODUCTION_SOURCE_RESOURCE, STOCK_STATUS]

"""

)


    try:
        response = llm_model.invoke(prompt)
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
    "Ensure that all mathematical queries are interpreted and answered with precise logical correctness, including additions (sum), substractions, divisions, multiplications,  comparisons such as greater than ( > ) and less than ( < )."
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
    "Overstock- 'Excess' = projected_stock > safety_stock; \n"
    "Understock- 'Deficit' = projected_stock < safety_stock.\n"
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

    # "MISSING-DATA PROTOCOL (for precise answers without hallucination):\n"
    # "- If the specific fields needed to answer the question are not present in the extract, respond:\n"
    # "  'The provided extract does not include <missing_fields> for <product_id> at <location_id>, so the components list cannot be confirmed.'\n"
    # "  Then provide a minimal SQL pattern the agent should run, e.g.:\n"
    # "  SELECT location_id, product_id, source_id, component_id, component_desc, qty_per_cycle, uom\n"
    # "  FROM Production_Source_Item\n"
    # "  WHERE product_id = '<PRODUCT>' AND location_id = '<PLANT>';\n\n"

    "TONE & OUTPUT:\n"
    "- Do not mentioned the tablenames, sql query in response."
    "- Concise, business-friendly, fact-based, and directly answers the user's question.\n"
    "- For components/BOM questions, output a bullet list or compact table of components; include Source ID if available.\n"
    "- Supply Chain Flow context (VEN ➝ Plant ➝ RDC ➝ DC) may be referenced only if it clarifies the specific answer.\n"
    
    "- Do NOT justify the user's claim if the data does not support it.\n"
    "- Always base the explanation strictly on what the data shows, even if it contradicts the user's question.\n"
    "- Generate a response strictly based on the user's query, using only the relevant parts of the imported data. Ignore any extra or unrelated data while answering.\n"
    # "- To find the total demand, you need sum or add up the values, Arthematic Addition. These should be accurate.\n"
    
    """NOTE:
    - If the extracted data has an entry with a ship-to location is 0 in REVIEW_VENDORS_HISTORY, Do NOT consider that for explanation and consider other values.
    - Components required to produce and/mean coeffients required are same.
    """
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
    sql_df = pd.DataFrame()
    sql_df1 = pd.DataFrame()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Generate SQL
            sql_query = Text_SQLquery(llm, int_db, user_input, schema_name)
            if not sql_query:
                raise ValueError("Generated SQL query is empty.")

            # Try executing SQL
            # sql_df = execute_sql_and_fetch(sql_query, engine)

            # Try executing SQL
            sql_df = execute_sql_and_fetch(sql_query, engine)
            
            # refresh_date column check:
            if "refresh_date" in sql_df.columns:
                sql_df1 = filter_max_refresh_date(sql_df)
            else:
                sql_df1 = sql_df
            
            print('sql_df1 shape :', sql_df1.shape)

            # If execution works without exception, break loop
            break

        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"[Retry {attempt}/{MAX_RETRIES}] SQL execution failed: {e}. Regenerating query...")
            else:
                return {"error": f"Failed to generate a valid SQL query after {MAX_RETRIES} attempts. Last error: {e}"}

    # 3. Generate Explanation
    try:
        response_text = df_to_chat_response(llm, sql_df1.drop_duplicates(), sql_query, user_input)  # .drop_duplicates()
    except Exception as e:
        response_text = f"Error generating explanation: {e}"


    return json.dumps(
        {
            "sql_query": sql_query,
            "data_preview": sql_df.to_dict(orient="records"),  # convert DataFrame
            "explanation": response_text
        },
        default=str # pretty-print
    )
    #return sql_query, sql_df, response_text
