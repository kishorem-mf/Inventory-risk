import warnings
warnings.filterwarnings("ignore")

from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine,inspect,text
from langchain.chains import create_sql_query_chain
from urllib.parse import quote_plus
import os
import re
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

from sqlalchemy import inspect
from datetime import datetime
from datetime import date, timedelta
import json
import time

from langchain.chains import create_sql_query_chain
from gen_ai_hub.proxy.langchain.init_models import init_llm, init_embedding_model

llm_model = 'gpt-4o'
llm = init_llm(llm_model, max_tokens=16384, temperature=0)

HANA_HOST = os.getenv("HANA_HOST")
HANA_USER = os.getenv("HANA_USER")
HANA_PASSWORD = os.getenv("HANA_PASSWORD")
schema_name = "CURRENT_INVT"
port = "443"

user_enc = quote_plus(HANA_USER)
password_enc = quote_plus(HANA_PASSWORD)
connection_str = f"hana://{user_enc}:{password_enc}@{HANA_HOST}:{port}/?currentSchema={schema_name}"

engine = create_engine(connection_str)

# import Data
## Importing data from HANA DB:

# def import_table_data(engine, schema_name, table_name):
#     # Use the engine to connect and read data
#     query = f'SELECT * FROM "{schema_name}"."{table_name}"'
#     conn = engine.connect()
#     df = pd.read_sql_query(text(query), con = conn)
#     return df

def import_table_data(engine, schema_name, table_name):
    """Fetch full table from the database."""
    query = text(f'SELECT * FROM "{schema_name}"."{table_name}"')
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(query, con=conn)
        if df.empty:
            print(f"No data fetched from {table_name}")
            return pd.DataFrame()
        return df
    except Exception as e:
        print(f"Error fetching {table_name}: {e}")
        return pd.DataFrame()

# Example usage
# df_stock_status = import_table_data(engine, "INVT_XAI", "STOCK_STATUS")
df_stock_status = import_table_data(engine, "CURRENT_INVT", "STOCK_STATUS_V2")
print(df_stock_status.shape)
df_stock_status.head(2)

df_stock_status['Alert'] = np.where((df_stock_status['stock_status_warning']=='normal'), 0, 1)
df_stock_status['Alert'].value_counts()

alert_df = df_stock_status[df_stock_status['Alert'] == 1]
alert_df.shape
alerts_lists = []
# Columns you want to extract combinations from (order matters)
columns_to_check = ['product_id', 'location_id','stock_status_warning']

# Store unique ordered combinations
unique_combinations = set()

for _, row in alert_df.iterrows():
    combo = tuple(row[col] for col in columns_to_check)  # keeps order
    unique_combinations.add(combo)

# Display results
for combo in unique_combinations:
    alerts_lists.append(list(combo))
# print(alerts_lists)
# print(len(alerts_lists))

############################# Trace Master Data ##################
# functions to call all the required data


# Script to get all associated resources (to check for capacity) and raw materials (to check for material shortage) for given product and location
# product df
product_df = import_table_data(engine, "CURRENT_INVT", "PRODUCT")

# prod_src_hdr_df
prod_src_hdr_df = import_table_data(engine, "CURRENT_INVT", "PRODUCTION_SOURCE_HEADER")

# prod_src_res_df
prod_src_res_df = import_table_data(engine, "CURRENT_INVT", "PRODUCTION_SOURCE_RESOURCE")

#prod_src_item_df
prod_src_item_df  = import_table_data(engine, "CURRENT_INVT", "PRODUCTION_SOURCE_ITEM")

#loc_src_df
loc_src_df = import_table_data(engine, "CURRENT_INVT", "LOCATION_SOURCE")

#location_df

location_df = import_table_data(engine, "CURRENT_INVT", "LOCATION")


# -------------------- VERBOSE TRACE FUNCTION WITH VENDOR INFO --------------------
def trace_bom_verbose_with_final_dest_and_vendors(product_id, location_id, visited=None, depth=0):
    if visited is None:
        visited = set()

    resources_used = set()
    raw_materials = set()
    vendors_associated = {}   # <-- NEW: store vendors for products/raws
    total_transport_lead = 0
    plant_id = None
    path_trace = []
    final_destinations = set()

    indent = "  " * depth
    #print(f"{indent}Tracing Product '{product_id}' from Location '{location_id}'")

    # Avoid infinite loops
    if (product_id, location_id) in visited:
        #print(f"{indent}Already visited {product_id} at {location_id}, skipping to avoid loop.")
        return resources_used, raw_materials, vendors_associated, plant_id, total_transport_lead, path_trace, final_destinations
    visited.add((product_id, location_id))

    # Step 1 - Move upstream until we reach a plant
    current_loc = location_id
    current_prod = product_id
    while not current_loc.startswith("PL"):
        row = loc_src_df[(loc_src_df["location_id"] == current_loc) &
                         (loc_src_df["product_id"] == current_prod)]
        if row.empty:
            #print(f"{indent}No upstream location found. Stopping.")
            return resources_used, raw_materials, vendors_associated, plant_id, total_transport_lead, path_trace, final_destinations
        trans_time = row.iloc[0]["transportation_lead_time"]
        ship_from = row.iloc[0]["ship_from_loc_id"]
        #print(f"{indent}From '{current_loc}' → '{ship_from}' (Transport Lead: {trans_time} weeks)")
        total_transport_lead += trans_time
        path_trace.append({"from": current_loc, "to": ship_from, "transport_lead": trans_time})
        current_loc = ship_from

    plant_id = current_loc
    #print(f"{indent}Reached Plant: {plant_id}")

    # Step 2 - Get Source ID & production lead time
    src_row = prod_src_hdr_df[(prod_src_hdr_df["location_id"] == plant_id) &
                              (prod_src_hdr_df["product_id"] == current_prod)]
    if src_row.empty:
        #print(f"{indent}No production source found for {current_prod} at {plant_id}")
        return resources_used, raw_materials, vendors_associated, plant_id, total_transport_lead, path_trace, final_destinations
    source_id = src_row.iloc[0]["source_id"]
    prod_lead_time = src_row.iloc[0]["production_lead_time"]
    #print(f"{indent}Source ID: {source_id}, Production Lead Time: {prod_lead_time} weeks")
    total_transport_lead += prod_lead_time
    path_trace.append({"plant": plant_id, "product": current_prod,
                       "source_id": source_id, "production_lead": prod_lead_time})

    # Step 3 - Get resources
    res_rows = prod_src_res_df[prod_src_res_df["source_id"] == source_id]
    if not res_rows.empty:
        found_res = res_rows["resource_id"].tolist()
        resources_used.update(found_res)
        #print(f"{indent}Resources Found: {found_res}")

    # Step 4 - Get parts
    part_rows = prod_src_item_df[prod_src_item_df["source_id"] == source_id]
    parts = part_rows["product_id"].tolist()
    # if parts:
    #     print(f"{indent}Parts Needed: {parts}")

    for part in parts:
        desc = product_df.loc[product_df["product_id"] == part, "product_desc"].values
        is_raw = desc.size > 0 and desc[0].lower().startswith("raw material")

        # --- NEW: find vendor supplying this part ---
        vendor_rows = loc_src_df[(loc_src_df["product_id"] == part)]
        vendor_rows = vendor_rows[vendor_rows["ship_from_loc_id"].isin(
            location_df[location_df["location_type"] == "V"]["location_id"]
        )]
        if not vendor_rows.empty:
            vendor_list = vendor_rows["ship_from_loc_id"].unique().tolist()
            vendors_associated[part] = vendor_list
            #print(f"{indent}Vendor(s) Found for {part}: {vendor_list}")

        if is_raw:
            raw_materials.add(part)
            #print(f"{indent}Raw Material Found: {part}")
        else:
            #print(f"{indent}Sub-component Found (SFG): {part} → Recursing...")
            sub_res, sub_raw, sub_vendors, _, _, sub_trace, _ = trace_bom_verbose_with_final_dest_and_vendors(part, plant_id, visited, depth + 1)
            resources_used.update(sub_res)
            raw_materials.update(sub_raw)
            vendors_associated.update(sub_vendors)
            path_trace.extend(sub_trace)

    # Step 5 - Trace final destinations from plant
    def find_final_destinations(start_loc, prod_id):
        visited_locs = set()
        finals = set()

        def dfs(loc, prod):
            if (loc, prod) in visited_locs:
                return
            visited_locs.add((loc, prod))

            rows = loc_src_df[(loc_src_df["ship_from_loc_id"] == loc) &
                              (loc_src_df["product_id"] == prod)]
            for _, r in rows.iterrows():
                next_loc = r["location_id"]
                next_prod = r["product_id"]
                #print(f"{indent} - {loc} → {next_loc} (Product: {next_prod})")

                if loc_src_df[loc_src_df["ship_from_loc_id"] == next_loc].empty:
                    finals.add(next_loc)
                else:
                    dfs(next_loc, next_prod)

        dfs(start_loc, prod_id)
        return finals

    if plant_id:
        #print(f"{indent}Tracing downstream from plant {plant_id} for product {product_id}...")
        final_destinations = find_final_destinations(plant_id, product_id)

    return resources_used, raw_materials, vendors_associated, plant_id, total_transport_lead, path_trace, final_destinations


# extracting shifted data based on lead times
# ---------- helpers ----------
def iso_monday(year: int, week: int) -> date:
    """ISO week-year -> Monday date"""
    return date.fromisocalendar(int(year), int(week), 1)

def date_to_iso_str(d: date) -> str:
    y, w, _ = d.isocalendar()
    return f"W{int(w):02d}_{int(y)}"

def parse_week_year_str(s: str) -> date:
    """Parse 'W01_2026' (case-insensitive) -> Monday date of that ISO week."""
    m = re.fullmatch(r"[Ww](\d{2})[_-](\d{4})", s.strip())
    if not m:
        raise ValueError(f"Invalid week_year '{s}' (expected like 'W05_2026').")
    week = int(m.group(1)); year = int(m.group(2))
    return iso_monday(year, week)

# ---------- core ----------
def filter_df2_by_df1_range(df1: pd.DataFrame,
                            df2: pd.DataFrame,
                            n_weeks: int,
                            week_col_df2: str = "week_year",
                            clamp: bool = False):
    """
    From df1 (WEEK_NUM, YEAR), take earliest & latest ISO week, subtract n weeks,
    and filter df2 (with a 'week_year' string column). If out of bounds, raise,
    unless clamp=True, in which case the range is clipped to df2's bounds.
    Returns (filtered_df, diagnostics_dict).
    """
    # df1 bounds (ensure numeric sort)
    df1_sorted = df1.sort_values(['year', 'week_num'],
                                 key=lambda s: s.astype(int))
    min_row = df1_sorted.iloc[0]
    max_row = df1_sorted.iloc[-1]

    start = iso_monday(min_row['year'], min_row['week_num'])
    end   = iso_monday(max_row['year'], max_row['week_num'])

    # subtract offset
    adj_start = start - timedelta(weeks=n_weeks)
    adj_end   = end   - timedelta(weeks=n_weeks)

    # df2 dates
    df2_dates = df2[week_col_df2].apply(parse_week_year_str)
    df2_min, df2_max = df2_dates.min(), df2_dates.max()

    # bounds check
    oob_msgs = []
    if adj_start < df2_min:
        oob_msgs.append(
            f"start {date_to_iso_str(adj_start)} < df2 min {date_to_iso_str(df2_min)}"
        )
    if adj_end > df2_max:
        oob_msgs.append(
            f"end {date_to_iso_str(adj_end)} > df2 max {date_to_iso_str(df2_max)}"
        )
    if oob_msgs and not clamp:
        raise ValueError("Out of bounds: " + "; ".join(oob_msgs))

    # clamp if requested
    filt_start = max(adj_start, df2_min)
    filt_end   = min(adj_end, df2_max)

    mask = (df2_dates >= filt_start) & (df2_dates <= filt_end)
    out = df2.loc[mask].copy()

    diagnostics = {
        "df1_original_start": date_to_iso_str(start),
        "df1_original_end":   date_to_iso_str(end),
        "adjusted_start(-n)": date_to_iso_str(adj_start),
        "adjusted_end(-n)":   date_to_iso_str(adj_end),
        "df2_range":          (date_to_iso_str(df2_min), date_to_iso_str(df2_max)),
        "used_filter_range":  (date_to_iso_str(filt_start), date_to_iso_str(filt_end)),
        "clamped":            bool(oob_msgs) and clamp
    }
    return out, diagnostics


#################### Get Current Time ######################################
def get_current_time_info():
    # Get current date
    today = datetime.today()
    
    # Get ISO calendar week and year
    year, week_num, _ = today.isocalendar()
    
    # Calculate week end date (Sunday of current ISO week)
    days_to_sunday = 7 - today.isoweekday()
    week_end_date = today + timedelta(days=days_to_sunday)
    
    # Get month and quarter (formatted as Q1, Q2, etc.)
    month = today.month
    quarter = f"Q{(month - 1) // 3 + 1}"
    
    # Build result dictionary
    result = {
        "week_num": week_num,
        "year": year,
        "week_end_date": week_end_date.strftime("%Y-%m-%d"),
        "month": month,
        "quarter": quarter
    }
    
    return result

###################### Part 1: Infromation Extraction Function ##############################

current_time = get_current_time_info()
def extract_info_query(query):
    extract_prompt_1 = f""" You are information processing module in supply chain analysis system which works on user query. Your task is to extract 3 points of information from the user query if present while following the instruction to process the information as required.
    The user query will be in format to where it's intention is to ask the reason behind understock or overstock. But your job is to only extract information from the query that you provide in your response. That response will be later be used by another module to filter the data and provide the reasoning to answer the user.  
    The user query may be rough and may have mistakes, you task is to estimate it correctly and provide the extracted data to best of your ability.
    The points of information fields to be extracted are -
    1. product_id: Unique identifier of the product (sample e.g., FG-100-001, FG-200-001).
    2. location_id: Identifier of the location center like Plant, regional distribution center or distribution center.(sample e.g., PL2000, DC1000, RDC1000).
    3. stock_issue: warning instance reason that the query is trying to investigate (overstock or understock).
    4. Time: User may provide time values. your task is to convert the time values to the mention sample format so the next module could work properly.
        a. time_instances: User may mention single or multiple instances for inquiry. Understand the user query and see if it's asking for any specific time(s). If multiple instances are asked then provide all those intances seprately.
            - week_num (integer): An integer (1-52) representing the ISO week number of the year.
            - week_end_date (date): Date (ISO format) representing the last day (typically Sunday) of the given week_num and year.
            - quarter (varchar): Calendar quarter (e.g., Q1, Q2) corresponding to the week_end_date.
            - month (varchar): Full month name (e.g., January, February) derived from the week_end_date.Make sure that its month name and not any integer month value.
            - year (integer): Four-digit calendar year associated with the week_num.
        b. time_range_instances: in some cases user may intent to provide a range of time where user wants to know the reason. In such case we need to extract start and end range of the time as per provided. There could one range or multiple such ranges provided. We need to extract all of them.
            - start_time:
                - week_num (integer): An integer (1-52) representing the ISO week number of the year.
                - week_end_date (date): Date (ISO format) representing the last day (typically Sunday) of the given week_num and year.
                - quarter (varchar): Calendar quarter (e.g., Q1, Q2) corresponding to the week_end_date.
                - month (varchar): Full month name (e.g., January, February) derived from the week_end_date. Make sure that its month name and not any integer month value.
                - year (integer): Four-digit calendar year associated with the week_num.
            - end_time
                - week_num (integer): An integer (1-52) representing the ISO week number of the year.
                - week_end_date (date): Date (ISO format) representing the last day (typically Sunday) of the given week_num and year.
                - quarter (varchar): Calendar quarter (e.g., Q1, Q2) corresponding to the week_end_date.
                - month (varchar): Full month name (e.g., January, February) derived from the week_end_date.Make sure that its month name and not any integer month value.
                - year (integer): Four-digit calendar year associated with the week_num.
 
 
    Instructions:
    1. First identify the points of information from user query. Understand the intent of query to figure out information for time realted extractions
    2. Query may provide multiple values for a field ,extract all of those values.
    3. VERY IMPORTANT THAT If the query has only the start time or the only the end time (i.e. not both start and end time the query), such cases must be responded as time_instances rather than time_range_instances.
    3. If any of the value is not present then respond its value as "Not provided by user" as per sample response format given.
    4. If and only if user asks query mentions/inquires related to current time then use this as current time : \n{current_time}\n
    """
    extract_prompt_2 = """Provide you response in the mentioned format.
    Sample Response format 1 -
    {
    "product_id" : "FG-100-001",
    "location_id": ["DC1000","DC2000"],
    "stock_issue" : "understock"
    "time_instances":
        {
        instance_1":
            {"week_num" : 30,
            "year": 2025,
            "week_end_date" :  "Not provided by user",
            "month": "Not provided by user",
            "quarter": "Not provided by user"
            }
        },
    "time_range_instances": "Not provided by user"
    }
 
    Sample Response format 2 -
    {
    "product_id" : ["FG-100-001","FG-200-001"],
    "location_id": "DC2000",
    "time_instances": "Not provided by user",
    "stock_issue" : "Not provided by user",
    "time_range_instances":
        {    
        instance_1":
            {
            start_time:  
                {"week_num" :  "Not provided by user",
                "year":  "Not provided by user",
                "week_end_date" :  "Not provided by user",
                "month": "July",
                "quarter": "Not provided by user"
                },
            end_time:  
                {"week_num" : "Not provided by user",
                "year": "Not provided by user",
                "week_end_date" :  "Not provided by user",
                "month": "September",
                "quarter": "Not provided by user"
                }
            },
        instance_2":
            {
            start_time:  
                {"week_num" : 30,
                "year": 2025,
                "week_end_date" :  "Not provided by user"
                "month": "Not provided by user"
                "quarter": "Not provided by user"
                }
            end_time:  
                {"week_num" : 30,
                "year": 2025,
                "week_end_date" :  "Not provided by user"
                "month": "Not provided by user"
                "quarter": "Not provided by user"
                }
            }
        }
    }
 
    """
    extract_prompt = extract_prompt_1 + extract_prompt_2

    messages = [
        {"role": "system", "content": extract_prompt},
        {"role": "user", "content": query}
    ]
    try:
        response = llm.invoke(messages)
        filter_str = response.content.strip()
        if filter_str.startswith("```json"):
            filter_str = filter_str.removeprefix("```json").removesuffix("```").strip()
        elif filter_str.startswith("```"):
            filter_str = filter_str.removeprefix("```").removesuffix("```").strip()
        # Convert string to dictionary
        filter_dict = json.loads(filter_str)
        response = filter_dict.copy()
    except Exception as e:
        response = {'error' : f"Error while getting response from LLM: {e}"}
    return response
    

######################## Helper function for Data Filter from information ###############################
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any
from datetime import datetime

def _to_list_if_needed(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        if value.strip().lower() == "not provided by user":
            return None
        if "," in value:
            return [v.strip() for v in value.split(",") if v.strip()]
        return [value.strip()]
    return [value]

def _make_week_index(year: int, week_num: int) -> int:
    return int(year) * 100 + int(week_num)

def _normalize_week_end_date_col(df: pd.DataFrame, col: str = 'week_end_date'):
    parsed_col = f"{col}_parsed"
    if parsed_col not in df.columns:
        df[parsed_col] = pd.to_datetime(df[col], errors='coerce')
    return parsed_col

def filter_alert_df(filter_dict: Dict[str, Any], alert_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Filters alert_df according to the extraction output (filter_dict).
    Returns a dict {'filtered_df': DataFrame, 'n_matches': int, 'applied_filters': dict}
    """
    df = alert_df.copy()

    # create lower-case helper columns for robust matching
    str_cols = ['product_id', 'location_id', 'stock_condition', 'stock_status_warning', 'month', 'quarter']
    for c in str_cols:
        if c in df.columns:
            df[f"{c}__lc"] = df[c].astype(str).str.lower()
        else:
            df[f"{c}__lc"] = ""

    week_end_parsed_col = _normalize_week_end_date_col(df, 'week_end_date')

    # --- parse incoming filter dict ---
    products = _to_list_if_needed(filter_dict.get('product_id', None))
    locations = _to_list_if_needed(filter_dict.get('location_id', None))
    stock_issue = filter_dict.get('stock_issue', None)
    if isinstance(stock_issue, str) and stock_issue.strip().lower() == "not provided by user":
        stock_issue = None
    if stock_issue is not None:
        stock_issue = str(stock_issue).strip().lower()

    time_instances = filter_dict.get('time_instances', None)
    if isinstance(time_instances, str) and time_instances.strip().lower() == "not provided by user":
        time_instances = None

    time_range_instances = filter_dict.get('time_range_instances', None)
    if isinstance(time_range_instances, str) and time_range_instances.strip().lower() == "not provided by user":
        time_range_instances = None

    overall_mask = pd.Series(True, index=df.index)

    # Product filter
    if products:
        prod_lc = [p.lower() for p in products]
        overall_mask &= df['product_id__lc'].isin(prod_lc)

    # Location filter
    if locations:
        locs_lc = [l.lower() for l in locations]
        overall_mask &= df['location_id__lc'].isin(locs_lc)

    # Stock issue (permissive matching)
    if stock_issue:
        cond_map = {
            'understock': ['deficit', 'understock'],
            'overstock': ['excess', 'overstock']
        }
        found_masks = []
        found_masks.append(df['stock_status_warning__lc'].str.contains(stock_issue, na=False))
        mapped = cond_map.get(stock_issue, [stock_issue])
        for m in mapped:
            found_masks.append(df['stock_condition__lc'].str.contains(m, na=False))
        mask_si = np.logical_or.reduce(found_masks)
        overall_mask &= mask_si

    # Collect time masks (OR across instances / ranges)
    time_masks = []

    # Helper: produce either integer week-index or ('mask', boolean_mask) or None
    def _get_week_index_from_timeblock(tb):
        if not isinstance(tb, dict):
            return None
        wk = tb.get('week_num', None)
        yr = tb.get('year', None)
        if wk not in (None, "Not provided by user") and yr not in (None, "Not provided by user"):
            try:
                return _make_week_index(int(yr), int(wk))
            except Exception:
                pass
        # week_end_date fallback
        wed = tb.get('week_end_date', None)
        if wed and wed != "Not provided by user":
            try:
                parsed = pd.to_datetime(wed, errors='coerce')
                if not pd.isna(parsed):
                    mask = df[week_end_parsed_col].dt.normalize() == parsed.normalize()
                    return ('mask', mask)
            except Exception:
                pass
        # month+year fallback => produce mask
        month = tb.get('month', None)
        yr = tb.get('year', None)
        if month and month != "Not provided by user":
            mon_lc = str(month).strip().lower()
            if yr not in (None, "Not provided by user"):
                try:
                    yr_i = int(yr)
                    mask = (df['month__lc'] == mon_lc) & (df['year'] == yr_i)
                    return ('mask', mask)
                except Exception:
                    pass
            mask = (df['month__lc'] == mon_lc)
            return ('mask', mask)
        return None

    # time_instances exact matches (weeks)
    if time_instances and isinstance(time_instances, dict):
        for inst_key, inst_val in time_instances.items():
            if not isinstance(inst_val, dict):
                continue
            wk = inst_val.get('week_num', None)
            yr = inst_val.get('year', None)
            if wk not in (None, "Not provided by user") and yr not in (None, "Not provided by user"):
                try:
                    wk_i = int(wk)
                    yr_i = int(yr)
                    mask = (df['week_num'] == wk_i) & (df['year'] == yr_i)
                    time_masks.append(mask)
                    continue
                except Exception:
                    pass
            # week_end_date
            wed = inst_val.get('week_end_date', None)
            if wed and wed != "Not provided by user":
                parsed = pd.to_datetime(wed, errors='coerce')
                if not pd.isna(parsed):
                    mask = df[week_end_parsed_col].dt.normalize() == parsed.normalize()
                    time_masks.append(mask)
                    continue
            # month/year fallback
            month = inst_val.get('month', None)
            if month and month != "Not provided by user":
                mon_lc = str(month).strip().lower()
                if inst_val.get('year', None) not in (None, "Not provided by user"):
                    try:
                        yr_i = int(inst_val.get('year'))
                        mask = (df['month__lc'] == mon_lc) & (df['year'] == yr_i)
                        time_masks.append(mask)
                        continue
                    except Exception:
                        pass
                mask = (df['month__lc'] == mon_lc)
                time_masks.append(mask)
                continue

    # time_range_instances (improved handling)
    if time_range_instances and isinstance(time_range_instances, dict):
        for inst_key, inst_val in time_range_instances.items():
            if not isinstance(inst_val, dict):
                continue
            start = inst_val.get('start_time', None)
            end = inst_val.get('end_time', None)
            start_idx = _get_week_index_from_timeblock(start) if start else None
            end_idx = _get_week_index_from_timeblock(end) if end else None

            # If both are mask types ('mask', boolean_series)
            def _wk_index_series():
                return df['year'].astype(int) * 100 + df['week_num'].astype(int)

            # Both masks
            if isinstance(start_idx, tuple) and isinstance(end_idx, tuple):
                start_mask = start_idx[1]
                end_mask = end_idx[1]
                wk_index = _wk_index_series()
                start_weeks = wk_index[start_mask]
                end_weeks = wk_index[end_mask]
                if (not start_weeks.empty) and (not end_weeks.empty):
                    s_i = int(start_weeks.min())
                    e_i = int(end_weeks.max())
                    if s_i > e_i:
                        s_i, e_i = e_i, s_i
                    mask = (wk_index >= s_i) & (wk_index <= e_i)
                    time_masks.append(mask)
                else:
                    # if one of them empty, fallback to union of masks to avoid dropping matches
                    combined = start_mask | end_mask
                    time_masks.append(combined)
                continue

            # start is mask, end is numeric
            if isinstance(start_idx, tuple) and isinstance(end_idx, (int, np.integer)):
                start_mask = start_idx[1]
                wk_index = _wk_index_series()
                start_weeks = wk_index[start_mask]
                if not start_weeks.empty:
                    s_i = int(start_weeks.min())
                    e_i = int(end_idx)
                    if s_i > e_i:
                        s_i, e_i = e_i, s_i
                    mask = (wk_index >= s_i) & (wk_index <= e_i)
                    time_masks.append(mask)
                else:
                    time_masks.append(start_mask)  # fallback
                continue

            # end is mask, start is numeric
            if isinstance(end_idx, tuple) and isinstance(start_idx, (int, np.integer)):
                end_mask = end_idx[1]
                wk_index = _wk_index_series()
                end_weeks = wk_index[end_mask]
                if not end_weeks.empty:
                    s_i = int(start_idx)
                    e_i = int(end_weeks.max())
                    if s_i > e_i:
                        s_i, e_i = e_i, s_i
                    mask = (wk_index >= s_i) & (wk_index <= e_i)
                    time_masks.append(mask)
                else:
                    time_masks.append(end_mask)
                continue

            # both numeric week index
            if isinstance(start_idx, (int, np.integer)) and isinstance(end_idx, (int, np.integer)):
                s_i = int(start_idx)
                e_i = int(end_idx)
                if s_i > e_i:
                    s_i, e_i = e_i, s_i
                wk_index = df['year'].astype(int) * 100 + df['week_num'].astype(int)
                mask = (wk_index >= s_i) & (wk_index <= e_i)
                time_masks.append(mask)
                continue

            # one-side numeric (treat as single instance per your rule)
            if isinstance(start_idx, (int, np.integer)):
                wk_index = df['year'].astype(int) * 100 + df['week_num'].astype(int)
                time_masks.append(wk_index == int(start_idx))
                continue
            if isinstance(end_idx, (int, np.integer)):
                wk_index = df['year'].astype(int) * 100 + df['week_num'].astype(int)
                time_masks.append(wk_index == int(end_idx))
                continue

            # If any side was a mask but none of above matched, append whatever mask exists
            if isinstance(start_idx, tuple):
                time_masks.append(start_idx[1])
            if isinstance(end_idx, tuple):
                time_masks.append(end_idx[1])

    # Combine time filters
    if time_masks:
        combined_time_mask = np.logical_or.reduce(time_masks)
        overall_mask &= combined_time_mask

    filtered_df = df[overall_mask].copy()
    # remove helper cols
    helper_cols = [c for c in filtered_df.columns if c.endswith('__lc') or c.endswith('_parsed')]
    filtered_df.drop(columns=[c for c in helper_cols if c in filtered_df.columns], inplace=True, errors='ignore')

    result = {
        'filtered_df': filtered_df,
        'n_matches': int(filtered_df.shape[0]),
        'applied_filters': {
            'products': products,
            'locations': locations,
            'stock_issue': stock_issue,
            'time_instances_provided': bool(time_instances),
            'time_range_instances_provided': bool(time_range_instances)
        }
    }
    return result
############ Gather Data function For Guided ################

## Gathering other datapoints than stock status
### 1. Capacity Data
def gather_capacity_data(df_stock_status_alert,final_result):
    # df_stock_status = import_table_data(engine, "INVT_XAI", "STOCK_STATUS")
    df_capacity = import_table_data(engine, "INVT_HISTORICAL_DATA", "REVIEW_CAPACITY_HISTORY")
    # Filtering on latest date
    df_capacity['refresh_date'] = pd.to_datetime(df_capacity['refresh_date'], dayfirst=True, errors='coerce')
    latest_date = df_capacity['refresh_date'].max()
    df_capacity = df_capacity[df_capacity['refresh_date'] == latest_date]

    # filtering for instance essential data
    df_capacity = df_capacity[df_capacity['resource_id'].isin(final_result['resources_needed'])]
    df_capacity = df_capacity[df_capacity['location_id']==final_result['final_plant']]
    df_capacity = df_capacity[df_capacity['product_id']==final_result['product_requested']]
    df_capacity = df_capacity[['location_id', 'resource_id', 'product_id', 'week_year', 'capacity_usage_of_production_resource']]

    n_weeks = int(final_result['total_lead_time_weeks'])
    filtered_clamped, info_clamped = filter_df2_by_df1_range(df_stock_status_alert, df_capacity, n_weeks=n_weeks, clamp=True)

    #print("Diagnostics (clamped):", info_clamped)
    # filtering based on only required resources
    filtered_clamped = filtered_clamped[filtered_clamped["resource_id"].isin(final_result['resources_needed'])]

    df_capcity_data = filtered_clamped.copy()
    return df_capcity_data

### 2. Gather Other DC data if present
def gather_other_dc_data(df_stock_status_alert,final_result):
    df_rev_dc = import_table_data(engine, "INVT_HISTORICAL_DATA", "REVIEW_DC_HISTORY")
    #print(df_rev_dc.columns)
    df_rev_dc['refresh_date'] = pd.to_datetime(df_rev_dc['refresh_date'], dayfirst=True, errors='coerce')
    latest_date = df_rev_dc['refresh_date'].max()
    df_rev_dc = df_rev_dc[df_rev_dc['refresh_date'] == latest_date]

    # filter for other dc
    final_destination_locations_list = final_result["final_destinations"]
    product_id_filter = final_result["product_requested"]
    item_to_remove  = final_result["location_requested"]
    # Create a new list without the given item
    filtered_final_destination_locations = [item for item in final_destination_locations_list if item != item_to_remove]

    # applying filter on product and location 
    df_rev_dc = df_rev_dc [df_rev_dc['product_id']==final_result["product_requested"]]
    df_rev_dc = df_rev_dc[df_rev_dc["location_id"].isin(filtered_final_destination_locations)]
    # only taking important columns for reasoning
    df_rev_dc = df_rev_dc[['product_id', 'location_id', 'week_year','incoming_transport_receipts','dependent_demand']]

    #print('-------- Alternate location ---------\n ',filtered_final_destination_locations)
    filtered_df_rev_dc = df_rev_dc[df_rev_dc["location_id"].isin(filtered_final_destination_locations)]
    filtered_df_rev_dc = filtered_df_rev_dc [filtered_df_rev_dc['product_id']==product_id_filter]

    n_weeks = int(final_result['total_lead_time_weeks'])
    df_other_dc, info_clamped = filter_df2_by_df1_range(df_stock_status_alert, filtered_df_rev_dc, n_weeks=0, clamp=True)
    return df_other_dc

### 3. Raw Material Data
def gather_raw_mat_data(df_stock_status_alert,final_result):
    # df_stock_status = import_table_data(engine, "INVT_XAI", "STOCK_STATUS")
    df_rv_comp = import_table_data(engine, "INVT_HISTORICAL_DATA", "REVIEW_COMPONENT_HISTORY")
    df_rv_comp['refresh_date'] = pd.to_datetime(df_rv_comp['refresh_date'], dayfirst=True, errors='coerce')
    latest_date = df_rv_comp['refresh_date'].max()
    df_rv_comp = df_rv_comp[df_rv_comp['refresh_date'] == latest_date]

    # filtering for instance essential data
    df_rv_comp = df_rv_comp[df_rv_comp['product_id'].isin(list(final_result['vendors_associated'].keys()))]
    df_rv_comp = df_rv_comp[df_rv_comp['location_id']==final_result['final_plant']]
    df_rv_comp = df_rv_comp[['product_id', 'location_id', 'week_year', 'dependent_demand','planned_transport_receipt']]


    # passing data
    n_weeks = int(final_result['total_lead_time_weeks'])
    filtered_clamped, info_clamped = filter_df2_by_df1_range(df_stock_status_alert, df_rv_comp, n_weeks=n_weeks, clamp=True)
    df_raw_mat = filtered_clamped.copy()
    return df_raw_mat

### 4. Gather Plant incoming data
def gather_plant_data(df_stock_status_alert,final_result):
    # df_stock_status = import_table_data(engine, "INVT_XAI", "STOCK_STATUS")
    df_rv_plant = import_table_data(engine, "INVT_HISTORICAL_DATA", "REVIEW_PLANT_HISTORY")
    df_rv_plant['refresh_date'] = pd.to_datetime(df_rv_plant['refresh_date'], dayfirst=True, errors='coerce')

    # Find the latest date
    latest_date = df_rv_plant['refresh_date'].max()
    # Find the date 1 week earlier
    historic_date = latest_date - pd.Timedelta(weeks=1)

    # Filter dataframe for that historic date
    df_rv_plant_historic = df_rv_plant[df_rv_plant['refresh_date'] == historic_date]
    # filtering for instance raw material
    df_rv_plant_historic = df_rv_plant_historic[df_rv_plant_historic['product_id']==final_result['product_requested']]
    df_rv_plant_historic = df_rv_plant_historic[df_rv_plant_historic['location_id']==final_result['final_plant']]
    df_rv_plant_historic = df_rv_plant_historic[['product_id', 'location_id', 'week_year', 'open_production_orders']]

    #Filter on latest data
    df_rv_plant_curr = df_rv_plant[df_rv_plant['refresh_date'] == latest_date]
    # filtering for instance raw material
    df_rv_plant_curr = df_rv_plant_curr[df_rv_plant_curr['product_id']==final_result['product_requested']]
    df_rv_plant_curr = df_rv_plant_curr[df_rv_plant_curr['location_id']==final_result['final_plant']]
    df_rv_plant_curr = df_rv_plant_curr[['product_id', 'location_id', 'week_year', 'open_production_orders']]

    # Cuuting the data according to the instance
    n_weeks = int(final_result['total_lead_time_weeks'])
    filtered_clamped, info_clamped = filter_df2_by_df1_range(df_stock_status_alert, df_rv_plant_historic, n_weeks=n_weeks, clamp=True)
    df_rv_plant_historic = filtered_clamped.copy()

    filtered_clamped, info_clamped = filter_df2_by_df1_range(df_stock_status_alert, df_rv_plant_curr, n_weeks=n_weeks, clamp=True)
    df_rv_plant_curr = filtered_clamped.copy()

    # Perform left join
    df_rv_plant_curr_plus_lag1 = pd.merge(
        df_rv_plant_curr,
        df_rv_plant_historic,
        on=["product_id", "location_id", "week_year"],
        how="left",
        suffixes=("", "_lag1")  # suffix for historic cols
    )
    return df_rv_plant_curr_plus_lag1

### 5. Direct Vendor supplied product data
def gather_direct_vendor_data(df_stock_status_alert,final_result):
    # df_stock_status = import_table_data(engine, "INVT_XAI", "STOCK_STATUS")
    df_direct_vendor = import_table_data(engine, "INVT_HISTORICAL_DATA", "REVIEW_VENDORS_HISTORY")
    # Filtering on latest date
    df_direct_vendor['refresh_date'] = pd.to_datetime(df_direct_vendor['refresh_date'], dayfirst=True, errors='coerce')
    latest_date = df_direct_vendor['refresh_date'].max()
    df_direct_vendor = df_direct_vendor[df_direct_vendor['refresh_date'] == latest_date]
    #df_direct_vendor.rename(columns={'WEEK_YEAR': 'week_year'}, inplace=True)
    #print(f"--------shape of vendor - {df_direct_vendor.shape}\{df_direct_vendor.columns}n----------")
    if final_result['product_requested'] in df_direct_vendor['product_id'].values:
        # filtering for instance essential data
        df_direct_vendor = df_direct_vendor[df_direct_vendor['product_id']==final_result['product_requested']]
        df_direct_vendor = df_direct_vendor[['location_id', 'product_id','week_year', 'maximum_external_receipt']]

        n_weeks = int(final_result['total_lead_time_weeks'])
        filtered_clamped, info_clamped = filter_df2_by_df1_range(df_stock_status_alert, df_direct_vendor, n_weeks=n_weeks, clamp=True)
        df_direct_vendor_data = filtered_clamped.copy()
    else:
        df_direct_vendor_data = pd.DataFrame()
    df_direct_vendor_data.head()
    return df_direct_vendor_data

################## Guided understock and Overstock pipeline ###############
#################### overstock Module ###################

# Creating overstock pipeline
def df_to_chat_response_overstock(df_stock_status_alert ,df_capcity_data, final_result, llm):
    # Ensure 'week_end_date' is in datetime format
    df_stock_status_alert["week_end_date"] = pd.to_datetime(df_stock_status_alert["week_end_date"])
    latest_date = df_stock_status_alert["week_end_date"].max()
    earliest_date = df_stock_status_alert["week_end_date"].min()

    # Filter rows for the latest available week
    filtered_alerts = df_stock_status_alert[df_stock_status_alert["week_end_date"] == latest_date]

    # Convert dataframes to string for prompt context
    # stock_data_str = filtered_alerts.to_string(index=False) if not filtered_alerts.empty else "No stock data available."
    stock_data_str = df_stock_status_alert.to_string(index=False) if not df_stock_status_alert.empty else "No stock data available."
    df_capcity_data_str = df_capcity_data.to_string(index=False) if not df_capcity_data.empty else "No capcity data available."

    prompt = f"""
You are an expert in Inventory Planning.

Your task is to provide L1 (level one) and L2 (level two) reason for inventory imbalance specifically overstock instances at the (product_id, location_id) level using the data provided.
L1 is the primary reason by which overstock has occured. While L2 reason provides the cause behind the L1 reason.
- Start Date: {earliest_date.date()}
- End Date: {latest_date.date()}

Important instructions-
- Include Start Date and End Date in the Alert generated.
- Strictly avoid assumptions. All insights must be grounded in the provided data.
- Thus, Data evidence and should explained step wise in the output field - 'Chain of thought'
- In 'Chain of thought' section, verify that every statement is directly supported by data followed by your conclusion on that data and logic.
- There can be single or multiple L1 and L2 reasons. Thus, It is important to mention all the applicable L1 and L2 reasons in your 'Chain of thought' which are proved to be the reason. 
- Thus during reasoning,if multiple checks are passed while following the provided steps, then make sure to specify all the corresponding L1 and L2 reasons as the cause for the instance

Explanation Guidelines:
- The alert must be based on one unique (product_id, location_id) instance.
- Supply Chain Flow: VEN ➝ Plant ➝ RDC ➝ DC
- 'Product Supply Chain details' for the instance product id and location id that is to be investigated as requested is provided below. Other details like source of product i.e. Plant, Vendor and raw materials (with their Vendor) needed is also provided. After that complete tracing of the product with their lead times in week is provided. Finally all the DC location where the origins Plants product gets divided for distribution is provided. The details are as follows:
\n {final_result}\n
- You have to use 5 tables of data that is provided to you - df_stock_status_alert, df_capcity_data, df_other_dc, df_raw_mat, df_rv_plant
- As df_stock_status_alert table contains the exact instance data and date, all the other tables has data which has its date shifter according to transportation lead time and production time as shown in 'Product Supply Chain details'. Thus, Take the leading time (in weeks) into account during your reasoning.
- If any of the tables data is not provided or blank that means that the table data is not required to use for reasoning
- Brief table details: 
1. df_stock_status_alert: It is the primary table to which contains majority of the significant data details for the overstock instance including week and dates
2. df_capacity_data: this data provides how much hours of capacity utilization was done for the product and the plant location by the resource
3. df_other_dc: The Product's origin Plant may distributes its supply between multiple DCs(Distribution centers). In such case , the data of supply done to DCs other than the requested DC is mentioned in this table.
4. df_raw_mat: This data shows raw material supply to a plant.
5. df_rv_plant: This data tells the production orders placed by the planner currently and historically one week before

df_stock_status_alert being the primary instance detection table, its details are:
- Column Name (Data Type): column description
- product_id (varchar): Unique identifier of the product (e.g., FG-1000, FG-2000).
- location_id (varchar): Identifier of the distribution center (e.g., DC1000, RDC1000).
- projected_stock (integer): Expected stock level after netting incoming and outgoing flows.
- safety_stock (integer): Planned Stock to be maintained for that week.
- stock_on_hand (integer): Inventory currently available at the location.
- incoming_recipts (integer): Confirmed quantity of supplies received by the current location.
- total_demand (integer): Dependent Demand received from customer or connected distribution center.
- outgoing_supply (integer): Actual outgoing quantity of supplies from the current location.
- supply_order (integer): Planned orders i.e. quantity of supply to be recieved that has be scheduled by the supply chain manager. 
- week_num (integer): An integer (1-52) representing the ISO week number of the year.
- week_end_date (date): Date (ISO format) representing the last day (typically Sunday) of the given week_num and year.
- quarter (varchar): Calendar quarter (e.g., Q1, Q2) corresponding to the week_end_date.
- month (varchar): Full month name (e.g., January, February) derived from the week_end_date.
- year (integer): Four-digit calendar year associated with the week_num.
- stock_condition (varchar): Indicates the comparison between projected_stock and safety_stock.
Values:
excess: when projected_stock > safety_stock
deficit: when projected_stock < safety_stock
in-stock: when projected_stock == safety_stock
- stock_status_warning (varchar): Alerts when 4 or more consecutive weeks of excess or deficit are detected for a given product_id, location_id, and location_type.
Values:
normal: No warning condition detected
overstock_instance_N: Nth occurrence of a prolonged excess stock condition
understock_instance_N: Nth occurrence of a prolonged deficit stock condition
In response, keep in mind to mention Nth occurrence of condition only once 
While responding with instance occurrence, don't mention instances verbatim (like overstock_instance_1) , mention it in more natural language format.
- transportation_lead_time (integer): time (in weeks) required for traversal of a product from RDC to the DC
- minimum_transportation (integer): minimum number of units that can be transported to the current location
- incremental_transportation (integer): additional minimum incremental units that can be added to minimum transportation unit to transport more units to the current location
- production_lead_time (integer): time (in weeks) required for the production of the good
- offset_stock (integer): difference between projected stock and safety stock that signifies amount of stock at the end of the week
- lag_1_dependent_demand (integer): These are depedent demand value predicted a week prior to current prediction data. (while responding use column name alias as 'Prior predicted depedent demand')
- lag_1_supply_orders (integer): These are supply order value predicted a week prior to current prediction data. (while responding use column name alias as 'Prior predicted supply order')

Only respond with a JSON block of Alert explanation which follows SAMPLE format template as:
{{
"Alert": "Excess observed for product_id FG-100-001 and location_id DC1000 during 2026-07-05 to 2026-07-05",
"Chain of thought": "Step 1. The overstock instance starts from  2026-07-05 to 2026-07-05. Step 2.The stock on hand value for the first week 0.That means value for first week 'stock on hand' for the instance doesn't exist. Thus,L1 reason is not related to stock hand. Step 3. As the previous step did not succeed this step is not applicable. Proceed to next step.........<completeing all 13 steps>.I have gone through and concluded on all 13 steps as instructed. Thus,there 2 applicable reasons for this instance."
"Reason 1": {{
"L1 Reason": "Larger Lot Size",
"L2 Reason": "At the beginning of the overstock period , the stock on hand was 800 units, which is significantly higher than the safety stock of 100 units.The production lead time is 6 weeks, which is longer than the duration of the overstock instance (9 weeks). This longer production lead time in the production plant resulted in maintaining high stock on hand to support lead time horizon demand.",
"Priority": "High"
}},
"Reason 1": {{
"L1 Reason": "Overforecasting Demand",
"L2 Reason": " The current demand and the prior demand values are .....<corresponding reasoning showcase as instructed>.",
"Priority": "Medium"
}}
}}

Prcoess to find L1 and L2 Reasons fo overstock, going step by step:
        Step 1. First look at that individual instance of overstock and confirm the checks given in subsequent steps. Based on the confirmed checks, provide all the passed check-wise L1 and L2 reasons as instructed. Remember that some checks are not applicable when requested location is a plant location. Such cases will be mentioned at that step of checks.
        Step 2. Find the 'stock on hand' value for the first week of the instance period i.e. the first week of the instance and check If 'stock on hand' value greater than 0
        Step 3. If check succeeds then check if the value is greater than 'safety stock' value for the same week (i.e. first week of the instance)
        Step 4. If the check is true then respond that the cause of overstock is because of high amount of 'stock on hand' at the start of the period as L1 (Level 1) reason
        Step 5. For (Level 2) L2 reason check 'transportation lead time' first and later 'production lead time' 
        If the requested location is plant location, then from below substeps, you must only consider 'production lead time' for reason generation.
        sub step 1 - if 'transportation lead time' is less than 4 weeks short than the duration (number of weeks i.e. records) in the overstock instance then say L2 reason as Longer Transportation Lead Time travel to the location resulted in High Stock on Hand to support Lead Time Horizon Demand.
        sub step 2 - If 'transportation lead time' is more than 4 weeks short than the instance duration then check if 'production lead time' is less than 3 weeks short than the duration (number of weeks i.e. records). If 'production lead time' qualifies the check then say the L2 reason as Longer Production Lead Time in the Production Plant, High Stock on Hand to support Lead Time Horizon Demand.
        sub step 3 - If both of the sub steps checks fails, then see if the addition of 'transportation lead time' and 'production lead time' is more than 3 weeks short than the instance duration. If check passes then say L2 reason as Longer Production Lead time and Transportation Lead Time travel to the location combined resulted in High Stock on Hand to support Lead Time Horizon Demand.
        Step 6. Moving to next reason verification, check if for first week of the instance period, is 'incoming receipts' has value greater than 'total demand'
        Step 7. If check is true, then say that L1 reason as Larger Lot Size is the reason for overstocking.this is because of strict lower limit on lot size on supply units being more the supply recieved were greater than demand causing excess.
        sub step 1 - For L2 reason, check if any present values of 'incoming receipts' has value as added multiples of 'incremental transportation' value to 'minimum transportation'(e.g. 'minimum transportation' 600 + n* 'incremental transportation' 100 -> 700,800,900,etc )  
        sub step 2 - If check is true, then say that L2 reason for overstocking is as Demand is lesser than Rounding Transportation Lot Size.
        sub step 3 - If check fails, then check if all present values of 'incoming receipts' for the instance duration is exactly same as 'minimum transportation' (i.e. Minimum Transportation Lot Size)
        sub step 4 - if above check is true then say that L2 reason for overstocking is as Demand is lesser than Minimum Transportation Lot Size.
        Step 8. Moving to next reason verification, check if 'total demand' is lower than 'lag_1_dependent_demand' for the instance by comparing SUM of their values for the duration with each other i.e. compare sum of all values for 'total demand' vs sum of all values for 'lag_1_dependent_demand' for the instance.
        Step 9. If the calulated values shows that 'lag_1_dependent_demand' sum is higher than 'total demand' sum, that means that the decrease in demand in the current scene as compared to prediction of prior week demand is the reason for overstocking. Thus, making L1 reason as Over forecasting Demand and L2 reason as Demand Drop in the Lead Time Horizon compared to last week.
        Step 10. Moving to next reason verification,check if the 'location id' in stock status alert data is of plant. (Plant have location id starting with "PL". e.g.' PL1000')
        Step 11. If the check is true,observe the the capcity data which provided to you (which is already been adjusted according to production lead time) for the complete instance. Detect if there are some weeks where capacity usage of production resource has been zero. Thus,This reason verification must be done only if the provided df_capcity_data has more than 4 weeks / records of data provided.
        Step 12. If the above check is true then also check in stock status data if there has been duration of consistent weeks where 'outgoing supply' is higher than 'total depand'. if yes, that means the plant has been prebuilding and stocking up for the upcoming reduction of the resourse capcity. In such case L1 reason is 'Production Prebuild' and L2 reason is 'Resource Capacity(Machine/Labor) is not available/enough in the weeks where demand is present, but available in the early weeks so plant is pre-building'
        Step 13. If and only if none of the above L1 reasons come up as reasons in all the steps provided, then say the L1 reason as the alert to be investigated by autonomous module. In this special case, skip the L2 reason.

Instructions for L2 reason response :
- First Mention the L2 reason and then you must respond on how L2 reason is in-depth reason that ties to the assoicated L1 reason. 
- Use atleast one point of data as evidence  while explaining in the L2 reason as to keep the L2 reason simple
- L2 reason must be easy to understand and in concise natural language.
- Refer to the DataFrame as “the table” or “the dataset”—do not use “df”.
- Always use the user-friendly aliases for column names in your descriptions.
** For example: ** Instead of “The capacity_usage_of_production_resource for more than 4 consecutive weeks in df is zero,” say:
  “Production resource utilization remained zero for more than four consecutive weeks in the dataset.”

Data:
- df_stock_status_alert :\n{stock_data_str}\n
- df_capcity_data :\n{df_capcity_data_str}\n

Only respond with a JSON block of alert explanation.
"""

# **Return only one alert per unique L1 reason. Do not repeat the same L1 reason across different weeks. Return a minimum of 1 and a maximum of 3 such alerts.**


    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Provide concise root cause analysis for each (product_id, location_id) alert using JSON format."}
    ]

    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"Error while getting response from LLM: {e}")
        return "LLM failed to generate explanation."


def execute_pipeline_overstock(df_stock_status_alert ,df_capcity_data, final_result, llm):
    if df_stock_status_alert.empty:
        return {"error": "No stock alerts found."}

    try:
        explanation = df_to_chat_response_overstock(df_stock_status_alert ,df_capcity_data, final_result, llm)
    except Exception as e:
        return {"error": f"LLM Explanation Error: {str(e)}"}

    return {
        "data": {
            "filtered_alerts": df_stock_status_alert.to_dict(orient="records"),
            "explanation": explanation
        }
    }

#################### Understock Module ###################

# creating a seperate pipline for understock
def df_to_chat_response_understock(df_stock_status_alert ,df_capcity_data, df_other_dc, df_raw_mat, df_rv_plant,df_direct_vendor, final_result, llm ):
    # Ensure 'week_end_date' is in datetime format
    df_stock_status_alert["week_end_date"] = pd.to_datetime(df_stock_status_alert["week_end_date"])
    latest_date = df_stock_status_alert["week_end_date"].max()
    earliest_date = df_stock_status_alert["week_end_date"].min()

    # Filter rows for the latest available week
    filtered_alerts = df_stock_status_alert[df_stock_status_alert["week_end_date"] == latest_date]

    # Convert dataframes to string for prompt context
    # stock_data_str = filtered_alerts.to_string(index=False) if not filtered_alerts.empty else "No stock data available."
    stock_data_str = df_stock_status_alert.to_string(index=False) if not df_stock_status_alert.empty else "No stock data available."
    df_capcity_data_str = df_capcity_data.to_string(index=False) if not df_capcity_data.empty else "No capcity data available."
    df_other_dc_str = df_other_dc.to_string(index=False) if not df_other_dc.empty else "No other dc data available."
    df_rv_plant_str = df_rv_plant.to_string(index=False) if not df_rv_plant.empty else "No plant data available."
    df_raw_mat_str = df_raw_mat.to_string(index=False) if not df_raw_mat.empty else "No raw material data available."
    df_direct_vendor_str = df_direct_vendor.to_string(index=False) if not df_direct_vendor.empty else "No raw material data available."


    prompt = f"""
You are an expert in Inventory Planning.
Your task is to provide L1 (level one) and L2 (level two) reason for inventory imbalance specifically understock instances at the (product_id, location_id) level using the data provided.
L1 is the primary reason by which understock has occurred. While L2 reason provides the cause behind the L1 reason.
- Start Date: {earliest_date.date()}
- End Date: {latest_date.date()}

Important instructions-
- Include Start Date and End Date in the Alert generated.
- Strictly avoid assumptions. All insights must be grounded in the provided data.
- Thus, Data evidence and should explained step wise in the output field - 'Chain of thought'
- In 'Chain of thought' section, verify that every statement is directly supported by data followed by your conclusion on that data and logic.
- There can be single or multiple L1 and L2 reasons. Thus, It is important to mention all the applicable L1 and L2 reasons in you 'Chain of thought' which are proved to be the reason. 
- Thus during reasoning,if multiple checks are passed while following the provided steps, then make sure to specify all the corresponding L1 and L2 reasons as the cause for the instance


Explanation Guidelines:
- The alert must be based on one unique (product_id, location_id) instance.
- Supply Chain Flow: VEN ➝ Plant ➝ RDC ➝ DC
- 'Product Supply Chain details' for the instance product id and location id that is to be investigated as requested is provided below. Other details like source of product i.e. Plant, Vendor and raw materials (with their Vendor) needed is also provided. After that complete tracing of the product with their lead times in week is provided. Finally all the DC location where the origins Plants product gets divided for distribution is provided. The details are as follows:
\n {final_result}\n
- You have to use 5 tables of data that is provided to you - df_stock_status_alert, df_capcity_data, df_other_dc, df_raw_mat, df_rv_plant
- As df_stock_status_alert table contains the exact instance data and date, all the other tables has data which has its date shifter according to transportation lead time and production time as shown in 'Product Supply Chain details'. Thus, Take the leading time (in weeks) into account during your reasoning.
- If any of the tables data is not provided or blank that means that the table data is not required to use for reasoning
- Brief table details: 
1. df_stock_status_alert: It is the primary table to which contains majority of the significant data details for the understock instance including week and dates
2. df_capacity_data: this data provides how much hours of capacity utilization was done for the product and the plant location by the resource
3. df_other_dc: The Product's origin Plant may distributes its supply between multiple DCs(Distribution centers). In such case , the data of supply done to DCs other than the requested DC is mentioned in this table.
4. df_raw_mat: This data shows raw material supply to a plant.
5. df_rv_plant: This data tells the production orders placed by the planner currently and historically one week before
6. df_direct_vendor_data: This data shows if any of the product is directly supplied any vendor(it's vendor ID represented by location_id ) rather than produced in plant. 

df_stock_status_alert being the primary instance detection table, its details are:
- Column Name (Data Type): column description
- product_id (varchar): Unique identifier of the product (e.g., FG-1000, FG-2000).
- location_id (varchar): Identifier of the distribution center (e.g., DC1000, RDC1000).
- projected_stock (integer): Expected stock level after netting incoming and outgoing flows.
- safety_stock (integer): Planned Stock to be maintained for that week.
- stock_on_hand (integer): Inventory currently available at the location.
- incoming_recipts (integer): Confirmed quantity of supplies received by the current location.
- total_demand (integer): Dependent Demand received from customer or connected distribution center.
- outgoing_supply (integer): Actual outgoing quantity of supplies from the current location.
- supply_order (integer): Planned orders i.e. quantity of supply to be recieved that has be scheduled by the supply chain manager. 
- week_num (integer): An integer (1-52) representing the ISO week number of the year.
- week_end_date (date): Date (ISO format) representing the last day (typically Sunday) of the given week_num and year.
- quarter (varchar): Calendar quarter (e.g., Q1, Q2) corresponding to the week_end_date.
- month (varchar): Full month name (e.g., January, February) derived from the week_end_date.
- year (integer): Four-digit calendar year associated with the week_num.
- stock_condition (varchar): Indicates the comparison between projected_stock and safety_stock.
Values:
excess: when projected_stock > safety_stock
deficit: when projected_stock < safety_stock
in-stock: when projected_stock == safety_stock
- stock_status_warning (varchar): Alerts when 4 or more consecutive weeks of excess or deficit are detected for a given product_id, location_id, and location_type.
Values:
normal: No warning condition detected
overstock_instance_N: Nth occurrence of a prolonged excess stock condition
understock_instance_N: Nth occurrence of a prolonged deficit stock condition
In response, keep in mind to mention Nth occurrence of condition only once 
While responding with instance occurrence, don't mention instances verbatim (like understock_instance_1) , mention it in more natural language format.
- transportation_lead_time (integer): time (in weeks) required for traversal of a product from RDC to the DC
- minimum_transportation (integer): minimum number of units that can be transported to the current location
- incremental_transportation (integer): additional minimum incremental units that can be added to minimum transportation unit to transport more units to the current location
- production_lead_time (integer): time (in weeks) required for the production of the good
- offset_stock (integer): difference between projected stock and safety stock that signifies amount of stock at the end of the week
- lag_1_dependent_demand (integer): These are depedent demand value predicted a week prior to current prediction data. (while responding use column name alias as 'Prior predicted depedent demand')
- lag_1_supply_orders (integer): These are supply order value predicted a week prior to current prediction data. (while responding use column name alias as 'Prior predicted supply order')

Only respond with a JSON block of Alert explanation which follows SAMPLE format template as:
{{
"Alert": "Deficit observed for product_id FG-100-001 and location_id DC1000 during 2025-08-24 to 2025-10-19",
"Chain of thought": "Step 1. The understock instance starts from  2025-08-24 to 2025-10-19. Step 2.All the incoming receipts for the instance duration is zero.The check fails. Thus one of the L1 reason is longer lead time. Proceed to next step.........<completeing all 21 steps>.I have gone through and concluded on all 21 steps as instructed. Thus, there 2 applicable reasons for this instance. "
"Reason 1": {{
"L1 Reason": "Longer Lead Time",
"L2 Reason": "Throughout the period from week 34 to 42, there were no incoming receipts for the majority of the weeks, except for weeks 40 and 41 where incoming receipts were recorded as 500 units. This indicates a prolonged period without sufficient supply. Furthermore, The transportation lead time is consistently 8 weeks, which is significant and suggests delays in the movement of goods from the regional distribution center to the distribution center. This extended transportation lead time is a contributing factor to the understock condition",
"Priority": "High"
}},
"Reason 2": {{
"L1 Reason": "Supplier Delays",
"L2 Reason": "Throughout the period from week 34 to 42 ....... <corresponding reasoning showcase as instructed>.",
"Priority": "Medium"
}}
}}

Process to find all L1 and L2 Reasons applicable for understock, going step by step:
        Step 1. First look at that individual instance of understock and confirm the checks given in subsequent steps. Based on the confirmed checks, provide all the passed check-wise L1 and L2 reasons as instructed. Remember that some checks are not applicable when requested location is a plant location. Such cases will be mentioned at that step of checks.
        Step 2. Check if there exists any values greater than 0 in 'incoming receipts' for that instance period.
        Step 3. If the check is False then say L1 reason is Longer Lead time ,as there are no supply being provided to the location due to high lead time.
        If the requested location is plant location, then from below substeps, you must only consider 'production lead time' for reason generation.
        sub step 1. For L2 reasoning, stricly check if value for 'transportation lead time' greater than 3 (weeks). if check is true then say L2 reason is Longer Transportation Lead, as the delay is supply is caused by high transportaion lead time to the requested location.(This L2 check is not applicable if the requested location is a plant location)
        sub step 2. Now if sub step 1 check fails, stricly check if value for 'production lead time' is greater than 3 (weeks). If check is true then say L2 reason Longer Production Lead Time in the Production Plant, as the delay is supply is caused by high production lead time.
        Step 4. Moving to next reason verification, check if sum of 'total demand' is higher than sum of 'lag_1_dependent_demand' for the instance by comparing SUM of their values for the duration with each other i.e. compare sum of all values for 'total demand' vs sum of all values for 'lag_1_dependent_demand'.
        Step 5. If difference the sums shows that  'lag_1_dependent_demand' was lower than 'total demand' sum, that means that the increase in demand in the current scene as compared to prediction of prior week demand is the reason for understock. Thus, making L1 reason as Underforecasting Demand and L2 reason as Demand Spike in the Lead Time Horizon compared to Last week.
        Step 6. Moving to next reason verification, check if the requested product is in product_ids in df_direct_vendor_data. If yes, that means this product is directly supplied by the vendor id provided in 'location_id' column of the df_direct_vendor_data. if df_direct_vendor_data is empty then move on to next L1 reason evaluation.
        Step7. Now, check if values exists for maximum_external_receipt for that product for the instance duration
        Step 7. If all values are '0' for the instance duration then it means that vendor id (provided in 'location_id' of df_direct_vendor_data) has stopped supply for the product for that duration and making L1 reason as Supplier Delays and L2 reason as Supplier Capacity Constraints
        Step 8. Moving to next reason verification, Check is if there are non zero 'lag_1_supply_order' values for the instance duration.
        Step 9. If the previous check is true, then check if 'supply order' values are different than 'lag_1_supply_order' values across time as such the 'lag_1_supply_order' has been pushed ahead compared to current orders i.e. 'supply order' resulting in no supply order. If check is true that means the planner has delayed the order. This means that L1 reason is Transportation delays and L2 reason as Delayed Purchase Orders.
        Step 10. Moving to next reason verification, Check if the capacity_usage_of_production_resource for more than 4 CONSECUTIVE weeks in the given duration is zero or blank in df_capcity_data. df_capcity_data has already week time adjusted according tp the lead time.Thus,This reason verification must be done only if the provided df_capcity_data has more than 4 weeks / records of data provided.
        Step 11. If the said check is true that means L1 reason is Production delays. Now check if the df_other_dc has non zero values in 'incoming recipts' for the same duration.
        Step 12. If check is true, then L2 reason is Resource Capacity(Machine/Labor) is not enough to meet the Production Requirement and the availbale stock was supplied to higher priority other DC location. If check Fails then L2 reason is Resource Capacity(Machine/Labor) is not enough to meet the Production Requirement for all DC locations.
        Step 13. Moving to next reason verification, First look at the undertsock instance , note the 'product id' from 'stock status' and consider raw materials that are required to prepare the product that would be provided to you in 'Raw material needed' section
        Step 14. Now from the df_raw_mat table , look for the raw material by fitering throgh 'product id' column. Thus,This reason verification must be done only if the provided df_raw_mat has more than 4 weeks / records of data provided for each raw material needed.
        Step 15. Now check if any of the raw material has lower 'planned transport receipts' than than the 'dependent demand' for more than 4 consecutive weeks.
        Step 16. If the above check is true, that means that the vendor was not able to supply the raw material as per the demand. Use the provided data in 'Raw material supplied by Vendor' section to associate vendor with the raw material. Thus, this makes L1 reason Production delays and L2 reason as Raw material shortages due to insufficient supply from vendor  
        Step 17. Moving to next reason verification, First look at the undertsock instance and its duration. Now look at the 'current plant supply data' table that will be provided with its duration already taken into account according to Plant to DC transportation lead time.
        Step 18. Check if there are any 'open production orders' (current orders) in the plant supply data that is not blank or zero. 
        Step 19. If above checks is true, Note the week duration of those orders and the values.
        Step 20. Now check the 'lag1_open_production_orders' which is historical plan for the same week duration provided in. If the lag1_open_production_orders has been pushed ahead compared to current orders i.e. open_production_orders, that means the planner has delayed the order. This means L1 reason as Production delays and L2 reason as Delayed production runs by planner.
        Step 21. If and only if none of the above L1 reasons come up as reasons in all the steps provided, then say the L1 reason as the alert to be investigated by autonomous module. In this special case, skip the L2 reason.

Instructions for L2 reason response :
- First Mention the L2 reason and then you must respond on how L2 reason is in-depth reason that ties to the assoicated L1 reason. 
- Use atleast one point of data as evidence  while explaining in the L2 reason as to keep the L2 reason simple
- L2 reason must be easy to understand and in concise natural language.
- Refer to the DataFrame as “the table” or “the dataset”—do not use “df”.
- Always use the user-friendly aliases for column names in your descriptions.
** For example: ** Instead of “The capacity_usage_of_production_resource for more than 4 consecutive weeks in df is zero,” say:
  “Production resource utilization remained zero for more than four consecutive weeks in the dataset.”

Data:
- Raw material needed: \n{list(final_result['vendors_associated'].keys())}\n
- Raw material supplied by Vendor : \n{final_result['vendors_associated']}\n
- current/historic plant supply data:\n{df_rv_plant}\n
- df_stock_status_alert :\n{stock_data_str}\n
- df_capcity_data :\n{df_capcity_data_str}\n
- df_other_dc :\n{df_other_dc_str}\n
- df_raw_mat :\n{df_raw_mat_str}\n
- df_direct_vendor_data:\n{df_direct_vendor_str}\n

Only respond with a JSON block of alert explanation.
"""

# **Return only one alert per unique L1 reason. Do not repeat the same L1 reason across different weeks. Return a minimum of 1 and a maximum of 3 such alerts.**


    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Provide concise root cause analysis for each (product_id, location_id) alert using JSON format."}
    ]

    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"Error while getting response from LLM: {e}")
        return "LLM failed to generate explanation."


def execute_pipeline_understock(df_stock_status_alert ,df_capcity_data, df_other_dc, df_raw_mat, df_rv_plant,df_direct_vendor,final_result, llm):
    if df_stock_status_alert.empty:
        return {"error": "No stock alerts found."}

    try:
        explanation = df_to_chat_response_understock(df_stock_status_alert ,df_capcity_data, df_other_dc, df_raw_mat, df_rv_plant,df_direct_vendor, final_result, llm)
    except Exception as e:
        return {"error": f"LLM Explanation Error: {str(e)}"}

    return {
        "data": {
            "filtered_alerts": df_stock_status_alert.to_dict(orient="records"),
            "explanation": explanation
        }
    }

# Configure pandas display
# pd.set_option("display.max_columns", None)

# Step 0.A --- Initializing LLM model
# from gen_ai_hub.proxy.langchain.init_models import init_llm
# llm = init_llm("gpt-4o", max_tokens=4096)


# Step 0.B --- Configuring database instance

schema_name2 = "INVT_HISTORICAL_DATA"

inspector = inspect(engine)
tables = inspector.get_table_names(schema=schema_name2)
db = SQLDatabase(engine=engine, schema=schema_name2)


# Step 1 --- Load supporting context files
def _load_context_files():
    with open("Data_description.txt", "r") as f1:
        data_description = f1.read()
    with open("stock_status_scenarios.txt", "r") as f2:
        stock_status_scenarios = f2.read()
    with open("supply_chain_context.txt", "r") as f3:
        supply_chain_context = f3.read()
    with open("product_supply_chain_flow.txt", "r") as f4:
        product_supply_chain_flow = f4.read()
    return data_description, stock_status_scenarios, supply_chain_context, product_supply_chain_flow


data_description, stock_status_scenarios, supply_chain_context, product_supply_chain_flow = (
    _load_context_files()
)


# Step 2 --- Import stock status table data
# def import_table_data(engine, schema_name, table_name):
#     """Fetch full table from database"""
#     query = f'SELECT * FROM "{schema_name}"."{table_name}"'
#     try:
#         conn = engine.connect()
#         df = pd.read_sql_query(text(query), con = conn)
#     except Exception as e:
#         print(f"Error fetching {table_name}: {e}")
#         return pd.DataFrame()
#     return df


# Step 3 --- Location source table
def get_lc_table(product_id, location_id):
    """Fetch location source details for a given product/location"""
    query = f'''
        SELECT location_id, product_id, ship_from_loc_id,
               incremental_transportation_lot_s,
               minimum_transportation_lot_size,
               transportation_lead_time
        FROM "{schema_name}".Location_Source
        WHERE Product_ID = :product_id
    '''
    params = {"product_id": product_id, "location_id": location_id}
    try:
        with engine.connect() as conn:
            result_table = pd.read_sql_query(text(query), con=conn, params=params)
    except Exception as e:
        print(f"Error querying location source: {e}")
        result_table = pd.DataFrame()
    return result_table


# Step 4 --- Executing transactional query
def execute_query(product_id, location_id, start_date, end_date, query):
    """Execute SQL query and split into snapshots by refresh_date"""
    params = {
        "product_id": product_id,
        "location_id": location_id,
        "start_date": start_date,
        "end_date": end_date,
    }
    try:
        with engine.connect() as conn:
            df_query = pd.read_sql_query(text(query), con=conn, params=params)
            print(df_query.head(1))
    except Exception as e:
        print(f"Error while fetching transaction data: {e}")
        return pd.DataFrame()

    if df_query.empty:
        return pd.DataFrame()

    # Rename and preprocess
    df_query = df_query.rename(columns={"week_end_date": "date"})
    df_query["date"] = pd.to_datetime(df_query["date"], errors="coerce")
    df_query["week_num"] = df_query["date"].dt.isocalendar().week
    df_query["year"] = df_query["date"].dt.isocalendar().year

    # Create snapshots (drop refresh_date column)
    snapshots = {
        ("current_snapshot" if i == 0 else f"{i}_week_earlier_snapshot"): (
            df_query[df_query["refresh_date"] == date_]
            .drop(columns=["refresh_date"])
            .copy()
        )
        for i, date_ in enumerate(sorted(df_query["refresh_date"].unique(), reverse=True))
    }
    return snapshots


# Step 5 --- LLM-generated SQL query
def generate_sql_query(product_id, location_id, start_date, end_date, table_information):
    """Ask LLM to generate SQL query and execute it"""
    generate_context = f"""
        You are an expert SQL query generator. Follow the INTERNAL PROCEDURE below exactly (these are the reasoning steps you must execute internally). 
        DO NOT output the internal reasoning or any explanations. 
        **Output should be only a sql query**.

        Schema:
        {table_information}

        INTERNAL PROCEDURE (do not output):
        1) From the provided Schema, find the table whose name contains the substring "review" (case-insensitive) and column name similar to ougoing supply.
        2) Among those, pick the single table that best matches the inputs by containing:
            - a product-like column (name contains any of: product, prod, sku, part, item, material)
            - a location-like column (name contains any of: location, loc, warehouse, dc, rdc, plant, site, store)
            - a date-like column (name contains any of: date, dt, txn, timestamp, time, receipt)
        Prefer tables that contain all three. If multiple tables still match, pick the one with most exact keyword hits.
        3) Identify the date column and its datatype if available in schema. Classify the date column datatype into:
            - DATE or TIMESTAMP -> treat as native dates
            - VARCHAR/CHAR/TEXT -> treat as STRING
            - If not provided -> UNKNOWN (assume STRING with default format)
        4) If date column is STRING, attempt to detect format from schema or example values (recognize common formats: DD-MM-YYYY, YYYY-MM-DD, DD/MM/YYYY, YYYY/MM/DD, DDMMYYYY). If not recognized, default to DD-MM-YYYY.
        5) Build the SQL:
        - Use placeholders exactly: :product_id, :location_id, :start_date, :end_date
        - If date column is native DATE/TIMESTAMP:
            WHERE <date_column> >= :start_date
                AND <date_column> <= :end_date
            Set "param_mode": "DATE_OBJECT"
        - If date column is STRING:
            WHERE TO_DATE(<date_column>, '<col_format>') >= TO_DATE(:start_date, 'YYYY-MM-DD')
                AND TO_DATE(<date_column>, '<col_format>') <= TO_DATE(:end_date, 'YYYY-MM-DD')
            Set "param_mode": "STRING_YYYY-MM-DD"
        - Use table and column names exactly as in the schema (preserve case).
        - ORDER BY the date column (if string, ORDER BY TO_DATE(...)).

        Inputs:
        - product_id = {product_id}
        - location_id = {location_id}
        - start_date = {start_date}
        - end_date = {end_date}

        Output Sql query:
                SELECT *
                    FROM table_name
                    WHERE date_column >= :start_date
                    AND date_column <= :end_date
                    AND product_id_column = :product_id
                    AND location_id_column = :location_id
                ORDER BY date_column

        """
    try:
        llm_output = llm.invoke(generate_context).content
    except Exception as e:
        print(f"Error invoking LLM for SQL generation: {e}")
        return pd.DataFrame()

    # Clean fenced SQL
    llm_output = llm_output.replace("```sql", "").replace("```", "").strip()
    return execute_query(product_id, location_id, start_date, end_date, llm_output)


# Step 6 --- Trace transaction chain
# def llm_transaction_chain(df_llm, product_id, location_id, start_date, end_date, table_information):
#     """Trace back through supply chain to collect transactional snapshots"""
#     current_location = location_id
#     transactional_data = {}
#     table_num = 1

#     while True:
#         print(product_id, current_location, start_date, end_date)
#         transaction_data = generate_sql_query(
#             product_id, current_location, start_date, end_date, table_information
#         )
#         transactional_data[f"table_number_{table_num}"] = transaction_data

#         row = df_llm[df_llm["location_id"] == current_location]
#         if row.empty:
#             break

#         ship_from = row["ship_from_loc_id"].iloc[0]
#         lead_time = int(row["transportation_lead_time"].iloc[0])

#         current_location = ship_from
#         start_date = start_date - timedelta(weeks=lead_time)
#         end_date = end_date - timedelta(weeks=lead_time)
#         table_num += 1

#     return transactional_data

def llm_transaction_chain(df_llm, product_id, location_id, start_date, end_date, table_information):
    """Trace back through supply chain to collect transactional snapshots"""
    current_location = location_id
    transactional_data = {}
    table_num = 1
 
    while True:
        # print(product_id, current_location, start_date, end_date)
        transaction_data = generate_sql_query(
            product_id, current_location, start_date, end_date, table_information
        )
        transactional_data[f"table_number_{table_num}"] = transaction_data
 
        row = df_llm[df_llm["location_id"] == current_location]
        if row.empty:
            break
 
        ship_from = row["ship_from_loc_id"].iloc[0]
        lead_time = int(row["transportation_lead_time"].iloc[0])
 
        current_location = ship_from
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        start_date = start_date - timedelta(weeks=lead_time)
        end_date = end_date - timedelta(weeks=lead_time)
        table_num += 1
 
    return transactional_data

# Step 7 --- LLM root cause analysis
def get_llm_response(product_id, location_id, stock_status_warning, transactional_data):
    """Use LLM to explain stock imbalance reasons"""
    stock_status = stock_status_warning.split("_")[0]

    general_prompt = f"""
        You are an expert in Inventory Planning.

        Task
        - Identify the reason(s) for the {stock_status} observed at the (product_id, location_id) level.
          product_id: {product_id}, location_id: {location_id}

        Inputs
        - Supply Chain Context (broad knowledge of supply chain operations, constraints, and planning principles):
          {supply_chain_context}

        - Product Supply Chain Flow (specific flow for this product from vendor → plant → RDC → DC → customer,
          including production/transport lead times, minimum/maximum capacities, and constraints):
          {product_supply_chain_flow}

        - Possible Root-Cause Scenarios (most likely, business-relevant drivers of {stock_status}):
          {stock_status_scenarios}

        - Predicted Plan Snapshots (dictionary of snapshots, each representing a view of the data at a given week):
            - `current_snapshot` = latest plan (most recent refresh).
            - `1_week_earlier_snapshot`, `2_week_earlier_snapshot`, ... = historical snapshots.:
          {transactional_data}

        - Data Dictionary (defines meanings of all fields like projected_stock, safety_stock, demand, receipts, lead times, lot sizes):
          {data_description}

        Instructions
          1. Use the Data Dictionary to correctly interpret field meanings.
          2. First, analyze the `current_snapshot` in isolation.  
            - Identify all possible reasons for {stock_status} using only this snapshot.  
          3. If no clear reasons can be found from the `current_snapshot`,  
            then compare it with earlier snapshots (`1_week_earlier_snapshot`, `2_week_earlier_snapshot`, etc.)  
            to detect trends or changes in:
                - demand forecasts
                - scheduled receipts
                - production quantities
                - safety stock
                - lot size policies  
          4. Only use historical comparisons when the `current_snapshot` alone does not provide sufficient explanation.  
            Clearly separate insights coming directly from the `current_snapshot` vs. those derived from historical trends.
          5. Use the Product Supply Chain Flow to trace upstream dependencies (vendor/plant/RDC) and explain how
            those locations affect the observed {stock_status}.
          6. Cross-check findings with the Possible Root-Cause Scenarios to ensure alignment with realistic supply chain drivers.
          7. If multiple reasons exist, provide up to three, each with a priority.
          8. Support each reason with actual numerical evidence across weeks and across snapshots (show differences by refresh_date).
          9. If required fields are missing, explicitly state which ones are unavailable and how that limits reasoning.
          10.- If no valid root cause can be identified, output:
              {{ "L1 Reason": "Unclear Root Cause", "L2 Reason": "No clear operational driver could be identified from available snapshot data.", "Priority": "Low" }}


        Output
        - Provide reasons in JSON format.
        - Each reason must contain:
            "L1 Reason": a short 3-4 word business heading,
            "L2 Reason": L2 Reason must include two parts--
                        - A brief explanation of the L1 reason using current evidence data over the weeks of instance duration  
                        - Support using data values to your explaination,
            "Priority": High/Medium/Low.
        - Include upstream location reasoning (previous supply chain step) as one of the reasons if relevant.
        - Limit total reasons to 3 maximum.
        - Do not invent values; base explanations only on provided data.

        Example Output:
          {{
          "Alert": "Excess observed for product FG-100-001 and location DC1000 during Week 30-2025 to Week 47-2025",
          "Reason 1": {{
          "L1 Reason": "Larger Lot Size",
          "L2 Reason": "At the beginning of the overstock period , the stock on hand was 5000 units, which is significantly higher than the safety stock of 500 units.The production lead time is 8 weeks, which is longer than the duration of the overstock instance (9 weeks). This longer production lead time in the production plant resulted in maintaining high stock on hand to support lead time horizon demand.",
          "Priority": "High"
          }},
          "Reason 2": {{
          "L1 Reason": "Overforecasting Demand",
          "L2 Reason": " The current demand and the prior demand values are .....<corresponding reasoning showcase as instructed>.",
          "Priority": "Medium"
          }}
          }}
    """
    try:
        reasons = llm.invoke(general_prompt)
        llm_output = reasons.content.strip()
    except Exception as e:
        print(f"Error invoking LLM for root cause analysis: {e}")
        return ""

    # Clean fenced code if any
    if llm_output.startswith("```"):
        llm_output = llm_output.split("```")[1].replace("json", "").strip()

    return llm_output


# Step 8 --- Main pipeline
def autonomous_pipeline(product_id, location_id, stock_status_warning, start_date, end_date):
    """Main function: traces supply chain and gets root cause analysis"""
    location_source_table = get_lc_table(product_id, location_id)
    transactional_data = llm_transaction_chain(
        location_source_table, product_id, location_id, start_date, end_date, db.table_info
    )
    stock_status_reasons = get_llm_response(
        product_id, location_id, stock_status_warning, transactional_data
    )
    #stock_status_reasons = re.sub(r"\s+", " ", stock_status_reasons)
    return stock_status_reasons


def flow_query_to_alert(query):
    #info extraction
    ans = extract_info_query(query)

    ### 2. Identify instances from infromation
    res = filter_alert_df(ans, alert_df)

    if res['n_matches']==0:
        response_one =  [{'message': 'There is no warning instance with the requested infromation.'}]
        return response_one

    ### 2. Instance shortlisting
    new_alerts_lists = []
    new_alert_df = res['filtered_df'].copy()

    # Columns you want to extract combinations from (order matters)
    columns_to_check = ['product_id', 'location_id','stock_status_warning']
    # Store unique ordered combinations
    unique_combinations = set()
    for _, row in new_alert_df.iterrows():
        combo = tuple(row[col] for col in columns_to_check)  # keeps order
        unique_combinations.add(combo)

    # Display results
    for combo in unique_combinations:
        new_alerts_lists.append(list(combo))
    print("Detected Instances : \n", new_alerts_lists)

    ######### Genereting alerts through the flow
    final_response = []
    for alert in new_alerts_lists:
        filter_values = {
            'product_id': alert[0],
            'location_id': alert[1],
            'stock_status_warning':alert[2]
        }
        # Apply filtering
        df_stock_status_alert = df_stock_status.copy()
        for col, val in filter_values.items():
            df_stock_status_alert = df_stock_status_alert[df_stock_status_alert[col] == val]

        # -------------------- Getting Master Data Traversal --------------------
        product = alert[0]
        location = alert[1]
        resources, raws, vendors, plant, lead_time, trace_steps, final_dests = trace_bom_verbose_with_final_dest_and_vendors(product, location)

        final_result = {
            "product_requested": product,
            "location_requested": location,
            "final_plant": plant,
            "total_lead_time_weeks": lead_time,
            "resources_needed": list(resources),
            "raw_materials_needed": list(raws),
            "vendors_associated": vendors,   # <-- NEW
            "trace_steps": trace_steps,
            "final_destinations": list(final_dests)
        }

        detected_instance = alert
        #print('----------- Detected Instance---------\n',detected_instance)
        # for k, v in final_result.items():
        #     print(f"{k}: {v}")

        # Gather Data
        if final_result['resources_needed']:
            df_capcity_data = gather_capacity_data(df_stock_status_alert,final_result)
            df_other_dc = gather_other_dc_data(df_stock_status_alert,final_result)
        else:
            df_capcity_data = pd.DataFrame()
            df_other_dc = pd.DataFrame()
            #print('Capacity data not applicable')
        if final_result['vendors_associated']:
            df_raw_mat = gather_raw_mat_data(df_stock_status_alert,final_result)
        else:
            df_raw_mat = pd.DataFrame()
            #print('Raw material data not applicable')
        if final_result['final_plant']:
            df_rv_plant =  gather_plant_data(df_stock_status_alert,final_result)
        else:
            df_rv_plant = pd.DataFrame()
            #print('Plant data not applicable')

        df_direct_vendor = gather_direct_vendor_data(df_stock_status_alert,final_result)
        #print(df_direct_vendor)

        ## Final LLM Response:
        if "overstock" in detected_instance[2].lower():
            response = execute_pipeline_overstock(df_stock_status_alert ,df_capcity_data, final_result, llm)
            # final_response.append(response)
            final_response.append(response["data"]["explanation"])
            #print(response["data"]["explanation"] if "data" in response else response["error"])
        elif "understock" in detected_instance[2].lower():
            response = execute_pipeline_understock(df_stock_status_alert ,df_capcity_data, df_other_dc, df_raw_mat, df_rv_plant,df_direct_vendor,final_result, llm)
            #final_response.append(response)
            final_response.append(response["data"]["explanation"])
            #print(response["data"]["explanation"] if "data" in response else response["error"])
        # else:
        #     print("Neither overstock nor understock found.")

        if "autonomous" in final_response[-1].lower():
            # delete last guided failed response and trigger autonomous response
            final_response = final_response[:-1]
            # start_date and end date for autonomous
            min_date = df_stock_status_alert["week_end_date"].min().strftime("%Y-%m-%d")
            max_date = df_stock_status_alert["week_end_date"].max().strftime("%Y-%m-%d")

            product_id,location_id,stock_status_warning,start_date,end_date = alert[0],alert[1],alert[2],str(min_date),str(max_date)
            reasons_all = autonomous_pipeline(product_id, location_id, stock_status_warning, start_date, end_date)
            final_response.append(reasons_all)

    return final_response