#-- import libraries
import warnings
warnings.filterwarnings("ignore")
from sqlalchemy import create_engine,inspect,text
from urllib.parse import quote_plus
import os
import re
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

from datetime import datetime
from datetime import date, timedelta
import json
from gen_ai_hub.proxy.langchain.init_models import init_llm, init_embedding_model


## LLM Credentials:
os.environ["AICORE_AUTH_URL"] = "https://btp-ai-developments-sl2f9ys4.authentication.eu10.hana.ondemand.com"
os.environ["AICORE_CLIENT_ID"] = "sb-38176009-b499-470f-a3b8-9cf98daac1d0!b503699|aicore!b540"
os.environ["AICORE_CLIENT_SECRET"] = "1ac5c77f-d5ac-4e2d-8c19-6ffc47113ec8$52U4q9NYAN-GBm23a2lm_SFVrzmWNhuS7l_qFXs4s4A="
os.environ["AICORE_BASE_URL"] = "https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com"
os.environ["AICORE_RESOURCE_GROUP"]="default"

# llm_model = 'gpt-4o-mini'
llm_model = 'gpt-4o'
llm = init_llm(llm_model, max_tokens=16384, temperature=0)

## configuring database instance
host = "cfe32093-429a-4e59-87dc-9f3e4da891bf.hna2.prod-eu10.hanacloud.ondemand.com"
port = "443"
# schema_name = "INVT_XAI"
schema_name = "INVT_XAI_RAW"
user = "DBADMIN"
password = "Bcone@1234567"

## Encode user and password
user_enc = quote_plus(user)
password_enc = quote_plus(password)

# Connecting to S4 HANA DB:
connection_str = f"hana://{user_enc}:{password_enc}@{host}:{port}/?currentSchema={schema_name}"
engine = create_engine(connection_str)

# --Functions for Supply chain analysis and data manipulation
# functions to call all the required data

## Importing data from HANA DB:

def import_table_data(engine, schema_name, table_name):
    # Use the engine to connect and read data
    query = f'SELECT * FROM "{schema_name}"."{table_name}"'
    conn = engine.connect()
    df = pd.read_sql_query(text(query), con = conn)
    return df


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

## Get scenario number from the user input

def query_details_instance(query,):


    extract_prompt = """Your task is to extract product ID , Location ID and time data like week number with year or complete date from the query the user has asked. 
    Details of the fields are - 
    - product_id(varchar): Unique identifier of the product (e.g., FG-100-001, FG-200-001).
    - location_id(varchar): Identifier of the distribution center (e.g., DC1000, RDC1000).
    - week_num(integer): An integer (1-52) representing the ISO week number of the year.
    - year (integer): Four-digit calendar year associated with the week_num.
    - week_end_date (date): Date (ISO format) representing the last day (typically Sunday) of the given week_num and year.

    Instructions:
    - First identify the said values and then strictly respond in provided sample format. 
    - If any of the value is not present then respond its value as "Not provided by user"
    - If there are multiple dates then provide the one that is the earliest.
    - If week number is provided, make sure you also extract year value from the query along with week value 
    Sample Response format 1 -
    {
    "product_id" : "FG-100-001",
    "location_id": "DC2000",
    "week_num" : 30,
    "year": 2025,
    "week_end_date" :  "Not provided by user"
    }
    Sample Response format 2 -
    {
    "product_id" : "FG-100-001",
    "location_id": "DC2000",
    "week_num" : "Not provided by user",
    "year": "Not provided by user",
    "week_end_date" :  "2025-07-27"
    }
    """

    messages = [
        {"role": "system", "content": extract_prompt},
        {"role": "user", "content": query}
    ]
    try:
        response = llm.invoke(messages)
    except Exception as e:
        print(f"Error while getting response from LLM: {e}")

    #print("\n--------------- Gathered response from user --------------\n",response.content.strip())

    #### Identifying the instance ###
    df_stock_status = import_table_data(engine, "CURRENT_INVT", "STOCK_STATUS_V2")
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
    #print(alerts_lists)

    df = df_stock_status.copy()
    filter_str = response.content.strip()

    # Convert string to dictionary
    filter_dict = json.loads(filter_str)

    # Remove keys where the value is "Not provided by user" or None
    filter_dict = {k: v for k, v in filter_dict.items() if v != "Not provided by user" and v is not None}

    # Apply filtering
    smp_filtered_df = df.copy()
    for col, val in filter_dict.items():
        smp_filtered_df = smp_filtered_df[smp_filtered_df[col] == val]

    # filtering on the instance
    prod = smp_filtered_df["product_id"].iloc[0]
    loctn = smp_filtered_df["location_id"].iloc[0]
    num_instance = smp_filtered_df["stock_status_warning"].iloc[0]
    sub_al = []
    for al in alerts_lists:
        if al[0]== prod and al[1]==loctn and al[2]==num_instance:
            #print(al)
            sub_al.append(al)

    detected_instance = sub_al[0]

    #print(f'# Identified the understock instance as \n {sub_al[0]} #')

    ############### getting the data for the instance

    final_response = []
    df_stock_status_alert = pd.DataFrame()
    # sub_al
    # for alert in alerts_lists:
    for alert in sub_al:
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
    product = filter_dict['product_id']
    location = filter_dict['location_id']
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

    detected_instance = sub_al[0]
    return df_stock_status_alert,final_result,detected_instance

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

#  Overstock and Undertsock pipeline

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
- Strictly avoid assumptions or generic explanations. All insights must be grounded in the provided data.
- Use data as evidence to showcase your summary of reasoning for overstock patterns in L2 Reasons.
- Verify that every statement is directly supported by data followed by your conclusion on that data and logic.
- Thus, Data evidence and should explained step wise in the output filed - 'Chain of thought'
- There can be single or multiple L1 and L2 reasons. Thus, It is important to mention all the applicable L1 and L2 reasons in you 'Chain of thought' which are proved to be the reason. 
- Thus during reasoning,if multiple checks are passed while following steps, then make sure to specify all the corresponding L1 and L2 reasons as the cause for the instance

Explanation Guidelines:
- The alert must be based on one unique (product_id, location_id) instance.
- Supply Chain Flow: VEN ➝ Plant ➝ RDC ➝ DC
- 'Product Supply Chain details' for the instance product id and location id that is to be investigated as requested is provided below. Other deatils like source of product i.e. Plant, Vendor and raw materials (with their Vendor) needed is also provided.After that complete tracing of the product with their lead times in week is provided. Finally all the DC location where the origins Plants product gets divided for distribution is provided. The details are as follows:
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
- product_id(varchar): Unique identifier of the product (e.g., FG-1000, FG-2000).
- location_id(varchar): Identifier of the distribution center (e.g., DC1000, RDC1000).
- week_num(integer): An integer (1-52) representing the ISO week number of the year.
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
- transportation_lead_time(integer): time (in weeks) required for traversal of a product from RDC to the DC
- minimum_transportation(integer): minimum number of units that can be transported to the current location
- incremental_transportation(integer): additional minimum incremental units that can be added to minimum transportation unit to transport more units to the current location
- production_lead_time(integer): time (in weeks) required for the production of the good
- offset_stock(integer): difference between projected stock and safety stock that signifies amount of stock at the end of the week
- lag_1_dependent_demand: These are depedent demand value predicted a week prior to current prediction data	
- lag_1_supply_orders: These are supply order value predicted a week prior to current prediction data	
- ven_300_001_maximum_external_receipt: the unit of supply by provided by Vendor 'VEN-300-001'. This vendor specifically supplies for product 'FG-300-001'.

Alert explanation must follow this format template:
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
        Step 1. First look at that individual instance of overstock and confirm the checks given in subsequent steps. Based on the confirmed checks, provide all the passed check-wise L1 and L2 reasons as instructed.
        Step 2. Find the 'stock on hand' value for the first week of the instance period i.e. the first week of the instance and check If 'stock on hand' value greater than 0
        Step 3. If check succeeds then check if the value is greater than 'safety stock' value for the same week (i.e. first week of the instance)
        Step 4. If the check is true then respond that the cause of overstock is because of high amount of 'stock on hand' at the start of the period as L1 (Level 1) reason
        Step 5. For (Level 2) L2 reason check 'transportation lead time' first and later 'production lead time' 
        sub step 1 - if 'transportation lead time' is less than 3 weeks short than the duration (number of weeks i.e. records) in the overstock instance then say L2 reason as Longer Transportation Lead Time travel to the location resulted in High Stock on Hand to support Lead Time Horizon Demand.
        sub step 2 - If 'transportation lead time' is more than 3 weeks short than the instance duration then check if 'production lead time' is less than 3 weeks short than the duration (number of weeks i.e. records). If 'production lead time' qualifies the check then say the L2 reason as Longer Production Lead Time in the Production Plant, High Stock on Hand to support Lead Time Horizon Demand.
        sub step 3 - If both of the sub steps checks fails, then see if the addition of  'transportation lead time' and 'production lead time' is more than 3 weeks short than the instance duration. If check passes then say L2 reason as Longer Production Lead time and Transportation Lead Time travel to the location combined resulted in High Stock on Hand to support Lead Time Horizon Demand.
        Step 6. Moving to next reason verification, check if 'incoming receipts' for first week of the instance period has value greater than 'total demand'
        Step 7. If check is true, then say that L1 reason as Larger Lot Size is the reason for overstocking 
        sub step 1 - For gain L2 reason, check if any present values of 'incoming receipts' has value as added multiples of 'incremental transportation' value to 'minimum transportation'(e.g. 'minimum transportation' 600 + n* 'incremental transportation' 100 -> 700,800,900,etc )  
        sub step 2 - If check is true, then say that L2 reason for overstocking is as Demand is lesser than Rounding Transportation Lot Size 
        sub step 3 - If check fails, then check if all present values of 'incoming receipts' for the instance duration is exactly same as 'minimum transportation' (i.e. Minimum Transportation Lot Size)
        sub step 4 - if above check is true then say that L2 reason for overstocking is as Demand is lesser than Minimum Transportation Lot Size
        Step 8. Moving to next reason verification, check if 'total demand' is same as 'lag_1_dependent_demand' for the instance by comparing sum of their values for the duration with each other
        Step 9. If the sum values are different, and the difference shows that  'lag_1_dependent_demand' was higher than 'total demand' sum, that means that the decrease in demand in the current scene as compared to prediction of prior week demand is the reason for understocking. Thus, making L1 reason as Under forecasting Demand and L2 reason as Demand Spike in the Lead Time Horizon compared to last week.
        Step 10. Moving to next reason verification,check if the 'location id' in stock status alert data is of plant. (Plant have location id starting with "PL". e.g.' PL1000')
        Step 11. If the check is true,observe the the capcity data which provided to you (which is already been adjusted according to production lead time) for the complete instance. Detect if there are some weeks where capacity usage of production resource has been zero.
        Step 12. If the above check is true then also check in stock status data if there has been duration of consistent weeks where 'outgoing supply' is higher than 'total depand'. if yes, that means the plant has been prebuilding and stocking up for the upcoming reduction of the resourse capcity. In such case L1 reason is 'Production Prebuild' and L2 reason is 'Resource Capacity(Machine/Labor) is not available/enough in the weeks where demand is present, but available in the early weeks so plant is pre-building'
        Step 13. If all the checks in all previous step fails then say the reason for understock is supply chain management 


L2 Reason must include two parts:
- A brief explanation of the L1 reason using current evidence data over the weeks of instance duration  
- Support using data values to your explaination

Data:
- df_stock_status_alert :\n{stock_data_str}\n
- df_capcity_data :\n{df_capcity_data_str}\n

Only respond with a JSON block of one alert explanation.
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
def df_to_chat_response_understock(df_stock_status_alert ,df_capcity_data, df_other_dc, df_raw_mat, df_rv_plant,final_result, llm ):
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


    prompt = f"""
You are an expert in Inventory Planning.
Your task is to provide L1 (level one) and L2 (level two) reason for inventory imbalance specifically understock instances at the (product_id, location_id) level using the data provided.
L1 is the primary reason by which understock has occurred. While L2 reason provides the cause behind the L1 reason.
- Start Date: {earliest_date.date()}
- End Date: {latest_date.date()}

Important instructions-
- Include Start Date and End Date in the Alert generated.
- Strictly avoid assumptions or generic explanations. All insights must be grounded in the provided data.
- Use data as evidence to showcase your reasoning for understock patterns in L2 Reasons.
- Verify that every statement is directly supported by data followed by your conclusion on that data and logic.
- Thus, Data evidence and should explained step wise in the output filed - 'Chain of thought'
- There can be single or multiple L1 and L2 reasons. Thus, It is important to mention all the applicable L1 and L2 reasons in you 'Chain of thought' which are proved to be the reason. 
- Thus during reasoning,if multiple checks are passed while following steps, then make sure to specify all the corresponding L1 and L2 reasons as the cause for the instance

Explanation Guidelines:
- The alert must be based on one unique (product_id, location_id) instance.
- Supply Chain Flow: VEN ➝ Plant ➝ RDC ➝ DC
- 'Product Supply Chain details' for the instance product id and location id that is to be investigated as requested is provided below. Other deatils like source of product i.e. Plant, Vendor and raw materials (with their Vendor) needed is also provided.After that complete tracing of the product with their lead times in week is provided. Finally all the DC location where the origins Plants product gets divided for distribution is provided. The details are as follows:
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

df_stock_status_alert being the primary instance detection table, its details are:
- Column Name (Data Type): column description
- product_id(varchar): Unique identifier of the product (e.g., FG-1000, FG-2000).
- location_id(varchar): Identifier of the distribution center (e.g., DC1000, RDC1000).
- week_num(integer): An integer (1-52) representing the ISO week number of the year.
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
- transportation_lead_time(integer): time (in weeks) required for traversal of a product from RDC to the DC
- minimum_transportation(integer): minimum number of units that can be transported to the current location
- incremental_transportation(integer): additional minimum incremental units that can be added to minimum transportation unit to transport more units to the current location
- production_lead_time(integer): time (in weeks) required for the production of the good
- offset_stock(integer): difference between projected stock and safety stock that signifies amount of stock at the end of the week
- lag_1_dependent_demand: These are dependent demand value predicted a week prior to current prediction data 
- lag_1_supply_orders: These are supply order value predicted a week prior to current prediction data   
- ven_300_001_maximum_external_receipt: the unit of supply by provided by Vendor 'VEN-300-001'. This vendor specifically supplies for product 'FG-300-001'.

Alert explanation must follow this format:
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
        Step 1. First look at that individual instance of understock and confirm the checks given in subsequent steps. Based on the confirmed checks, provide all the passed check-wise L1 and L2 reasons as instructed.
        Step 2. Check if there exists any value greater than 0 in 'incoming receipts' for that instance period.
        Step 3. If the check is False then say L1 reason is Longer Lead time
        sub step 1. For L2 reasoning, check if 'transportation lead time' greater than 1. if check is true then say L2 reason is Longer Transportation Lead Time Between RDC & DC
        sub step 2. Now if sub step 1 check fails check if 'production lead time' is greater than 1. If check is true then say L2 reason Longer Production Lead Time in the Production Plant
        Step 4. Moving to next reason verification, Check if 'total demand' is same as 'lag_1_dependent_demand' for the instance by comparing sum of their values for the duration with each other
        Step 5. If the sum values are different, and the difference shows that  'lag_1_dependent_demand' was lower than 'total demand' sum, that means that the increase in demand in the current scene as compared to prediction of prior week demand is the reason for understock. Thus, making L1 reason as Underforecasting Demand and L2 reason as Demand Spike in the Lead Time Horizon compared  to Last week.
        Step 6. Moving to next reason verification, check if the product is 'FG-300-001' and check if values exists for ven_300_001_maximum_external_receipt for that product for the intance duration
        Step 7. If values all values are '0' for the instance duration then it means that vendor with id 'VEN-300-001' has stopped supply for the product 'FG-300-001' for that duration and making L1 reason as Supplier Delays and L2 reason as Supplier Capacity Constraints
        Step 8. Moving to next reason verification, Check is if there are non zero 'lag_1_supply_order' values for the instance duration.
        Step 9. If the previous check is true, then check if 'supply order' values are different than 'lag_1_supply_order' values across time as such the 'lag_1_supply_order' has been pushed ahead compared to current orders i.e. 'supply order' resulting in no supply order. If check is true that means the planner has delayed the order. This means that L1 reason is Transportation delays and L2 reason as Delayed Purchase Orders.
        Step 10. Moving to next reason verification, Check if the capacity_usage_of_production_resource for more than 4 consecutive weeks in the given duration is zero or blank in df_capcity_data. df_capcity_data has already week time adjusted according tp the lead time
        Step 11. If the said check is true that means L1 reason is Production delays. Now check if the df_other_dc has 'incoming recipts' for the same duration.
        Step 12. If check is true, then L2 reason is Resource Capacity(Machine/Labor) is not enough to meet the Production Requirement and the availbale stock was supplied to higher priority other DC location. If check Fails then L2 reason is Resource Capacity(Machine/Labor) is not enough to meet the Production Requirement for all DC locations.
        Step 13. Moving to next reason verification, First look at the undertsock instance , note the 'product id' from 'stock status' and consider raw materials that are required to prepare the product that would be provided to you in 'Raw material needed' section
        Step 14. Now from the review component table , look for the raw material by fitering throgh 'product id' column.
        Step 15. Now check if any of the raw material has lower 'planned transport receipts' than than the 'dependent demand' for more than 4 consecutive weeks.
        Step 16. If the above check is true, that means that the vendor was not able to supply the raw material as per the demand. Use the provided data in 'Raw material supplied by Vendor' section to associate vendor with the raw material. Thus, this makes L1 reason Production delays and L2 reason as Raw material shortages due to insufficient supply from vendor  
        Step 17. Moving to next reason verification, First look at the undertsock instance and its duration. Now look at the 'current plant supply data' table that will be provided with its duration already taken into account according to Plant to DC transportation lead time.
        Step 18. Check if there are any 'open production orders' (current orders) in the plant supply data that is not blank or zero. 
        Step 19. If above checks is true, Note the week duration of those orders and the values.
        Step 20. Now check the 'lag1 open production orders' which is historical plan for the same week duration provided in. If the lag1_open_production_orders has been pushed ahead compared to current orders i.e. open_production_orders, that means the planner has delayed the order. This means L1 reason as Production delays and L2 reason as Delayed production runs by planner.
        Step 21. If all the checks in all previous step fails then say the reason for understock is supply chain management 



L2 Reason must include two parts:
- A brief explanation of the L1 reason using current evidence data over the weeks of instance duration  
- Support using data values to your explaination

Data:
- df_stock_status_alert :\n{stock_data_str}\n
- df_capcity_data :\n{df_capcity_data_str}\n
- df_other_dc :\n{df_other_dc_str}\n
- df_raw_mat :\n{df_raw_mat_str}\n
- Raw material needed: \n{list(final_result['vendors_associated'].keys())}\n
- Raw material supplied by Vendor : \n{final_result['vendors_associated']}\n
- current/historic plant supply data:\n{df_rv_plant}\n
Only respond with a JSON block of one alert explanation.
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


def execute_pipeline_understock(df_stock_status_alert ,df_capcity_data, df_other_dc, df_raw_mat, df_rv_plant,final_result, llm):
    if df_stock_status_alert.empty:
        return {"error": "No stock alerts found."}

    try:
        explanation = df_to_chat_response_understock(df_stock_status_alert ,df_capcity_data, df_other_dc, df_raw_mat, df_rv_plant,final_result, llm)
    except Exception as e:
        return {"error": f"LLM Explanation Error: {str(e)}"}

    return {
        "data": {
            "filtered_alerts": df_stock_status_alert.to_dict(orient="records"),
            "explanation": explanation
        }
    }



def reasoning_execution_pipeline(query):

    # Detect Instance
    df_stock_status_alert,final_result,detected_instance = query_details_instance(query)
    print('----------- Detected Instance---------\n',detected_instance)
    print('\n----------- Supply Chain details---------\n')
    for k, v in final_result.items():
        print(f"{k}: {v}")

    # Gather Data
    if final_result['resources_needed']:
        df_capcity_data = gather_capacity_data(df_stock_status_alert,final_result)
        df_other_dc = gather_other_dc_data(df_stock_status_alert,final_result)
    else:
        df_capcity_data = pd.DataFrame()
        df_other_dc = pd.DataFrame()
        print('Capacity data not applicable')
    if final_result['vendors_associated']:
        df_raw_mat = gather_raw_mat_data(df_stock_status_alert,final_result)
    else:
        df_raw_mat = pd.DataFrame()
        print('Raw material data not applicable')
    if final_result['final_plant']:
        df_rv_plant =  gather_plant_data(df_stock_status_alert,final_result)
    else:
        df_rv_plant = pd.DataFrame()
        print('Plant data not applicable')



    ## Final LLM Response:
    final_response = []
    if "overstock" in detected_instance[2].lower():
        response = execute_pipeline_overstock(df_stock_status_alert ,df_capcity_data, final_result, llm)
        final_response.append(response)
        ans = (response["data"]["explanation"] if "data" in response else response["error"])
    elif "understock" in detected_instance[2].lower():
        response = execute_pipeline_understock(df_stock_status_alert ,df_capcity_data, df_other_dc, df_raw_mat, df_rv_plant,final_result, llm)
        final_response.append(response)
        ans = (response["data"]["explanation"] if "data" in response else response["error"])
    else:
        ans = ("Neither overstock nor understock found.")
 
    # sql_df = pd.DataFrame()
    # sql_query = ""
    # return {
    #     "sql_query": sql_query,
    #     "data_preview": sql_df.to_dict(orient='records'),
    #     "explanation": ans
    # }
    return ans
