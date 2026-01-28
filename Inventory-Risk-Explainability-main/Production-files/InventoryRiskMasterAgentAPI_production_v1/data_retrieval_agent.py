# import libraries
import os
from gen_ai_hub.proxy.langchain.init_models import init_llm
from sqlalchemy import create_engine,inspect,text
from urllib.parse import quote_plus
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langgraph.prebuilt import create_react_agent
import json


llm_model = init_llm('gpt-4o', max_tokens=16000, temperature = 0)


HANA_HOST = os.getenv("HANA_HOST")
HANA_USER = os.getenv("HANA_USER")
HANA_PASSWORD = os.getenv("HANA_PASSWORD")
schema_name = "CURRENT_INVT"
port = "443"

user_enc = quote_plus(HANA_USER)
password_enc = quote_plus(HANA_PASSWORD)
connection_str = f"hana://{user_enc}:{password_enc}@{HANA_HOST}:{port}/?currentSchema={schema_name}"

engine = create_engine(connection_str)
sql_database = SQLDatabase(engine,schema=schema_name)

# llm_model = init_llm('gpt-4o', max_tokens=10000, temperature=0)

toolkit = SQLDatabaseToolkit(db=sql_database, llm=llm_model)

information_retirval_tools = toolkit.get_tools()


# metadata = table defination, table schema and relations
with open('data_schema.txt', 'r') as file:
    data_schema = file.read()
 
with open('Data_description.txt', 'r') as file:
    data_description = file.read()


nl_sql_prompt  = f"""
            You are an agent designed to interact with a SQL database.
            Given an input question, create a syntactically correct hana query to run.
            then look at the results of the query and return the answer.
            Data schema: {data_schema}.

            Data description: {data_description}         

            You MUST double check your query before executing it. If you get an error while
            executing a query, rewrite the query and try again.

            To start you should ALWAYS look at the tables in the database to see what you
            can query. Do NOT skip this step.
            Make sure to qualify column names with their respective table names.
            Never invent or assume table or column names. If the schema does not define a specific field, do not reference it.
            When the user query is about overstock or understock instance, first fetch the instance details from stock_Status_v2 table then get the further details by quering the transcation tables (Review DC, Review Plant, etc).
            In other cases query the master table and transaction table.

            When the query directly provides numeric inputs (for example, given values for minimum demand, maximum demand, and average demand, or any numeric inputs used for calculation),
            do NOT create or execute a SQL query. Instead, perform the requested calculation analytically using formulas and show each step clearly.
            - Display the formulas used and substitute the numbers given.
            - Show each intermediate calculation.
            - Include interpretations where appropriate (e.g., if variation is high, explain what that means for planning or lot sizing).
            - Use a numbered, step-by-step format.
            - Round results sensibly (for example, to the nearest 10 if appropriate).
            - If the question involves lot size, demand variation, or rule-of-thumb calculations, follow this reasoning pattern:
                1) Compute demand range and variation ratio:
                   Range = Dmax − Dmin
                   Variation ratio = Range / Davg
                   Include the calculated results and short interpretation.
                2) Determine minimum lot size (Qmin) as approximately equal to average demand (Qmin = Davg), and round sensibly.
                3) For variation-based sizing, compute 25%, 30%, and 35% of average demand and show rounded values.
                4) Interpret what these values mean (e.g., higher variation suggests larger lot sizes).

            Then you should query the schema of the most relevant tables. Give only the valid sql query. No additional text.
            DO NOT mention about column names in the header.
            Expand week ranges into individual week fields in SQL.
            Always interpret “today”, “current”, “as of now”, or “from current date” as the database function CURRENT_DATE.
            If user inquired about next n weeks start from current date.
            DO NOT return the sql query as response.
            If no location/product is specified then response should contain data about all locations/products.
            Provide only the final business result in natural language, including computed values, summaries, and insights.
            Focus on answering the user's question directly (e.g., totals, trends, or comparisons) without exposing technical implementation details.

            When the user requests one or more aggregate values (for example maximum, minimum, average, total, sum, or count),
            Step 1. First create a query that retrieves all the relevant data records based on the provided filters (such as product, location, and time period). 
                    The query should extract the complete set of values needed to calculate these metrics.
                    Do not apply aggregate functions like MAX(), MIN(), or AVG() directly inside SQL unless explicitly instructed. 
                    If the dataset includes multiple numeric columns (such as week-wise values), unpivot or flatten them so each numeric value can be included in the calculations.

            Step 2. After receiving the dataset from Step 1, you may perform another SQL query or a precise numeric calculation on that dataset to compute the requested aggregates.
                    To calculate average or mean - (sum of all values)/(count of values)
                    To find maximum and minimum - pick the highest and lowest numeric values.
                    To calculate total or sum - add all numeric values.
                    Always base these operations strictly on the numeric results retrieved in Step 1.

            Step 3. Provide the required numbers as the response only. Do not explain the formula or show the SQL statements used in the background.
            """


information_retrieval_agent = create_react_agent(
    model=llm_model,
    tools=information_retirval_tools,
    name="information_retrieval_agent",
    prompt=nl_sql_prompt
)

def get_response(query):

    try:

        response_text = ""
        for step in information_retrieval_agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            # step["messages"][-1].pretty_print()
            response_text = step["messages"][-1].content
        # print(response_text)
        return json.dumps(
            {
                "info_retrival_ans": response_text
            },
            default=str # pretty-print
        )

    except:
        return json.dumps(
        {
            "info_retrival_ans": "I wasn’t able to generate a response at the moment. Please try again."
        },
        default=str # pretty-print
    )

