import os
import base64
import json
import io
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from sqlalchemy import create_engine, text
# from data_retrieval_agent import get_response
from urllib.parse import quote_plus
from PIL import Image
from langsmith import traceable
# ---- Import your AI modules ----
from gen_ai_hub.proxy.langchain.init_models import init_llm
from reasoning_agent_pipeline import reasoning_execution_pipeline
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
import re
from datetime import datetime
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import Table, Column, String, DateTime, MetaData, insert
from datetime import datetime
from memory_retention import format_conversation_memory
from memory_retention import add_to_memory
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# --- Flask setup ---
app = Flask(__name__)
# Enable CORS for all routes and origins
CORS(app)
port = int(os.environ.get('PORT', 8080))


# --- Environment setup ---
# These should be set via environment variables before running the application
# Required: AICORE_AUTH_URL, AICORE_CLIENT_ID, AICORE_CLIENT_SECRET, AICORE_BASE_URL,
#           AICORE_RESOURCE_GROUP, TAVILY_API_KEY, LANGSMITH_API_KEY, HANA_HOST, HANA_PASSWORD

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "test_langsmith_prod"

# --- DB connection ---
host = os.getenv("HANA_HOST", "")
port = "443"
schema_name = "CURRENT_INVT"
user = os.getenv("HANA_USER", "DBADMIN")
password = os.getenv("HANA_PASSWORD", "")
user_enc = quote_plus(user)
password_enc = quote_plus(password)
connection_str = f"hana://{user_enc}:{password_enc}@{host}:{port}/?currentSchema={schema_name}"
engine = create_engine(connection_str)
sql_database = SQLDatabase(engine,schema=schema_name)

# --- Model ---
llm_model = init_llm('gpt-4o', max_tokens=10000, temperature=0)

toolkit = SQLDatabaseToolkit(db=sql_database, llm=llm_model)

information_retirval_tools = toolkit.get_tools()

# =========================
# Query Refiner
# =========================
def refine_user_query(query, conversation_memory):
    """
    Refines the user query by considering the conversation history.
    Uses the same LLM model for refinement.
    """
    try:
        conversation_memory = format_conversation_memory()

        refinement_prompt = f"""
        You are a query refinement assistant.
        Your task is to take the current user query and the past conversation history,
        and refine/modify the query to make it self-contained and more precise.
        Conversation history:{conversation_memory}
        Current query: {query}
        Refined query (only return the refined query, no explanations):
"""

        response = llm_model.invoke([HumanMessage(content=refinement_prompt)])
        refined_query = response.content.strip()

        return refined_query

    except Exception as e:
        return f"[Query Refinement Failed: {str(e)}]"

# =========================
# Table Import (History Table - Latest 10 Records)
# =========================

def load_latest_records(engine, schema_name, table_name, order_col="QueryDateTime", limit=10):
    try:
        query = text(f'''SELECT * 
                        FROM "{schema_name}"."{table_name}" 
                        ORDER BY "{order_col}" DESC
                        LIMIT {limit} ''')
        
        df = pd.read_sql(query, engine)
        print(f"Loaded {table_name} (latest {len(df)} rows)")
        return df
    except Exception as e:
        print(f"Error loading {table_name}: {e}")
        return pd.DataFrame()



# --- Graph Plotting ---
@traceable
def plot_graph_from_query(df, user_query, explanation, llm, filename="chart.png"):
    try:
        trimmed_df = False
        if len(df) <=10:
            sample_data = df.to_dict(orient="records")
        else:
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            representative_records = df.drop_duplicates(subset=categorical_cols)
            sample_data = representative_records.to_dict(orient="records")
            trimmed_df = True
        
        prompt = f"""You are an assistant that generates valid Python code to create an insightful visualization using matplotlib. 
        You are given:
        query : {user_query}
        response: {explanation}
        schema: {list(df.columns)}
        
        sample_data: {sample_data}

        - Follow color schema:
          background: #FFFFFF (white)
          text/labels: #2B2C2E (black)
          markers/lines: #4BC884, #22A6BD
        - Consider the categories for deciding the type of graph
        
        - Add data labels only if meaningful and avoid overlap (skip labels if they would clutter the chart)
        - Save with: plt.savefig("{filename}", facecolor=fig.get_facecolor())
        - Close with: plt.close(fig)


        Output only Python code. If no valid chart, return exactly: No graph generated
        """

        try:
            code = llm.invoke(prompt)
        except BaseException as be:  # catches SystemExit, KeyboardInterrupt too
            print(f"Fatal error from LLM call: {be}")
            return ""
        if "No graph generated" in str(code):
            return ""

        # Clean response
        code = str(code.content)
        code = re.sub(r"```python", "", code)
        code = re.sub(r"```", "", code)
        # print("Generated Code:\n", code)
        if trimmed_df:
            code = re.sub(r"(?s)data\s*=\s*\[.*?\]", "", code)
            code = re.sub(r"df\s*=\s*pd\.DataFrame\(.*?\)", "", code)
            # Provide df in exec environment
            local_vars = {}
            if 'df' in code:
                local_vars = {"df": df, "plt": plt, "pd": pd}
            elif 'data' in code:
                local_vars = {"data": df, "plt": plt, "pd": pd}
                
            exec(code, {}, local_vars)
        else:
            exec(code)

        image_path = 'chart.png'
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    except Exception as e:
        print(f"Error generating graph: {e}")
        encoded_image = ""

    return encoded_image


def save_conversation_to_db(engine, schema_name, UserID, SessionID, UserQuery, LLMResponse):
    """
    Insert a record into QUERY_HISTORY table.

    Parameters:
        engine: SQLAlchemy engine object
        schema_name (str): Schema name where the table resides
        UserID (str): ID of the user
        UserQuery (str): User's query text
        LLMResponse (str): Response from LLM
    """
    # =========================
    # Define table metadata
    # =========================
    metadata = MetaData(schema=schema_name)

    query_history = Table(
        "QUERY_HISTORY",
        metadata,
        Column("UserID", String(255)),
        Column("SessionID", String(255)),
        Column("QueryDateTime", DateTime),
        Column("UserQuery", String(10000000)),
        Column("LLMResponse", String(10000000)),
        extend_existing=True
    )

    # =========================
    # Data to insert
    # =========================
    data_dict = {
        "UserID": UserID,
        "SessionID": SessionID,
        "QueryDateTime": datetime.now(),
        "UserQuery": UserQuery,
        "LLMResponse": LLMResponse
    }

    # =========================
    # Insert into DB
    # =========================
    with engine.connect() as connection:
        stmt = insert(query_history).values(**data_dict)
        connection.execute(stmt)
        connection.commit()
        print("Data inserted successfully!")


# --- Agents ---
@tool
def reasoning_pipeline_tool(user_input: str):
    """Runs the reasoning pipeline and generate reasons and returns results."""
    print("-"*100)
    print("resoning agent invoked")
    return reasoning_execution_pipeline(user_input)

reasoning_agent = create_react_agent(
    model=llm_model,
    tools=[reasoning_pipeline_tool],
    name="reasoning_agent",
    prompt="You are an expert Inventory Risk Management Reasoning Agent. Your task is to generate the reasons for overstock and understock instance occurance"
)

# @tool
# def information_retrieval_tool(user_input: str):
#     """Runs the information retrival pipeline and returns results."""
#     print("-"*100)
#     print("information agent invoked")
#     return get_response(user_input)

# information_retrieval_agent = create_react_agent(
#     model=llm_model,
#     tools=[information_retrieval_tool],
#     name="information_retrieval_agent",
#     prompt="""
#     You are an information retrieval agent.
#         - use the `information_retrieval_tool` to generate response.
#         - Pass the user’s query exactly as it was written.
#         - Do NOT rewrite, translate, or convert the query into SQL.
#         - Simply forward the original query string unchanged.
#         """
    
#     )


toolkit = SQLDatabaseToolkit(db=sql_database, llm=llm_model)

information_retirval_tools = toolkit.get_tools()
# metadata = table defination, table schema and relations
with open('data_schema.txt', 'r') as file:
    data_schema = file.read()
 
with open('Data_description.txt', 'r') as file:
    data_description = file.read()


nl_sql_prompt = nl_sql_prompt = f"""
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
            When the user query is about overstock or understock instance, first fetch the instance details from stock_Status_v2 table then get the further details by quering the transcation tables (Review DC, Review Plant, etc).
            In other cases query the master table and transaction table
            Then you should query the schema of the most relevant tables. Give only the valid sql query. No additional text
            DO NOT mention about column names in the header.
            Expand week ranges into individual week fields in SQL
            Always interpret “today”, “current”, “as of now”, or “from current date” as the database function CURRENT_DATE.
            DO NOT return the sql query as response. 
            """

information_retrieval_agent = create_react_agent(
    model=llm_model,
    tools=information_retirval_tools,
    name="information_retrieval_agent",
    prompt=nl_sql_prompt
)


# --- Supervisor ---
supervisor = create_supervisor(
    model=llm_model,
    agents=[reasoning_agent, information_retrieval_agent],
        prompt="""You are the Supervisor Agent. 
                You oversee and delegate tasks between two specialized agents:

                1. Reasoning Agent  
                - Handles analytical, logical, and risk management reasoning tasks.  When user query explicity asks for a reason invoke this agent
                - Use this agent when the task requires reasoning. The user query is about why overstock or understock is occuring  

                2. Information Retrieval  Agent  
                - Handles database-related tasks such as generating queries, retrieving structured data, and executing SQL.  
                - Use this agent whenever information retrieval from the database is required.  

                Rules for Delegation:
                - Assign tasks to only one agent at a time. Do not call multiple agents in parallel.
                - Do not translate, rewrite, or reformulate the user's question.
                - Always select the agent that is best suited for the current user request.  
                - Do not attempt to solve or execute tasks yourself; your role is purely delegation and coordination.  
                - After the assigned agent completes its work, 
                    return its response directly as the final output. 
                    Do not rephrase, wrap, or alter the agent’s output. : 
                - The task is complete and the final result can be returned, or  
                - Additional work must be handed off to the same or another agent.  
                Always call the respective agent, never execute the tasks yourself.
                Your objective is to ensure tasks are completed efficiently by leveraging the agents expertise without doing the work directly yourself.
                """,
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()

# --- Core pipeline ---
@traceable
def run_supervisor(query, conversation_memory):
    #refined_query = refine_query_with_history(llm_model, query)
    try:

        ## sessions memory:
        conversation_memory = add_to_memory("user", query)

        ## query refiner:
        query = refine_user_query(query, conversation_memory)
        print('Modified Query :', query)
        response = supervisor.invoke({"messages": [{"role": "user", "content": query}]})
        explanation, data_preview = "", ""
        df = pd.DataFrame()
        delegated = False
        messages = response.get("messages", [])

        final_ans = ""
        for message in messages:

            if "Chain of thought" in message.content:
                ans = message.content
                delegated = True
                repharse_ans_prompt = f"""Rephrase the understock/Overstock analysis response to strictly follow this format:

                Format Template:
                The reason for understock/Overstock is [L1 reason 1 and L1 reason 2...] because [combined explanation from all L2 reasons + supporting COT details]

                Instructions:

                Extract all L1 reasons from the JSON response and join them with "and".

                Create a single, unified explanation by combining:

                All L2 reasons (merged naturally without repetition).

                The most critical parts of the Chain of Thought (COT) that strengthen those L2 reasons.

                Ensure the explanation flows logically and avoids repeating the same sentence/idea.

                Produce only one formatted output (do not repeat the response).
                
                ans: {ans}
                """
                #print(ans)
                repharsed_ans = llm_model.invoke(repharse_ans_prompt)
                explanation = repharsed_ans.content

            response = message.content
                
            # --- Fallback Policy ---
        if not delegated:
            explanation = response
            
        # convetring dataframe to json for output payload. The json is then displayed in tabluar format on UI
        data = df.head(15)
        data = data.fillna("")
        # Convert DataFrame to JSON format as per your requirement
        json_result = data.to_dict(orient='index')

        #log_query_and_response(engine, query, explanation)

        # Adding row labels to match the required JSON structure
        json_result = {f'row{index + 1}': row for index, row in json_result.items()}

        encoded_graph = plot_graph_from_query(df, query,explanation, llm_model) if df is not None else ""
        if "```json" in encoded_graph:
            encoded_graph = ""

        conversation_memory = add_to_memory("ai", explanation)

        return explanation, encoded_graph, json_result,conversation_memory
    except:
        explanation = "I wasn’t able to generate a response at the moment. Please try again."
        data = pd.DataFrame()
        json_result = data.to_dict(orient='index')

        conversation_memory = add_to_memory("ai", explanation)

        return explanation,"",json_result, conversation_memory



@app.route("/handle_query", methods=["POST"])
def handle_query():

    input_data = request.get_json()
    query = input_data.get("query")
    # UserID = input_data.get("UserID")
    conversation_memory = input_data.get("conversation_memory")
    conversation_memory = []
    response, graph, json_result, conversation_memory_updated = run_supervisor(query, conversation_memory)
    serialized_memory = [m.dict() for m in conversation_memory_updated]
    return jsonify(
        {"response": response},
        {"graph_base64": graph},
        {"json_data" : json_result},
        {"conversation_memory" : serialized_memory}
    )
    # print(response)
    # print(conversation_memory_updated)

if __name__ == "__main__":
  # Cloud Foundry provides PORT env
    app.run(host='0.0.0.0', port=8080,debug=True)

# handle_query()