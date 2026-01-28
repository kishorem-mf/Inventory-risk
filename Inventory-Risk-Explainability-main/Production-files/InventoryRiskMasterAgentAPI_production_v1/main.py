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
from data_retrieval_agent import get_response
from urllib.parse import quote_plus
from langsmith import traceable
# ---- Import your AI modules ----

from gen_ai_hub.proxy.langchain.init_models import init_llm
from reasoning_agent_pipeline import flow_query_to_alert
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import re
from datetime import datetime
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

## Memory Retention:
from sqlalchemy import MetaData, Table, Column, String, DateTime, insert
from memory_retention import format_conversation_memory
from memory_retention import add_to_memory
from memory_retention import reset_memory
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# --- Flask setup ---
app = Flask(__name__)
# Enable CORS for all routes and origins
# CORS(app)
# Enable CORS for all routes, all origins, and all methods
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


port = int(os.environ.get('PORT', 8080))

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

# --- Model ---
llm_model = init_llm('gpt-4o', max_tokens=16000, temperature=0)

toolkit = SQLDatabaseToolkit(db=sql_database, llm=llm_model)

information_retirval_tools = toolkit.get_tools()


# =========================
# Query Refiner
# =========================
def refine_user_query(query, chat_type, conversation_memory, past_history):
    """
    Refines the user query by considering the conversation history.
    Uses the same LLM model for refinement.
    """
    try:
        if chat_type == 'new_chat':
            conversation_memory = format_conversation_memory()
            past_history = []

        elif chat_type == 'same_chat':
            conversation_memory = format_conversation_memory()
            past_history = []
            # pass

        elif chat_type == 'past_chat':
            past_historical_chat = past_history
            conversation_memory = []

        refinement_prompt = f"""
        You are a query refinement assistant.
        The data represents a supply chain planning dataset, capturing master data (products, locations, transportation, production sources) and transactional data (weekly demand, stock, capacity, and flows).
        It is used to analyze inventory health (understock/overstock), production, and demand fulfillment across plants, RDCs, DCs, and vendors.
        Your task is to take the current user query and the past conversation history,
        and refine/modify the query to make it self-contained and more precise.
        Conversation history: {conversation_memory}
        Past historical chat from DB: {past_history}
        Current query: {query}
        Refined query (only return the refined query, no explanations):
"""

        response = llm_model.invoke([HumanMessage(content=refinement_prompt)])
        refined_query = response.content.strip()

        return refined_query

    except Exception as e:
        return f"[Query Refinement Failed: {str(e)}]"


## save_conversation_to_db for long term memory usage:

def save_conversation_to_db(engine, schema_name, UserID, ChatID, UserQuery, LLMResponse):
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
        Column("ChatID", String(255)),
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
        "ChatID": ChatID,
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




# --- Graph Plotting ---
@traceable
def plot_graph_from_query(df, user_query, explanation, llm, filename="chart.png"):
    try:
        # Skip if less than 2 columns
        if df is None or df.shape[0] <= 2 or df.shape[1] <= 2:
            print("Insufficient columns for plotting.")
            return ""

        trimmed_df = False
        if len(df) <= 10:
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

- Add data labels ** on the peaks (local maxima) and lows (local minima) and at all points** to highlight important points. Avoid cluttering by skipping other labels.
- Include gridlines to improve readability.
- Save with: plt.savefig("{filename}", facecolor=fig.get_facecolor())
- Close with: plt.close(fig)

Output only Python code. If no valid chart, return exactly: No graph generated
"""
 
        try:
            code = llm.invoke(prompt)
        except BaseException as be:
            print(f"Fatal error from LLM call: {be}")
            return ""
        
        if "No graph generated" in str(code):
            return ""

        # Clean LLM code output
        code = str(code.content if hasattr(code, 'content') else code)
        code = re.sub(r"```python", "", code)
        code = re.sub(r"```", "", code)

        if trimmed_df:
            code = re.sub(r"(?s)data\s*=\s*\[.*?\]", "", code)
            code = re.sub(r"df\s*=\s*pd\.DataFrame\(.*?\)", "", code)
            local_vars = {}
            if 'df' in code:
                local_vars = {"df": df, "plt": plt, "pd": pd}
            elif 'data' in code:
                local_vars = {"data": df, "plt": plt, "pd": pd}
            exec(code, {}, local_vars)
        else:
            exec(code)

        with open(filename, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    except Exception as e:
        print(f"Error generating graph: {e}")
        encoded_image = ""

    return encoded_image



## Convert Text to Tabular Data:

def extract_table_from_text(query, response, llm):
    """
    Converts textual response into structured tabular data using an LLM.
    
    Parameters:
        query (str): The original user query.
        response (str): The textual response from the assistant.
        llm: The language model object (e.g., LangChain or OpenAI wrapper with .invoke()).
    
    Returns:
        pd.DataFrame: The extracted table as a DataFrame (empty if failed).
    """
    prompt = f"""
You are a helpful assistant that extracts structured tabular data from natural language.

Given the following information:
- User Query: {query}
- Textual Response: {response}

Your task is to extract the relevant data and represent it as a clean tabular dataset in Python using pandas.

Output Python code that:
1. Uses pandas to construct the DataFrame.
2. Does NOT include explanations or markdown – just valid Python code.
3. Only returns code that defines a `df` variable with the structured data.

If no tabular data is present, return exactly: No table generated
"""

    try:
        code = llm.invoke(prompt)
    except BaseException as be:
        print(f"Fatal error from LLM call: {be}")
        return pd.DataFrame()

    # Clean LLM response
    code = str(code.content if hasattr(code, 'content') else code)
    
    if "No table generated" in code:
        return pd.DataFrame()

    code = re.sub(r"```python", "", code)
    code = re.sub(r"```", "", code)

    # Execute the code and extract DataFrame
    local_vars = {"pd": pd}
    try:
        exec(code, {}, local_vars)
        df = local_vars.get("df", pd.DataFrame())
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Returned object is not a DataFrame")
    except Exception as e:
        print(f"Error parsing table from LLM code: {e}")
        df = pd.DataFrame()

    return df



# --- Agents ---
@tool
def reasoning_pipeline_tool(user_input: str):
    """Runs the reasoning pipeline and generate reasons and returns results."""
    print("-"*100)
    print("resoning agent invoked")
    return flow_query_to_alert(user_input)

reasoning_agent = create_react_agent(
    model=llm_model,
    tools=[reasoning_pipeline_tool],
    name="reasoning_agent",
    prompt="You are an expert Inventory Risk Management Reasoning Agent. Your task is to generate the reasons for overstock and understock instance occurance"
)

@tool
def information_retrieval_tool(user_input: str):
    """Runs the information retrival pipeline and returns results."""
    print("-"*100)
    print("information agent invoked")
    return get_response(user_input)

information_retrieval_agent = create_react_agent(
    model=llm_model,
    tools=[information_retrieval_tool],
    name="information_retrieval_agent",
    prompt="""
    You are an information retrieval agent.
        - use the `information_retrieval_tool` to generate response.
        - Pass the user’s query exactly as it was written.
        - Do NOT rewrite, translate, or convert the query into SQL.
        - Simply forward the original query string unchanged.
        """
    
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
def run_supervisor(query, conversation_memory, chat_type, past_history):
    try:
        ## sessions memory:
        add_to_memory("user", query)

        ## query refiner:
        query = refine_user_query(query, chat_type, conversation_memory, past_history)
        print('Modified Query :', query)

        response = supervisor.invoke({"messages": [{"role": "user", "content": query}]})
        explanation, data_preview = "", ""
        df = pd.DataFrame()
        delegated = False
        messages = response.get("messages", [])

        final_ans = ""
        for message in messages:

            if "info_retrival_ans" in message.content:
                delegated = True
                ans = message.content
                ans = json.loads(ans)
                explanation = ans['info_retrival_ans']
                print("-----------explanation-------------")
                print(explanation)

            if "Chain of thought" in message.content:
                ans = message.content
                delegated = True
                repharse_ans_prompt = f"""
                You are supply chain assistant that answer user query specifcally related to the reasoning behind understock or overstock instances occuring at location for products.
                Your provide answer to user query based on instance alert that is provided to you. The alerts provided to you are given by reasoning module who insvestigates reasoning behind the warning instances.
                The alert(s) are the list of alert instances detected by the reasoning module. Each instance contains reason(s) for the occurence of alert instance. 
                Use data in Alert details to provide a natural and friendly response to answer the user query.

                In you response these instructions must be followed:
                - Initially give the user a one line brief at what and how many alert instances are detected based on user query mentioned product, location and time (if provided by user)
                - Then mention each warning instance with 2-3 line summary of reasons associated with that instance which is backed by data evidence from chain of thought section.
                - give your response in natural language
                - When the response involves multiple records, present it as a Markdown table using | separators, with headers in the first row.
         
                user query: {query}
                
                alert details: {ans}
                
                """
                #print(ans)
                repharsed_ans = llm_model.invoke(repharse_ans_prompt)
                explanation = repharsed_ans.content

            # response = message.content
                
            # --- Fallback Policy ---
        if not delegated:
            explanation = ""
            for step in information_retrieval_agent.stream(
                {"messages": [{"role": "user", "content": query}]},
                stream_mode="values",
            ):
                # step["messages"][-1].pretty_print()
                explanation = step["messages"][-1].content
    
            
        # convetring dataframe to json for output payload. The json is then displayed in tabluar format on UI
        data = df.head(15)
        data = data.fillna("")
        # Convert DataFrame to JSON format as per your requirement
        json_result = data.to_dict(orient='index')

        #log_query_and_response(engine, query, explanation)

        # Adding row labels to match the required JSON structure
        json_result = {f'row{index + 1}': row for index, row in json_result.items()}

        ## Extract table from text response:
        tabular_df = extract_table_from_text(query, explanation, llm_model)
        tabular_df = tabular_df.fillna(0)

        encoded_graph = plot_graph_from_query(tabular_df, query, explanation, llm_model)# if df is not None else ""
        if "```json" in encoded_graph:
            encoded_graph = ""

        ## Sessions memory:
        add_to_memory("ai", explanation)

        return explanation, encoded_graph, json_result, tabular_df
    except:
        explanation = "I wasn’t able to generate a response at the moment. Please try again."
        data = pd.DataFrame()
        tabular_df = pd.DataFrame()
        json_result = data.to_dict(orient='index')
        add_to_memory("ai", explanation)
        return explanation, "", json_result, tabular_df



# @app.route("/handle_query", methods=["POST"])
@app.route("/handle_query", methods=["POST", "OPTIONS"])
def handle_query():

    global chat_type

    if request.method == "OPTIONS":
        # Return 204 No Content (standard for CORS preflight)
        return '', 204

    input_data = request.get_json()

    query = input_data.get("query")
    UserID = input_data.get("UserID")
    ChatID = input_data.get("ChatID")
    chat_type = input_data.get("chat_type")
    past_history = input_data.get("past_history")

       
    if not query:
        return jsonify({"error": "Missing 'query'"}), 400

    conversation_memory = []
    # --- Reset memory if new chat ---
    if chat_type == "new_chat":
        reset_memory()
        response, graph, json_result, table_df = run_supervisor(query, conversation_memory, chat_type, past_history)

    # --- Run supervisor without resetting ---
    elif chat_type == "same_chat":
        response, graph, json_result, table_df = run_supervisor(query, conversation_memory, chat_type, past_history)

    # --- Run supervisor without resetting for past chat history---
    elif chat_type == "past_chat":
        response, graph, json_result, table_df = run_supervisor(query, conversation_memory, chat_type, past_history)

    else:
        return jsonify({"error": f"Invalid chat_type '{chat_type}'. Expected 'new_chat' or 'same_chat'."}), 400

    ## Save to DB
    save_conversation_to_db(engine, schema_name, UserID, ChatID, query, str(response))

    # return jsonify([{"response":response}, {"graph_base64":graph}, {"json_data":json_result}])

    return jsonify([
        {"response": response},
        {"graph_base64": graph},
        {"json_data" : json_result},
        {"Tabular_Data": table_df.to_dict(orient='records')}
        ])

@app.route("/health", methods=["GET"])
def health():
    return 'OK'


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)

