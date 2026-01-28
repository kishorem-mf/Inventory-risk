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

# --- Flask setup ---
app = Flask(__name__)
# Enable CORS for all routes and origins
CORS(app)
port = int(os.environ.get('PORT', 8080))


# --- Environment setup ---
# These should be set via environment variables before running the application
# Required: AICORE_AUTH_URL, AICORE_CLIENT_ID, AICORE_CLIENT_SECRET, AICORE_BASE_URL,
#           AICORE_RESOURCE_GROUP, LANGSMITH_API_KEY, HANA_HOST, HANA_PASSWORD

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "invt_analysis_prod"

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
def run_supervisor(query):
    #refined_query = refine_query_with_history(llm_model, query)
    try:
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
                Use data in Alert details to provide a natural and friendly response to answer the user query.
                In you response these instructions must be followed:
                - Initially give the user a one line sneak peek at what and how many instances are detected based on user provided product, location and time (if provided by user)
                - Then mention each warning instance with 2-3 line summary of reasons associated with the instance which is backed by data evidence from chain of thought section
                - give your response in natural language
                
                query: {query}
                
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

        encoded_graph = plot_graph_from_query(df, query,explanation, llm_model) if df is not None else ""
        if "```json" in encoded_graph:
            encoded_graph = ""
        return explanation, encoded_graph, json_result
    except:
        explanation = "I wasn’t able to generate a response at the moment. Please try again."
        data = pd.DataFrame()
        json_result = data.to_dict(orient='index')
        return explanation,"",json_result



@app.route("/handle_query", methods=["POST"])
def handle_query():
    input_data = request.get_json()
    query = input_data.get("query")
    response, graph, json_result = run_supervisor(query)
    return jsonify(
        {"response": response},
        {"graph_base64": graph},
        {"json_data" : json_result}
    )

if __name__ == "__main__":
  # Cloud Foundry provides PORT env
    app.run(host='0.0.0.0', port=8080,debug=True)
