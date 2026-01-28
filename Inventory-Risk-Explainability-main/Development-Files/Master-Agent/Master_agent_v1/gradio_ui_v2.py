import os
import gradio as gr
from gen_ai_hub.proxy.langchain.init_models import init_llm
from sqlalchemy import create_engine,inspect,text
from urllib.parse import quote_plus
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langgraph.prebuilt import create_react_agent
from reasoning_agent_pipeline import reasoning_execution_pipeline
from langgraph_supervisor import create_supervisor
import pandas as pd
import re
import base64
import json
from io import BytesIO
from PIL import Image
from datetime import datetime
from langsmith import traceable
from data_retrieval_agent import get_response
from reasoning_agent_v2 import flow_query_to_alert
# --- Graph module ---
import json
import matplotlib.pyplot as plt
import base64
import io

# --- Environment setup ---
# These should be set via environment variables before running the application
# Required: AICORE_AUTH_URL, AICORE_CLIENT_ID, AICORE_CLIENT_SECRET, AICORE_BASE_URL,
#           AICORE_RESOURCE_GROUP, TAVILY_API_KEY, LANGSMITH_API_KEY, HANA_HOST, HANA_PASSWORD

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "test_langsmith"

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
llm_model = init_llm('gpt-4o', max_tokens=16000, temperature=0)

toolkit = SQLDatabaseToolkit(db=sql_database, llm=llm_model)

information_retirval_tools = toolkit.get_tools()

# def refine_user_query(query, conversation_memory):
#     """
#     Refines the user query by considering the conversation history.
#     Uses the same LLM model for refinement.
#     """
#     try:
#         # conversation_memory = format_conversation_memory()

#         refinement_prompt = f"""
# You are a query refinement assistant.
# Your task is to take the current user query and the past conversation history,
# and refine/modify the query to make it self-contained and more precise.
# Conversation history:{conversation_memory}
# Current query: {query}
# Refined query (only return the refined query, no explanations):
# """

#         response = llm_model.invoke([HumanMessage(content=refinement_prompt)])
#         refined_query = response.content.strip()
#         # print(refined_query)
#         return refined_query

#     except Exception as e:
#         return f"[Query Refinement Failed: {str(e)}]"



##############
# GRAPH MODULE
##############
@traceable

def plot_graph_from_query(df, user_query, explanation, llm, filename="chart.png"):
    try:
        # to limit the size of dataframe given as context to llm. used trimmed_df to get subset of df containing all categories data point needed to generate graph. why to define labels and color schema you need all category names (location id, product id)
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

        Follow color schema:
        background: #FFFFFF (white)
        text/labels: #2B2C2E (black)
        markers/lines: #4BC884, #22A6BD, or red, yellow and any more colors for warnings
        Consider the categories for deciding the type of graph
        
        Add data labels if meaningful. Ensure they don't overlap
        Save with: plt.savefig("{filename}", facecolor=fig.get_facecolor())
        Close with: plt.close(fig)


        Output only Python code. If no valid chart, return exactly: No graph generated
        """

        code = llm.invoke(prompt)
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


# def log_query_and_response(engine, user_query, llm_response):
#     query = text("""
#                 INSERT INTO "INVT_XAI_RAW"."QUERY_HISTORY" ("QueryDateTime", "UserQuery", "LLMResponse")
#                 VALUES (:dt, :uq, :lr)
#                 """)
#     try:
#         with engine.begin() as conn:  # ensures automatic commit
#             conn.execute(query, {
#                 "dt": datetime.utcnow(),  # or use datetime.now()
#                 "uq": user_query,
#                 "lr": llm_response
#             })
#     except Exception as e:
#         print(f"[LOGGING ERROR] Failed to log query and response: {e}")



# --- Agents ---
@tool
def reasoning_pipeline_tool(user_input: str):
    """Runs the reasoning pipeline and generate reasons and returns results. Use this agent when the task requires reasoning. The user query is about why overstock or understock is occuring   """
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

# --- Function to run supervisor ---
@traceable
def run_supervisor(query):

    #refined_query = refine_query_with_history(llm_model, query, max_prompt_length=3000)

    response = supervisor.invoke({
        "messages": [
            {"role": "user", "content": query}
        ]
    })
    #print(response)
    explanation = ""
    data_preview = ""
    df = pd.DataFrame()

    delegated = False
    messages = response.get("messages", [])
    final_ans = ""
    info_retrival= False
    for message in messages:
        # print(message)
        if "info_retrival_ans" in message.content:
            delegated = True
            explanation = message.content
             
        if "Chain of thought" in message.content:
            ans = message.content
            delegated = True
            repharse_ans_prompt = f"""
            
            For given user query and alert information. Generate a summarized response for user.

            First list the number of alerts/instances identified. 
            Then generate the summary of based on chain of though reasoning provided. 
            follow this format:

            number of instances
            instance details 
            for each instance provide a consice summary on all reasons in 2-3 lines using chain of thought as data evidence behind the reasoning
            give your response in user friendly natural language

            query: {query}

            alert details: {ans}

            
            """
            #print(ans)
            repharsed_ans = llm_model.invoke(repharse_ans_prompt)
            explanation = repharsed_ans.content

    #     if "information_retrieval_agent" in message.content:
    #         print("----------------------------information_retrieval_agent -------------------")
    #         delegated = True
    #         info_retrival = True


    #     response = message.content

    # if delegated and info_retrival:
    #     explanation = response
    
    

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

    # Adding row labels to match the required JSON structure
    json_result = {f'row{index + 1}': row for index, row in json_result.items()}

    #log_query_and_response(engine, query, explanation)

    print("*"*80)
    print("response generated")
    print(explanation)


    encoded_graph = plot_graph_from_query(df, query,explanation, llm_model, filename="chart.png")

    return query, explanation, encoded_graph, json_result

def decode_base64_to_image(encoded_str):
    try:
        image_data = base64.b64decode(encoded_str)
        return Image.open(BytesIO(image_data))
    except Exception:
        return None

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("## Supervisor Agent Interface")
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(label="Enter your query")
            submit_btn = gr.Button("Run")
        with gr.Column():
            refined_query = gr.Textbox(label="Refined Query")
            response_output = gr.Textbox(label="Supervisor Response")
            json_result = gr.Textbox(label="Json Result")
            encoded_graph = gr.Textbox(label="Encoded Graph (Base64)")
            graph_image = gr.Image(label="Graph Image", type="pil")

    submit_btn.click(run_supervisor, inputs=query_input, outputs=[refined_query, response_output,encoded_graph, json_result])
    encoded_graph.change(
        fn=decode_base64_to_image,
        inputs=encoded_graph,
        outputs=graph_image
    )

if __name__ == "__main__":
    demo.launch()
