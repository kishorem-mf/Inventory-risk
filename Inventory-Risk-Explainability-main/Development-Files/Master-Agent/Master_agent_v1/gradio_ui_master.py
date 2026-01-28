import os
import gradio as gr
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine,inspect,text
from urllib.parse import quote_plus
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from information_retival_pipeline import execute_pipeline
from reasoning_agent_pipeline import reasoning_execution_pipeline
from langgraph_supervisor import create_supervisor
from langchain_tavily import TavilySearch
import pandas as pd
import re
import base64
import json
from io import BytesIO
from PIL import Image

# --- Environment setup ---
os.environ["AICORE_AUTH_URL"] = "https://btp-ai-developments-sl2f9ys4.authentication.eu10.hana.ondemand.com"
os.environ["AICORE_CLIENT_ID"] = "sb-38176009-b499-470f-a3b8-9cf98daac1d0!b503699|aicore!b540"
os.environ["AICORE_CLIENT_SECRET"] = "1ac5c77f-d5ac-4e2d-8c19-6ffc47113ec8$52U4q9NYAN-GBm23a2lm_SFVrzmWNhuS7l_qFXs4s4A="
os.environ["AICORE_BASE_URL"] = "https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com"
os.environ["AICORE_RESOURCE_GROUP"] = "default"
os.environ["TAVILY_API_KEY"] = "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"

# --- DB connection ---
host = "cfe32093-429a-4e59-87dc-9f3e4da891bf.hna2.prod-eu10.hanacloud.ondemand.com"
port = "443"
schema_name = "CURRENT_INVT"
user = "DBADMIN"
password = "Bcone@1234567"
user_enc = quote_plus(user)
password_enc = quote_plus(password)
connection_str = f"hana://{user_enc}:{password_enc}@{host}:{port}/?currentSchema={schema_name}"
engine = create_engine(connection_str)

# --- Model ---
llm_model = init_llm('gpt-4o', max_tokens=10000, temperature=0)


# ---- Query refiner ---

def refine_query_with_history(llm, user_query, max_prompt_length=3000):
    """
    Refines the current user query using recent query history to add context.
    """
    # Combine recent queries until reaching the max_prompt_length (character count for simplicity)

    query = text("""
        SELECT * FROM "INVT_XAI_RAW"."QUERY_HISTORY"
        ORDER BY "QueryDateTime" DESC LIMIT 5
    """)
    with engine.connect() as conn:
        result = conn.execute(query)
        history_combined = result.fetchall()  


    # history_combined = ""
    # for i, row in recent_queries_df.iterrows():
    #     line = f"Previous Query {i+1}: {row['UserQuery']}\n"
    #     if len(history_combined + line) > max_prompt_length:
    #         break
    #     history_combined += line

    prompt = f"""
    Refine the user's query using recent chat history. Follow these strict rules:

    1. Analyze if the user query is incomplete or unclear. If so, add only the missing generic details 
       (like week, year, product ID, or metric names).
    2. If the question depends on a prior query, align it with the entities/IDs used before.
    3. DO NOT under any circumstances invent, remove, or modify location IDs (DC, PL, RDC). 
       If a location is specified, preserve it exactly as given.
    4. If no location is mentioned, do not add one.
    5. Do not include any additional commentary or explanations in the output.
    
    Here is the recent query history:
    {history_combined}

    User query:
    {user_query}
    
    """

    try:
        response = llm.invoke(prompt)
        refined_query = response.content.strip()
        return refined_query
    except Exception as e:
        print(f"Error refining query from history: {e}")
        return user_query

# --- Graph module ---
import json
import matplotlib.pyplot as plt
import base64
import io

##############
# GRAPH MODULE
##############

def plot_graph_from_query(df, user_query, llm, filename="chart.png"):
    encoded_image = ""
    try:
        # Step 1: Provide schema and sample data to LLM
        df_schema = {
            "columns": list(df.columns),
            "sample_data": df.head(3).to_dict(orient="records")
        }

        prompt = f"""
        You are a data visualization assistant.
        The user wants a chart for their query.

        User query: "{user_query}"

        DataFrame schema and sample values:
        {json.dumps(df_schema, indent=2)}

        Based on the query and available columns, return the BEST chart configuration in JSON:
        - chart_type: one of ["line", "bar", "pie", "scatter", "histogram"]
        - x_axis: column name for x-axis (if applicable)
        - y_axis: column name for y-axis (if applicable, null if not needed)
        - title: title of the chart
        """

        llm_response = llm.invoke(prompt).content

        # Step 2: Clean LLM JSON output
        cleaned_response = llm_response.strip()
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response.strip("`")
            if cleaned_response.lower().startswith("json"):
                cleaned_response = cleaned_response[4:].strip()

        # Step 3: Parse JSON
        chart_details = json.loads(cleaned_response)

        chart_type = chart_details["chart_type"].lower()
        x_axis = chart_details.get("x_axis")
        y_axis = chart_details.get("y_axis")
        title = chart_details.get("title", "Generated Chart")

        # Step 4: Plot
        plt.figure(figsize=(6, 4))

        if chart_type == "bar":
            plt.bar(df[x_axis], df[y_axis])
        elif chart_type == "line":
            plt.plot(df[x_axis], df[y_axis], marker='o')
        elif chart_type == "pie":
            plt.pie(df[y_axis], labels=df[x_axis], autopct='%1.1f%%')
        elif chart_type == "scatter":
            plt.scatter(df[x_axis], df[y_axis])
        elif chart_type == "histogram":
            plt.hist(df[x_axis], bins=10)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        plt.xlabel(x_axis if x_axis else "")
        plt.ylabel(y_axis if y_axis else "")
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save to file
        plt.savefig(filename, dpi=300, bbox_inches="tight")

        # Convert to base64 encoded string
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

        # Optional: also display
        plt.show()

    except Exception as e:
        print(f"Error generating graph: {e}")

    return encoded_image



# --- Agents ---
@tool
def reasoning_pipeline_tool(user_input: str) -> dict:
    """Runs the reasoning pipeline and returns results."""
    print("-"*100)
    print("resoning agent invoked")
    return reasoning_execution_pipeline(user_input)

reasoning_agent = create_react_agent(
    model=llm_model,
    tools=[reasoning_pipeline_tool],
    name="reasoning_agent",
    prompt="You are an expert Inventory Risk Management Reasoning Agent."
)

@tool
def sql_pipeline_tool(user_input: str) -> dict:
    """Runs the SQL pipeline and returns results."""
    print("-"*100)
    print("sql agent invoked")
    return execute_pipeline(user_input)

sql_agent = create_react_agent(
    model=llm_model,
    tools=[sql_pipeline_tool],
    name="sql_agent",
    prompt="You are an SQL processing agent. Generate queries, execute via the SQL tool, and return results."
)


web_search = TavilySearch(max_results=3)

research_agent = create_react_agent(
    model=llm_model,
    tools=[web_search],
    prompt="You are a research agent.",
    name="research_agent"
)

# --- Supervisor ---
supervisor = create_supervisor(
    model=llm_model,
    agents=[reasoning_agent, sql_agent],
        prompt="""You are the Supervisor Agent. 
                You oversee and delegate tasks between two specialized agents:

                1. Reasoning Agent  
                - Handles analytical, logical, and risk management reasoning tasks.  
                - Use this agent when the task requires reasoning. The user query is about why overstock or understock is occuring  

                2. SQL Agent  
                - Handles database-related tasks such as generating queries, retrieving structured data, and executing SQL.  
                - Use this agent whenever information retrieval from the database is required.  

                ### Rules for Delegation:
                - Assign tasks to only one agent at a time. Do not call multiple agents in parallel.  
                - Always select the agent that is best suited for the current user request.  
                - Do not attempt to solve or execute tasks yourself; your role is purely delegation and coordination.  
                - After the assigned agent completes its work, review the response and determine whether:  
                - The task is complete and the final result can be returned, or  
                - Additional work must be handed off to the same or another agent.  

                Your objective is to ensure tasks are completed efficiently by leveraging the agents’ expertise without doing the work directly yourself.
                """,
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()

# --- Function to run supervisor ---
def run_supervisor(query):

    refined_query = refine_query_with_history(llm_model, query, max_prompt_length=3000)

    response = supervisor.invoke({
        "messages": [
            {"role": "user", "content": refined_query}
        ]
    })

    explanation = ""
    data_preview = ""
    content = ""
    df = pd.DataFrame()
    messages = response.get("messages", [])
    for message in messages:
        
        if 'sql_query' in message.content:
            content = json.loads(message.content)
            sql_query = content["sql_query"]
            data_preview = content["data_preview"]
            df = pd.DataFrame(data_preview)
            explanation = content["explanation"]
    # convetring dataframe to json for output payload. The json is then displayed in tabluar format on UI
    data = df.head(5)
    # Convert DataFrame to JSON format as per your requirement
    json_result = data.to_dict(orient='index')

    # Adding row labels to match the required JSON structure
    json_result = {f'row{index + 1}': row for index, row in json_result.items()}

    encoded_graph = plot_graph_from_query(df, query, llm_model, filename="chart.png")

    return refined_query, explanation, encoded_graph, json_result

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
