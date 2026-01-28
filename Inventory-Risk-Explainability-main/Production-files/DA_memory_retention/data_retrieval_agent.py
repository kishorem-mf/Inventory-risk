# import libraries
import os
from gen_ai_hub.proxy.langchain.init_models import init_llm
from sqlalchemy import create_engine,inspect,text
from urllib.parse import quote_plus
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langgraph.prebuilt import create_react_agent
import json

# define llm
os.environ["AICORE_AUTH_URL"] = "https://btp-ai-developments-sl2f9ys4.authentication.eu10.hana.ondemand.com"
os.environ["AICORE_CLIENT_ID"] = "sb-38176009-b499-470f-a3b8-9cf98daac1d0!b503699|aicore!b540"
os.environ["AICORE_CLIENT_SECRET"] = "1ac5c77f-d5ac-4e2d-8c19-6ffc47113ec8$52U4q9NYAN-GBm23a2lm_SFVrzmWNhuS7l_qFXs4s4A="
os.environ["AICORE_BASE_URL"] = "https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com"
os.environ["AICORE_RESOURCE_GROUP"]="default"

llm_model = init_llm('gpt-4o', max_tokens=16000, temperature = 0)

# connect to db
host = "cfe32093-429a-4e59-87dc-9f3e4da891bf.hna2.prod-eu10.hanacloud.ondemand.com"
port = "443"
schema_name = "CURRENT_INVT"
user = "DBADMIN"
password = "Bcone@1234567"
user_enc = quote_plus(user)
password_enc = quote_plus(password)
connection_str = f"hana://{user_enc}:{password_enc}@{host}:{port}/?currentSchema={schema_name}"
engine = create_engine(connection_str)
sql_database = SQLDatabase(engine,schema=schema_name)

toolkit = SQLDatabaseToolkit(db=sql_database, llm=llm_model)

tools = toolkit.get_tools()

# metadata = table defination, table schema and relations
with open('data_schema.txt', 'r') as file:
    data_schema = file.read()
 
with open('Data_description.txt', 'r') as file:
    data_description = file.read()


meta_data = f"""
            Data schema: {data_schema}.

            Data description: {data_description}
            """

            
# domain knowledge

# persona, tone and response format

# NL-SQL conversion - prompt input: metadata, SQL generation prompt + domain knowledge
nl_sql_prompt = f"""
            You are an agent designed to interact with a SQL database.
            Given an input question, create a syntactically correct hana query to run.
            then look at the results of the query and return the answer.
            {meta_data}            

            You MUST double check your query before executing it. If you get an error while
            executing a query, rewrite the query and try again.

            To start you should ALWAYS look at the tables in the database to see what you
            can query. Do NOT skip this step.
            Make sure to qualify column names with their respective table names.
            Never invent or assume table or column names. If the schema does not define a specific field, do not reference it.
            When the user query is about overstock or understock instance, first fetch the instance details from stock_Status_v2 table then get the further details by quering the transcation tables (Review DC, Review Plant, etc).
            In other cases query the master table and transaction table
            Then you should query the schema of the most relevant tables. Give only the valid sql query. No additional text
            DO NOT mention about column names in the header.
            Expand week ranges into individual week fields in SQL
            Always interpret “today”, “current”, “as of now”, or “from current date” as the database function CURRENT_DATE.
            DO NOT return the sql query as response. 
            If no location/product in specified teh response should contain data about all locations/products.
            Provide only the final business result in natural language, including computed values, summaries, and insights. 
            Focus on answering the user’s question directly (e.g., totals, trends, or comparisons) without exposing technical implementation details.
            """



agent = create_react_agent(
    llm_model,
    tools,
    prompt=nl_sql_prompt,
)

def get_response(query):

    try:

        response_text = ""
        for step in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            step["messages"][-1].pretty_print()
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
            "info_retrival_ans": "Could not generate response please, try again"
        },
        default=str # pretty-print
    )
