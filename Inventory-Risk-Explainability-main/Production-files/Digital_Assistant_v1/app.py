import os
from flask import Flask, request, jsonify
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine,inspect,text
from langchain.chains import create_sql_query_chain
import pandas as pd
import re
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from flask_cors import CORS
import json
import base64


import platform
from hdbcli import dbapi

from langchain_core.language_models import LLM
from langchain_core.outputs import Generation, LLMResult
from typing import List, Optional, Any
from gen_ai_hub.proxy.native.openai import chat
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_community.agent_toolkits import SQLDatabaseToolkit


from sqlalchemy import create_engine, inspect
from sqlalchemy.engine.url import URL
from langchain_community.utilities import SQLDatabase
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent




os.environ["AICORE_AUTH_URL"] = "https://btp-ai-developments-sl2f9ys4.authentication.eu10.hana.ondemand.com"
os.environ["AICORE_CLIENT_ID"] = "sb-38176009-b499-470f-a3b8-9cf98daac1d0!b503699|aicore!b540"
os.environ["AICORE_CLIENT_SECRET"] = "1ac5c77f-d5ac-4e2d-8c19-6ffc47113ec8$52U4q9NYAN-GBm23a2lm_SFVrzmWNhuS7l_qFXs4s4A="
os.environ["AICORE_BASE_URL"] = "https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com"
os.environ["AICORE_RESOURCE_GROUP"]="default"
###### connecting to LLM #############
llm = init_llm('gpt-4', max_tokens=4096, temperature = 0)


##### DB connection
schema_name = "INVTRISK"

# Build the connection URL manually
connection_url = URL.create(
    drivername="hana",  # NOT hana+hana
    username="DBADMIN",
    password="Bcone@1234567",
    host="cfe32093-429a-4e59-87dc-9f3e4da891bf.hna2.prod-eu10.hanacloud.ondemand.com",
    port=443,
    query={
        "encrypt": "true",
        "sslValidateCertificate": "false",
        "currentSchema": schema_name
    }
)

# Create engine
engine = create_engine(connection_url)

# Inspect
inspector = inspect(engine)
tables = inspector.get_table_names(schema=schema_name)

# Get the db variable as before
db = SQLDatabase(engine=engine, schema=schema_name, include_tables=tables)

toolkit = SQLDatabaseToolkit(db=db, llm=llm, top_k = 100)
tools = toolkit.get_tools()


################## Define and create app ##############
app = Flask(__name__)
# Enable CORS for all routes and origins
CORS(app)
port = int(os.environ.get('PORT', 8001))



@app.route('/langraph_agent', methods=['POST'])
def langraph_agent():

    try:
        input_data = request.get_json()
        query = input_data.get('query')  


        table_name = 'stock_status'
        data_context = f'''
        You are working with a supply chain planning table named **{table_name}**
        This table tracks weekly stock status for various products at different plants and distribution centers.
        
        Here are the column definitions :
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
        While responding with instance occurrence, always specify its location and the period range of instance(Always use week_end_date for precise instance period caluclation ) 
        and don't mention instances verbatum (like overstock_instance_1) , mention it in more natural laguage format.

        To provide the reason for overstock instance follow this instruction:
        Go step by step :
        Step 1. First look at that individual instance of overstock
        Step 2. Find the 'stock on hand' value for the first week of the instance period i.e. the first week of the instance
        Step 3. If 'stock on hand' value exists then check if the value is greater than 'safety stock' value for the same week (i.e. first week of the instance)
        Step 4. If the check is true then respond that the cause of overstock is because of high amount of stock on hand at the start of the period
        Step 5. If 'stock on hand' value does not exist for the first week then respond that the overstock is because of supply chain mismanagement

        To provide the reason for understock instance follow this instruction and consider these variable values -
        Total demand for product FG-1000 is 48000, Total production of FG-1000 is 40000
        Step 1. Check whether demand is greater than the capcity
        Step 2. If yes, the reason of understock intance is production of the product not meeting the demand
        Step 3. If no, the reason of understock intance is because of supply chain mismanagement
        IMPORTANT INSTRUCTIONS:
        - Always use the **{table_name} ** table for all queries related to stock status.

        '''

        dialect="SQLite"
        top_k=100

        system_message = f"""
        You are an agent designed to interact with a SQL database.
        Also keep these things in mind :{data_context}
        Given an input question, create a syntactically correct {dialect} query to run,
        then look at the results of the query and return the answer. Unless the user
        specifies a specific number of examples they wish to obtain, always atleast provide limit your
        query to at most {top_k} results.

        You can order the results by a relevant column to return the most interesting
        examples in the database. Never query for all the columns from a specific table,
        only ask for the relevant columns given the question.

        You MUST double check your query before executing it. If you get an error while
        executing a query, rewrite the query and try again.
        While writing query keep in mind that query will be ran on SAP HANA SQL console, so adjust the query accordingly.

        SQL query instructions
        Make sure to qualify column names with their respective table names.
        Similarly, Make sure to qualify table names with their respective Schema names.
        Include all columns names in select statment which are present in join, where and order by clause
        This will ensure the relevant identifier is retrieved along with the other requested data.
        CHARINDEX doesn't work ,so instead use INSTR() is the correct function.
        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
        database.

        To start you should ALWAYS look at the tables in the database to see what you
        can query. Do NOT skip this step.

        Then you should query the schema of the most relevant tables.

    """
        agent_executor = create_react_agent(llm, tools, prompt=system_message)

        result = agent_executor.invoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        # The assistant’s final message is always the last one in the list
        response = result["messages"][-1].content

        #return response
        return jsonify({"response": f"{response}"})
    
    except Exception as e:
        return jsonify({"response": f"{e}"})



if __name__ == '__main__':
    # Run the app with host '0.0.0.0' to make it accessible
    app.run(host='0.0.0.0', port=port,debug=True)


