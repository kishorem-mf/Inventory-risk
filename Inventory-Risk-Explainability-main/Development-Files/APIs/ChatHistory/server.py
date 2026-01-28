import os
import json
from dotenv import load_dotenv
from hdbcli import dbapi
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from cfenv import AppEnv
from sap import xssec
import requests

load_dotenv()

hana_conn = dbapi.connect(
    address=os.environ["HANA_HOST"],
    port=os.environ["HANA_PORT"], 
    user=os.environ["HANA_USER"], 
    password=os.environ["HANA_PWD"]
    #databasename=os.environ["HANA_DB_NAME"]
)

app = FastAPI(
    title="App to get chat history",
    version="1.0.0",
    description="FastAPI based app for getting chat history"
)

origins = [
    "http://localhost:4200",  # Frontend URL , local host 
    "*",  # Allows all origins (not recommended for production)
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # allow methods
    allow_headers=["*"], # allow headers
)

env = AppEnv()

uaa_service = env.get_service(name='gen-ai-xsuaa-service').credentials


@app.get("/")
def root(request:Request):
    print("Request header:")
    print(request.headers)
    error_raised = False
    isAuthorized = False
    try:
        access_token = request.headers.get('authorization')[7:]
        security_context = xssec.create_security_context(access_token, uaa_service)
        isAuthorized = security_context.check_scope('uaa.resource')
    except:
        error_raised = True
    
    if error_raised:
        raise HTTPException(status_code=403, detail="Not Authorized")

    
    if not isAuthorized:
        raise HTTPException(status_code=403, detail="Not Authorized")

    return {"message": "Welcome to chat history app"}

@app.get("/getToken")
def generate_access_token(request:Request):
    try:
        url = uaa_service["url"] + '/oauth/token'
        payload = {
            "grant_type" : "client_credentials",
            "client_id": uaa_service["clientid"],
            "client_secret": uaa_service["clientsecret"]
        }
        headers = {
            "Content-Type" : "application/x-www-form-urlencoded"
        }
        response = requests.post(url, data=payload, headers=headers)
        return response.json()["access_token"]
    except Exception as e:
        response = f"Error generating access token: {e}"
        return response

@app.get("/getChatHistory")
def get_chat_history(request:Request, 
                      userID: Optional[str] = Query(None, description="User ID"),
                      chatID: Optional[str] = Query(None, description="Chat ID")):
    error_raised = False
    isAuthorized = False
    try:
        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Missing Bearer token")
        
        access_token = auth_header[7:]     

        security_context = xssec.create_security_context(access_token, uaa_service)
        isAuthorized = security_context.check_scope('uaa.resource')
    except:
        error_raised = True
    
    if error_raised:
        raise HTTPException(status_code=403, detail="Not Authorized")

    
    if not isAuthorized:
        raise HTTPException(status_code=403, detail="Not Authorized")

    cursor = hana_conn.cursor()
    # Build query dynamically based on filters
    sql_command = "SELECT * FROM CURRENT_INVT.QUERY_HISTORY"
    conditions = []
    params = []

    if userID:
        conditions.append('"UserID" = ?')
        params.append(userID)
    if chatID:
        conditions.append('"ChatID" = ?')
        params.append(chatID)

    if conditions:
        sql_command += " WHERE " + " AND ".join(conditions)

    print(f"Executing SQL: {sql_command} with params {params}")
    cursor.execute(sql_command, tuple(params))
    matching_rows = cursor.fetchall()
    cursor.close()
    print("Results:")
    print(matching_rows)

    keys = ["UserID", "ChatID", "QueryDateTime", "UserQuery", 
    "LLMResponse"]
    
    # Convert list of tuples to list of dictionaries
    list_of_dicts = [dict(zip(keys, t)) for t in matching_rows]
    for x in list_of_dicts:
        date_str = x["QueryDateTime"].strftime('%Y-%m-%d %H:%M:%S')
        x["QueryDateTime"] = date_str

    # Convert list of dictionaries to JSON string
    #json_data = json.dumps(list_of_dicts, indent=4)
    return {
        "status_code" : 200,
        "results": list_of_dicts
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
