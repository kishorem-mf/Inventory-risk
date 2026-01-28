from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, EmailStr
from hdbcli import dbapi
import bcrypt, jwt, os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from cfenv import AppEnv
import uuid
from datetime import datetime, timedelta, timezone

load_dotenv()

app = FastAPI(
    title="App for user registration and login",
    version="1.0.0",
    description="FastAPI based app for user registration and login"
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

# secret key (use environment variable in production)
SECRET_KEY = uaa_service["clientsecret"]


# DB connection factory
def get_connection():
    return dbapi.connect(
        address=os.environ["HANA_HOST"],
        port=os.environ["HANA_PORT"], 
        user=os.environ["HANA_USER"], 
        password=os.environ["HANA_PWD"]
    )

# Pydantic models for request validation
class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    username_or_email: str
    password: str

class UpdatePasswordRequest(BaseModel):
    new_password: str

# Register Endpoint
@app.post("/register")
def register(request: RegisterRequest):
    username = request.username.strip().lower()
    email = request.email.strip().lower()
    password = request.password

    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # check if username or email already exists
        cursor.execute("SELECT USER_ID FROM INVT_XAI_RAW.USERS WHERE USERNAME = ? OR EMAIL = ?", (username, email))
        if cursor.fetchone():
            raise HTTPException(status_code=409, detail="Username or Email already exists")
        
        user_id = str(uuid.uuid4())

        # insert new user
        cursor.execute(
            "INSERT INTO INVT_XAI_RAW.USERS (USER_ID, USERNAME, EMAIL, PASSWORD_HASH) VALUES (?, ?, ?, ?)",
            (user_id, username, email, hashed_pw)
        )
        conn.commit()
        cursor.close()
        conn.close()

        return {"message": "User registered successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Login Endpoint
@app.post("/login")
def login(request: LoginRequest):
    key = request.username_or_email.strip().lower()
    password = request.password

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT USER_ID, USERNAME, PASSWORD_HASH FROM INVT_XAI_RAW.USERS WHERE USERNAME = ? OR EMAIL = ?", (key, key))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        user_id, username, stored_hash = row

        if bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8")):
            payload = {
                "user_id": str(user_id),
                "username": username,
                "exp": datetime.now(timezone.utc) + timedelta(hours=1)
            }
            token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
            return {"message": "Login successful", "token": token}

        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Utility: verify JWT token
def get_current_user(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token header")
    token = authorization.split(" ")[1]

    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return decoded  # contains user_id, username, exp
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Update Password Endpoint
@app.post("/update-password")
def update_password(req: UpdatePasswordRequest, current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    new_password = req.new_password.strip()

    if not new_password:
        raise HTTPException(status_code=400, detail="Password cannot be empty")

    # Hash new password
    hashed_pw = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE INVT_XAI_RAW.USERS
            SET PASSWORD_HASH = ?, UPDATED_AT = CURRENT_UTCTIMESTAMP
            WHERE USER_ID = ?
            """,
            (hashed_pw, user_id)
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")

        conn.commit()
        cursor.close()
        conn.close()

        return {"message": "Password updated successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
