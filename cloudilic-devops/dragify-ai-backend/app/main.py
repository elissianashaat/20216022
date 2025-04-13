from fastapi import FastAPI
from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware
import uvicorn
from app.routes import router
import sys


# Import other routers as needed
import os
print("Current working directory:", os.getcwd())
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY"))

# Include the router
app.include_router(router)



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)