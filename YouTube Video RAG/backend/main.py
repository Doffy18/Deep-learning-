from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ai import router as ai_router


app = FastAPI()
app.add_middleware(
    CORSMiddleware, 
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"]
)

app.include_router(ai_router)

