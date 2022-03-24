from fastapi import FastAPI
from .controllers import controller

app = FastAPI()

app.include_router(controller.router)
