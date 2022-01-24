from fastapi import FastAPI
from .controllers import translation_controller

app = FastAPI()

app.include_router(translation_controller.router)
