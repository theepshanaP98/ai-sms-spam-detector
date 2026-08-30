from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.model import MODEL_PATH, predict_message

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="AI SMS Spam Detector",
    description="NLP application that classifies SMS messages as spam or ham.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"model_ready": MODEL_PATH.exists()},
    )


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "AI SMS Spam Detector",
        "model_ready": MODEL_PATH.exists(),
    }


@app.post("/api/predict")
def predict(payload: MessageRequest):
    try:
        return predict_message(payload.message)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
