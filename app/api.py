from fastapi import FastAPI, File, UploadFile, HTTPException
import asyncio, logging

from .models import TaskRequest, AnswerPayload
from .pipeline import make_plan_with_gemini, run_plan_with_gpt_tools

app = FastAPI(title="TDS Data Analyst Agent")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/", response_model=AnswerPayload)
async def analyze(file: UploadFile = File(...)):
    try:
        task = (await file.read()).decode()
        # 1️⃣ optional: persist initial task here
        plan = await make_plan_with_gemini(task)
        logging.info("Gemini plan:\n%s", plan)
        # 2️⃣ run full pipeline
        payload = await run_plan_with_gpt_tools(task)
        return payload
    except Exception as exc:
        logging.exception("analysis failure")
        raise HTTPException(status_code=500, detail=str(exc))
