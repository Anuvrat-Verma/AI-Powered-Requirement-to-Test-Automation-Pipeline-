import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluator import selenium_evaluator
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil
from typing import List
from rag import ingest_documents, retrieve_context
from stt import transcribe_audio
from agents import agent1_generate_stories, agent2_generate_test_cases, agent3_generate_code
from schemas import GenerateRequest, GenerateResponse, TranscribeResponse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
app = FastAPI(title="AI Requirement-to-Test Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(audio: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await audio.read()
            tmp.write(content)
            temp_path = tmp.name

        transcribed_text = await transcribe_audio(temp_path)
        os.unlink(temp_path)  # cleanup

        return {"transcribed_text": transcribed_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Backend/main.py

@app.post("/generate")
def generate(request: GenerateRequest):
    try:
        # 1. BA and QA agents run once
        stories = agent1_generate_stories(request.requirements, request.use_rag)
        test_cases = agent2_generate_test_cases(stories, request.use_rag)

        # 2. Automation Agent runs with a Retry Loop
        code = agent3_generate_code(test_cases, request.use_rag)
        compliance_result = selenium_evaluator.evaluate(code)

        max_retries = 1  # We'll try to fix it once if it fails
        retry_count = 0

        while "❌ REJECTED" in compliance_result and retry_count < max_retries:
            print(f"🔄 [Self-Correction] Code failed (Score: {compliance_result}). Retrying...")

            # Feed the 'REJECTED' message back into the agent
            code = agent3_generate_code(test_cases, request.use_rag, feedback=compliance_result)

            # Re-evaluate the new code
            compliance_result = selenium_evaluator.evaluate(code)
            retry_count += 1

        if retry_count > 0:
            print(f"✅ [Self-Correction] Attempt {retry_count} finished with: {compliance_result}")

        return {
            "user_stories": stories,
            "test_cases": test_cases,
            "test_code": code,
            "compliance_evaluation": compliance_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_docs")
def upload_documents(files: List[UploadFile] = File(...)):
    try:
        # 1. Create docs directory if it doesn't exist
        os.makedirs("docs", exist_ok=True)

        saved_files = []
        # 2. Save uploaded files to the folder
        for file in files:
            file_path = os.path.join("docs", file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)

        # 3. Trigger the RAG ingestion to update ChromaDB
        ingest_documents("docs")

        return {"message": f"Successfully embedded: {', '.join(saved_files)}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "message": "Backend is running"}