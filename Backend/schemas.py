from pydantic import BaseModel

class GenerateRequest(BaseModel):
    requirements: str
    use_rag: bool = False   # Disabled by default for now

class GenerateResponse(BaseModel):
    user_stories: str
    test_cases: str
    test_code: str

class TranscribeResponse(BaseModel):
    transcribed_text: str