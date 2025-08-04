from typing import List
from pydantic import BaseModel

class TaskRequest(BaseModel):
    text: str

class AnswerPayload(BaseModel):
    answers: List[str]
