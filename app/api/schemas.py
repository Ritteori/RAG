from pydantic import BaseModel, Field
from app.api.limits import API_MAX_QUESTION_LENGTH, API_MAX_ANSWER_LENGTH

class QueryRAG(BaseModel):
    question: str = Field(...,min_length=3,max_length=API_MAX_QUESTION_LENGTH)
    user_answer: str = Field(...,min_length=1,max_length=API_MAX_ANSWER_LENGTH)