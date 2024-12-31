from pydantic import BaseModel, EmailStr
from datetime import datetime
from .tables import Role
from typing import Optional, Any

class UserModel(BaseModel):
    id: int
    name: str
    email: EmailStr
    password: str
    role: Role

    class config:
        orm_mode = True


class QuizModel(BaseModel):
    topic: Optional[str]  # Optional topic
    content: Any  # Storing quiz questions and answers (JSON compatible)
    user_id: int  # User who generated the quiz
    score: Optional[float]  # Optional score

    class config:
        orm_mode = True