"""Pydantic schemas for request/response validation."""
from pydantic import BaseModel
from typing import Optional


class UserCreate(BaseModel):
    username: str
    password: str


class QuestionUpdate(BaseModel):
    answer: str
    citation: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
