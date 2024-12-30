from sqlalchemy import Integer, String, DateTime, Enum, ForeignKey, Column, Float, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime
from sqlalchemy.ext.declarative import declarative_base
from enum import Enum as PyEnum

Base = declarative_base()

# Enum class for User roles
class Role(PyEnum):
    STUDENT = "student"
    TEACHER = "teacher"

class User(Base):
    __tablename__ = "users_table"  # Table name

    # Primary key with auto-incrementing integer
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Non-nullable fields
    name: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    password: Mapped[str] = mapped_column(String, nullable=False)

    # Timestamp for record creation
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Enum for role (student or teacher)
    role: Mapped[Role] = mapped_column(Enum(Role), nullable=False)

    # Relationship with Quiz
    quizzes = relationship("Quiz", back_populates="user")

class Quiz(Base):
    __tablename__ = "quizzes"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Optional topic
    topic = Column(String, nullable=True)

    # Store quiz questions and answers
    content = Column(JSON, nullable=False)
    user_id = Column(Integer, ForeignKey("users_table.id"), nullable=False)
    score = Column(Float, nullable=True)
    date_time = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="quizzes")
