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
    quizzes = relationship("Quiz", back_populates="user", cascade="all, delete",passive_deletes=True,)
    summaries= relationship("Summary", back_populates="user", cascade="all, delete",passive_deletes=True,)
    notes= relationship("Notes", back_populates="user", cascade="all, delete",passive_deletes=True,)

class Quiz(Base):
    __tablename__ = "quizzes"

    # Primary key
    id = mapped_column(Integer, primary_key=True, index=True)

    # Optional topic
    topic = mapped_column(String, nullable=True)

    # Store quiz questions and answers
    content = mapped_column(JSON, nullable=False)
    user_id = mapped_column(Integer, ForeignKey("users_table.id",ondelete="CASCADE"), nullable=False)
    score = mapped_column(Float, nullable=True)
    date_time = mapped_column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="quizzes") 

class Summary(Base):
    __tablename__ = "summary_table"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users_table.id", ondelete="CASCADE"), nullable=False)
    content = Column(String, nullable=False)
    summary = Column(String, nullable=False)

    user = relationship("User", back_populates="summaries")

class Notes(Base):
    __tablename__ = "notes"

    id = Column(Integer, primary_key=True, index=True)
    link = Column(String, nullable=False)  # Path to the stored file
    comment = Column(String, nullable=True)  # Optional comment about the note
    user_id = Column(Integer, ForeignKey("users_table.id", ondelete="CASCADE"), nullable=False)  # Foreign key to the users table
    creation_dateTime = Column(DateTime, default=datetime.utcnow)  # Timestamp of creation
    update_dateTime = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)  # Timestamp of last update

    # Relationship with the User table (optional, depending on ORM usage)
    user = relationship("User", back_populates="notes")