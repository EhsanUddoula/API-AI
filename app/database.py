from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from .tables import Base
import os

# Define the SQLite database path (ensure it's in your project folder or adjust the path)
DATABASE_URL = "sqlite:///./database.db"  # This creates a file called 'database.db' in the current directory

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Create a sessionmaker that will provide a session for database interaction
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create the database tables (if they don't exist)
def create_db_and_tables():
    # Create all tables from the models in app
    Base.metadata.create_all(bind=engine)

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
