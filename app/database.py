from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from .tables import Base
import os

# Define the SQLite database path (ensure it's in your project folder or adjust the path)
DATABASE_URL = "sqlite:///./database.db"  # This creates a file called 'database.db' in the current directory

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}) 

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON;")  # Enable foreign key constraints
    cursor.close()

# Create a sessionmaker that will provide a session for database interaction
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) 

# Create the database tables (if they don't exist)
def create_db_and_tables():
    Base.metadata.create_all(bind=engine)

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
