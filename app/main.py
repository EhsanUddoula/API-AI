from fastapi import FastAPI
from .database import get_db,create_db_and_tables
from .routers import users,quiz,auth,summary,translate,notes,chat

app = FastAPI()
app.include_router(users.router)
app.include_router(quiz.router)
app.include_router(auth.router)
app.include_router(summary.router)
app.include_router(translate.router)
app.include_router(notes.router)
app.include_router(chat.router)

# Initialize the database (create tables if not already created)
@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Note Sharing App!"}

