from fastapi import FastAPI
from .database import get_db,create_db_and_tables
from .routers import users,quiz,auth,summary,translate,notes,chat

from fastapi.middleware.cors import CORSMiddleware 

app = FastAPI()
app.include_router(users.router)
app.include_router(quiz.router)
app.include_router(auth.router)
app.include_router(summary.router)
app.include_router(translate.router)
app.include_router(notes.router)
app.include_router(chat.router)

def configure_cors(app): 
    origins = [ 
        "http://localhost", 
        "https://localhost", 
        "http://localhost:5174", 
        "http://localhost:5173", 
        "http://127.0.0.1", 
    ] 
 
    app.add_middleware( 
        CORSMiddleware, 
        allow_origins=origins,  # Allows all origins or specific ones 
        allow_credentials=True, 
        allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.) 
        allow_headers=["*"],  # Allows all headers 
    ) 
     
    # Add COOP and COEP headers to the response 
    @app.middleware("http") 
    async def add_coop_coep_headers(request, call_next): 
        response = await call_next(request) 
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin" 
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp" 
        return response 
 
configure_cors(app)

# Initialize the database (create tables if not already created)
@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Note Sharing App!"}

