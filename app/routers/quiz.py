from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, APIRouter
from sqlalchemy.orm import Session
from typing import List, Optional
import google.generativeai as genai
from io import BytesIO
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .. import models
from ..database import get_db
from ..tables import Quiz
from ..oauth2 import get_current_user

# Api part starts from here

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
# genai.configure(api_key=GEMINI_API_KEY) 
# model = genai.GenerativeModel("gemini-1.5-flash")

router=APIRouter(
    prefix="/quiz",
    tags=['Quiz']
)

@router.post("/generate")
async def generate_quiz(
    topic: str | None = None,
    file: UploadFile | None = None,
    num_questions: int = Form(5),  # Default to 5 questions
    difficulty: str = Form("medium"),  # difficulty levels: easy, medium, hard
    current_user: dict = Depends(get_current_user),  # Inject the logged-in user
):
    if current_user["role"] != "student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to generate quizzes",
        )
    # If a file is uploaded, extract text from it
    if file:
        pdf_text = await extract_pdf_text(file)
    elif topic:
        pdf_text = topic
    else:
        raise HTTPException(status_code=400, detail="You must provide a topic or upload a PDF file.")

    # Generate quiz based on text (either from topic or PDF)
    quiz = await generate_quiz_from_text(pdf_text, num_questions, difficulty)
    
    return {"quiz": quiz}


async def extract_pdf_text(file: UploadFile):
    # Read the PDF file content
    file_content = await file.read()
    pdf_reader = PdfReader(BytesIO(file_content))
    
    # Extract text from all pages
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text


async def generate_quiz_from_text(text: str, num_questions: int, difficulty: str) -> List[dict]:
    # Define the chat prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a quiz generation assistant. Generate a quiz based on the provided content."
            ),
            (
                "human",
                """
                Content:
                {text}

                Difficulty: {difficulty}
                Provide {num_questions} questions. For each question, include:
                - A clear question
                - 4 multiple-choice options (A, B, C, D)
                - Mark the correct answer.

                Format the quiz as JSON:
                [
                  {{
                    "question": "Question text",
                    "options": ["A", "B", "C", "D"],
                    "answer": "Correct option"
                  }},
                  ...
                ]
                """
            ),
        ]
    )

    # Initialize the Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
    output_parser=StrOutputParser()
    # Chain the prompt and LLM
    chain = prompt | llm | output_parser

    try:
        # Invoke the chain with input variables
        response = chain.invoke(
            {
                "text": text,
                "num_questions": num_questions,
                "difficulty": difficulty,
            }
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quiz: {str(e)}")

@router.post("/save")
def save_quiz(
    quiz_data: models.QuizModel,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    db_quiz = Quiz(
        topic=quiz_data.topic,
        content=quiz_data.content,
        user_id=current_user["user_id"],
        score=quiz_data.score,
    )
    db.add(db_quiz)
    db.commit()
    db.refresh(db_quiz)
    return {"message": "Quiz saved successfully", "quiz_id": db_quiz.id}

@router.get("/{user_id}")
def get_quizzes_by_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),

):
    if current_user["role"] != "student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to generate quizzes",
        )
    # Query quizzes by user_id
    quizzes = db.query(Quiz).filter(Quiz.user_id == user_id).all()

    if not quizzes:
        raise HTTPException(status_code=404, detail="No quizzes found for the given user ID")

    return {"quizzes": quizzes}

@router.delete("/delete/{quiz_id}")
def delete_quiz(
    quiz_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    # Query the quiz by ID
    quiz = db.query(Quiz).filter(Quiz.id == quiz_id).first()

    # Check if the quiz exists
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")

    # Check if the quiz belongs to the current user
    if quiz.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="You are not authorized to delete this quiz")

    # Delete the quiz
    db.delete(quiz)
    db.commit()

    return {"message": "Quiz deleted successfully"}
