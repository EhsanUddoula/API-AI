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
):
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
):
    db_quiz = Quiz(
        topic=quiz_data.get("topic"),
        content=quiz_data.get("content"),
        user_id=user.id,
        score=quiz_data.get("score"),
    )
    db.add(db_quiz)
    db.commit()
    db.refresh(db_quiz)
    return {"message": "Quiz saved successfully", "quiz_id": db_quiz.id}
