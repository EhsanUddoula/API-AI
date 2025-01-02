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
from ..tables import Summary
from ..oauth2 import get_current_user
from PIL import Image
import pytesseract
from io import BytesIO
import openai
import base64
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Api part starts from here

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
OPENAI_API_KEY= os.getenv("OPEN_AI_API_KEY")
# genai.configure(api_key=GEMINI_API_KEY) 
# model = genai.GenerativeModel("gemini-1.5-flash")
# nltk.download('vader_lexicon')

router=APIRouter(
    prefix="/summary",
    tags=['Summary']
)

@router.post("/generate")
async def summarize_content(
    file: UploadFile | None = None,
    current_user: dict = Depends(get_current_user),  # Check logged-in user
):
    if not file :
        raise HTTPException(status_code=400, detail="You must upload a file for summarization.")

    # Ensure the user has appropriate permissions
    if current_user["role"] not in ["student", "teacher"]:
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to summarize content."
        )

    # Extract text from the uploaded file if provided
    if file:
        content_text = await extract_text_from_file(file)
    else:
        content_text = text

    # Perform summarization
    summary = await generate_summary(content_text)
    
    return {"summary": summary}


async def extract_text_from_file(file: UploadFile) -> str:
    # Read file content
    file_content = await file.read()

    # Determine file type and extract text accordingly
    if file.filename.endswith(".pdf"):
        pdf_reader = PdfReader(BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    else:
        # Handle plain text or other text-based files
        return file_content.decode("utf-8")


async def generate_summary(content: str) -> str:
    # Define a prompt template for summarization
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a summarization assistant. Provide a short content name and a concise summary of the provided text."
            ),
            (
                "human",
                """
                Content:
                {content}

                Instructions:
                - For the content name, use the main heading or the first significant line from the content.
                - Provide a concise summary of the text(not too big, not too small).

                Format the output as JSON:
                {{
                    "content": "Short name reflecting the heading or first line",
                    "summary": "Generated summary"
                }}
                """
            ),
        ]
    )

    # Initialize the Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
    output_parser = StrOutputParser()
    
    # Chain the prompt and LLM
    chain = prompt | llm | output_parser

    try:
        # Invoke the chain with input variables
        response = chain.invoke({"content": content})
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

@router.post("/save")
def save_summary(
    summary_data: dict,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    if "content" not in summary_data or "summary" not in summary_data:
        raise HTTPException(status_code=400, detail="Invalid data. 'content' and 'summary' are required.")
    
    db_summary = Summary(
        user_id=current_user["user_id"],
        content=summary_data["content"],
        summary=summary_data["summary"],
    )
    db.add(db_summary)
    db.commit()
    db.refresh(db_summary)

    return {"message": "Summary saved successfully", "summary_id": db_summary.id}

@router.delete("/delete/{summary_id}")
def delete_summary(
    summary_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    # Query the summary by ID
    summary = db.query(Summary).filter(Summary.id == summary_id).first()

    # Check if the summary exists
    if not summary:
        raise HTTPException(status_code=404, detail="Summary not found")

    # Check if the summary belongs to the current user
    if summary.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="You are not authorized to delete this summary")

    # Delete the summary
    db.delete(summary)
    db.commit()

    return {"message": "Summary deleted successfully"}

@router.post("/extract-text-gemini")
async def extract_text(file: UploadFile = File(...)):
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Upload a PNG or JPEG image.")

    # Read the image file
    image = Image.open(BytesIO(await file.read()))

    # Step 1: Extract raw text using OCR
    raw_text = pytesseract.image_to_string(image)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
    
    # Step 2: Enhance text with LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant that cleans and enhances OCR output from images."),
        ("human", f"""
            The following text was extracted from an image:
            {raw_text}

            Please clean it up, fix any errors, and return a readable and meaningful text.
        """)
    ])
    
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    
    # Step 3: Get enhanced text
    enhanced_text = chain.invoke({"raw_text": raw_text})
    return {"raw_text": raw_text, "enhanced_text": enhanced_text}



# @router.post("/extract-text-openai")
# async def extract_text_from_image(image_file: UploadFile):
#     try:
#         # Step 1: Read the uploaded image file
#         image = Image.open(BytesIO(await image_file.read()))

#         # Step 2: Extract text using Tesseract OCR
#         text = pytesseract.image_to_string(image)
#         if not extracted_text.strip():
#             raise HTTPException(status_code=400, detail="No text detected in the image.")

#         # Step 3: Define the ChatPromptTemplate
#         system_message = SystemMessagePromptTemplate.from_template(
#             "You are a helpful assistant specializing in organizing and cleaning up text extracted from images."
#         )
#         human_message = HumanMessagePromptTemplate.from_template(
#             "The following text was extracted from an image:\n\n{text}\n\n"
#             "Please clean and organize the text, making it readable and well-structured."
#         )

#         prompt_template = ChatPromptTemplate.from_messages([system_message, human_message])

#         # Initialize the OpenAI chat model
#         llm = ChatOpenAI(
#             model="gpt-3.5-turbo",
#             temperature=0.3,
#             openai_api_key=OPENAI_API_KEY
#         )
#         output_parser = StrOutputParser()
#         # Create the LangChain LLM chain
#         chain = prompt_template | llm | output_parser

#         # Step 4: Process the extracted text through LangChain
#         enhanced_text = chain.invoke({"raw_text": text})
#         return {"raw_text": text, "enhanced_text": enhanced_text}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error extracting text from image: {str(e)}")