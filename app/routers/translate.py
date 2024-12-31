from fastapi import APIRouter, HTTPException, Depends
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..oauth2 import get_current_user
import os
from dotenv import load_dotenv
from pathlib import Path

router = APIRouter(
    prefix="/translator",
    tags=["Translator"]
)

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Load API key from environment
GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

@router.post("/translate")
async def translate_text(
    input_language: str,
    output_language: str,
    text: str,
    current_user: dict = Depends(get_current_user),  # Inject the logged-in user
):
    """
    Translates text from the input language to the output language using LLM.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Translation service not configured")

    # Define the translation prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a translation assistant. Translate text from one language to another."
            ),
            (
                "human",
                """
                Translate the following text from {input_language} to {output_language}:
                Text: "{text}"

                Provide a clear and accurate translation.
                """
            ),
        ]
    )

    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    try:
        # Invoke the chain
        translation = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        return {"input_language": input_language, "output_language": output_language, "original_text": text, "translated_text": translation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during translation: {str(e)}")
