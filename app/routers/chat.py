from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List
from ..database import get_db
from ..tables import Chat
from ..oauth2 import get_current_user
import os
from dotenv import load_dotenv
from pathlib import Path

router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Load API key from environment
GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")


@router.post("/")
async def chatbot(
    message: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]

    # Fetch recent chat history for context
    recent_chats = (
        db.query(Chat)
        .filter(Chat.user_id == user_id)
        .order_by(Chat.timestamp.desc())
        .limit(5)  # Fetch last 5 messages
        .all()
    )

    # Format chat history for LLM
    chat_context = "\n".join(
        [f"User: {chat.message}\nChatbot: {chat.response}" for chat in reversed(recent_chats)]
    )

    # Add user message to context
    chat_context += f"\nUser: {message}"

    # Initialize LLM and construct the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a conversational assistant. Continue the conversation based on the context.",
            ),
            (
                "human",
                f"""
                Conversation Context:
                {chat_context}

                Your response should be concise and relevant.
                """,
            ),
        ]
    )

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    try:
        # Generate response
        response = chain.invoke({})

        # Save the chat to the database
        new_chat = Chat(user_id=user_id, message=message, response=response)
        db.add(new_chat)
        db.commit()

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@router.get("/")
def get_chat_history(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]

    # Query all chats for the user
    chats = db.query(Chat).filter(Chat.user_id == user_id).order_by(Chat.timestamp).all()

    if not chats:
        return {"message": "No chat history found."}

    return {"chat_history": [{"message": chat.message, "response": chat.response, "timestamp": chat.timestamp} for chat in chats]}
