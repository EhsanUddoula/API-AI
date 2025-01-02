from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter 
from fastapi.responses import JSONResponse 
from dotenv import load_dotenv 
import openai 
import os 
from pathlib import Path
 
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
 
# Initialize OpenAI API 
openai.api_key = os.getenv("OPENAI_API_KEY") 
router = APIRouter( 
    prefix="/stt", 
    tags=["Speech-to-Text"] 
) 
 
from openai import OpenAI 
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 
 
@router.post("/transcribe/") 
async def transcribe_audio(file: UploadFile = File(...)): 
    """ 
    Endpoint to transcribe and translate an audio file. 
    Args: 
        file (UploadFile): The audio file uploaded by the user. 
 
    Returns: 
        JSONResponse: Transcription and translation of the audio file. 
    """ 
    try: 
        # Validate file type 
        print(f"Uploaded file content type: {file.content_type}") 
        print(f"Uploaded file name: {file.filename}") 
 
        # Save the uploaded file to disk 
        file_path = f"temp_{file.filename}" 
        with open(file_path, "wb") as f: 
            f.write(await file.read()) 
 
        # Open the file for transcription 
        with open(file_path, "rb") as audio_file: 
            # Transcribe the audio 
            transcript = client.audio.transcriptions.create( 
                model="whisper-1", 
                response_format="text", 
                file=audio_file 
            ) 
            print("Transcript: ", transcript) 
 
            # Translate the audio 
            translated_transcript = client.audio.translations.create( 
                model="whisper-1", 
                response_format="text", 
                file=audio_file 
            ) 
            print("Translated Transcript: ", translated_transcript) 
 
        # Remove the temporary file after processing 
        os.remove(file_path) 
 
        return JSONResponse( 
            content={ 
                "transcription": transcript, 
                "translation": translated_transcript, 
            }, 
            status_code=200 
        ) 
 
    except Exception as e: 
        return JSONResponse( 
            content={"error": str(e)}, 
            status_code=500 
        )