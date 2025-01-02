from fastapi import FastAPI, HTTPException, Form, APIRouter
from fastapi.responses import FileResponse
from gtts import gTTS
import os
import pygame
import threading

# Initialize FastAPI app
router = APIRouter(
    prefix="/tts",
    tags=["TextToSpeech"]
)

# Initialize the pygame mixer globally
pygame.mixer.init()

def delete_file_after_playback(filename: str):
    """
    Waits for the playback to complete and then deletes the file.
    """
    while pygame.mixer.music.get_busy():  # Wait for playback to finish
        pass
    pygame.mixer.music.unload()  # Unload the file to release the lock
    if os.path.exists(filename):
        os.remove(filename)

@router.post("/")
async def text_to_speech(
    text: str = Form(...),
    language: str = Form("en"),
    slow: bool = Form(False)
):
    """
    Convert text to speech and play the audio.

    Args:
    - text (str): The text to convert to speech.
    - language (str): The language code (default: 'en').
    - slow (bool): Whether the speech should be slow (default: False).

    Returns:
    - dict: A success message.
    """
    try:
        # Validate the input text
        if not text.strip():
            raise HTTPException(status_code=400, detail="Input text cannot be empty.")

        # Generate the TTS audio
        tts = gTTS(text=text, lang=language, slow=slow)

        # Save the audio file
        output_file = "output_audio.mp3"
        tts.save(output_file)

        # Play the audio file using pygame
        pygame.mixer.music.load(output_file)
        pygame.mixer.music.play()

        # Start a thread to handle cleanup after playback
        threading.Thread(target=delete_file_after_playback, args=(output_file,)).start()

        return {"message": "Audio is playing successfully!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating TTS audio: {str(e)}")
