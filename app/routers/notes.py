from fastapi import UploadFile, File, Form, HTTPException, APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime
import os
import uuid
from ..database import get_db
from ..oauth2 import get_current_user
from ..tables import Notes

router = APIRouter(
    prefix="/notes",
    tags=["Notes"]
)

UPLOAD_DIR = "uploaded_notes/"  # Directory to store uploaded notes
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_note(
    file: UploadFile =Form(),
    comment: str = Form(...),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Generate a unique file name
    unique_name = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)
    
    # Save the uploaded file
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Create a new note record
    note = Notes(
        link=file_path,
        comment=comment,
        user_id=current_user["user_id"],
        creation_dateTime=datetime.now(),
        update_dateTime=datetime.now(),
    )
    
    db.add(note)
    db.commit()
    db.refresh(note)

    return {"message": "Note uploaded successfully", "note_id": note.id}

@router.put("/{note_id}")
async def update_note(
    note_id: int,
    comment: str,
    file: UploadFile | None = None,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    note = db.query(Notes).filter(Notes.id == note_id, Notes.user_id == current_user["user_id"]).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    if file:
        # Generate a new unique name and replace the old file
        unique_name = f"{uuid.uuid4()}_{file.filename}"
        new_file_path = os.path.join(UPLOAD_DIR, unique_name)
        with open(new_file_path, "wb") as f:
            f.write(await file.read())

        # Remove old file
        if os.path.exists(note.link):
            os.remove(note.link)

        note.link = new_file_path

    note.comment = comment
    note.update_dateTime = datetime.now()

    db.commit()
    db.refresh(note)

    return {"message": "Note updated successfully"}

@router.delete("/{note_id}")
def delete_note(
    note_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    note = db.query(Notes).filter(Notes.id == note_id, Notes.user_id == current_user["user_id"]).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    # Remove the file from the local directory
    if os.path.exists(note.link):
        os.remove(note.link)

    db.delete(note)
    db.commit()

    return {"message": "Note deleted successfully"}
