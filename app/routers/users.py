from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends,APIRouter
from sqlalchemy.orm import Session
from ..database import get_db,create_db_and_tables
from ..models import UserModel
from ..tables import User
from typing import List, Optional


router= APIRouter(
    prefix="/users",
    tags=['Users']
)


@router.post("/")
def post_user(user : UserModel, db: Session = Depends(get_db)):
    # Create a new user instance
    db_user = User(name=user.name, email=user.email, password=user.password, role=user.role)

    try:
        # Add and commit the new user to the database
        db.add(db_user)
        db.commit()
        db.refresh(db_user)  # Retrieve the user with the newly generated ID
        return db_user
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Email already exists.")
    
# DELETE endpoint to remove a user by their ID
@router.delete("/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    # Try to find the user by their ID
    db_user = db.query(User).filter(User.id == user_id).first()

    if db_user is None:
        # If the user is not found, raise an HTTP 404 error
        raise HTTPException(status_code=404, detail="User not found")
    
    # If the user exists, delete them
    db.delete(db_user)
    db.commit()
    return {"message": f"User with ID {user_id} has been deleted successfully."}

# GET all users
@router.get("/", response_model=List[UserModel])
def get_all_users(db: Session = Depends(get_db)):
    # Retrieve all users from the database
    users = db.query(User).all()
    return users

@router.get("/user",response_model=UserModel)
def get_user_by_id_or_email(
    db: Session = Depends(get_db),
    id: Optional[int] = None,
    email: Optional[str] = None,
):
    # Check if either id or email is provided
    if not id and not email:
        raise HTTPException(status_code=400, detail="You must provide either an id or an email.")

    # Build the query to filter the user based on id or email
    if id:
        db_user = db.query(User).filter(User.id == id).first()
    elif email:
        db_user = db.query(User).filter(User.email == email).first()

    # If no user found, raise an error
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    return db_user