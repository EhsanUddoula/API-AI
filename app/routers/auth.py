from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, APIRouter
from sqlalchemy.orm import Session
from ..database import get_db
from .. import models,tables,utils,oauth2
from sqlalchemy.exc import IntegrityError
from fastapi.security import OAuth2PasswordBearer
from typing import Annotated

router= APIRouter(
    prefix="/auth",
    tags=['Auth']
)

@router.post("/register")
def post_user(user : Annotated[models.UserModel, Form()], db: Session = Depends(get_db)):
    hashed_password= utils.hash(user.password)
    user.password= hashed_password
    # Create a new user instance
    db_user = tables.User(name=user.name, email=user.email, password=user.password, role=user.role)

    try:
        # Add and commit the new user to the database
        db.add(db_user)
        db.commit()
        db.refresh(db_user)  # Retrieve the user with the newly generated ID
        # Generate a JWT token
        token_data = {"user_id": db_user.id, "role": db_user.role.value}
        token = oauth2.create_access_token(data=token_data)

        # Return the user and token
        return {"access_token": token}
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Email already exists.")
    

@router.post("/login")
def login(user: models.LoginModel, db: Session = Depends(get_db)):
    # Retrieve the user from the database by email
    db_user = db.query(tables.User).filter(tables.User.email == user.email).first()
    if not db_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    
    # Verify the password
    if not utils.verify(user.password, db_user.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    
    # Generate JWT token
    token_data = {
        "user_id": db_user.id,
        "role": db_user.role.value  # Convert Enum to string
    }
    access_token = oauth2.create_access_token(data=token_data)
    return {"access_token": access_token}
