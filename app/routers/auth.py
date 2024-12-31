from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, APIRouter
from sqlalchemy.orm import Session

router= APIRouter(
    prefix="/auth",
    tags=['Auth']
)

