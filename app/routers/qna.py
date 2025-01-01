from fastapi import APIRouter, UploadFile, HTTPException, Form, File, Depends
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

router = APIRouter(
    prefix="/qna",
    tags=["QNA"]
)

# Utility Functions
def extract_text_from_file(file: UploadFile) -> str:
    text = ""
    try:
        if file.content_type == "application/pdf":
            pdf_reader = PdfReader(file.file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif file.content_type.startswith("text/"):
            text = file.file.read().decode("utf-8")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")
    return text


def split_text_into_chunks(text: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


def create_vector_store(chunks: list):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GEMINI_API_KEY)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def load_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not available in the provided context, respond with 
    "The answer is not available in the context." Do not provide incorrect information.

    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def search_and_respond(user_question: str):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GEMINI_API_KEY)
    vector_store = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    relevant_docs = vector_store.similarity_search(user_question)
    chain = load_conversational_chain()
    response = chain.invoke({"input_documents": relevant_docs, "question": user_question})
    return response["output_text"]



# API Endpoints
@router.post("/upload-files")
async def upload_files(files: list[UploadFile] = File(...)):
    try:
        full_text = ""
        for file in files:
            full_text += extract_text_from_file(file) + "\n"

        if not full_text.strip():
            raise HTTPException(status_code=400, detail="No valid text extracted from files.")

        chunks = split_text_into_chunks(full_text)
        create_vector_store(chunks)
        return {"message": "Files processed and vector store updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")


@router.post("/ask-question")
async def ask_question(question: str = Form(...)):
    try:
        response = search_and_respond(question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
