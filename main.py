from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from pymongo import MongoClient
import gridfs
import os
from dotenv import load_dotenv

# Load API keys
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# 1. Configure LlamaIndex (Same as before)
Settings.llm = GoogleGenAI(model_name="models/gemini-1.5-flash", api_key=GOOGLE_API_KEY)
Settings.embed_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004", api_key=GOOGLE_API_KEY)

# Global variable to hold our query engine
query_engine = None

def initialize_index():
    """
    Downloads PDFs from MongoDB GridFS, saves them locally, and builds the index.
    """
    global query_engine
    
    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client["my_portfolio_db"] # Change to your DB name
    fs = gridfs.GridFS(db)

    # Create a temp directory for PDFs
    if not os.path.exists("temp_pdfs"):
        os.makedirs("temp_pdfs")

    # Download specific files (Modify 'filename' logic as needed)
    # This reads ALL files in GridFS. To read only 3-4, add a filter.
    print("Downloading PDFs from MongoDB...")
    for grid_out in fs.find():
        # Extract just the filename, removing any path components
        filename = os.path.basename(grid_out.filename)
        with open(f"temp_pdfs/{filename}", "wb") as f:
            f.write(grid_out.read())
            
    # Load and Index
    print("Indexing documents...")
    documents = SimpleDirectoryReader("temp_pdfs").load_data()
    index = VectorStoreIndex.from_documents(documents)

    # Custom Prompt Template
    template = (
        "You are a cool, calm, and professional assistant. You answer questions based on the provided context. "
        "If the user greets you, reply politely in a professional manner.\n"
        "Here are some examples of how you should respond:\n"
        "User: Hello\n"
        "Assistant: Hello there! How can I assist you today?\n"
        "User: Hi\n"
        "Assistant: Hi! I'm here to help. What would you like to know?\n"
        "User: Good morning\n"
        "Assistant: Good morning! I hope you're having a great day. How can I be of service?\n"
        "\n"
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, answer the query.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_template = PromptTemplate(template)

    query_engine = index.as_query_engine(text_qa_template=qa_template)
    print("System ready!")

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_index()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","https://portfolio-peach-five-31.vercel.app/"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Validation error: {exc}")
    body = await request.body()
    print(f"Request body: {body.decode()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": body.decode()},
    )

# Define the data format coming from React
class QueryRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    if not query_engine:
        raise HTTPException(status_code=503, detail="Index not ready yet")
    
    # Use aquery (async) to avoid "asyncio.run() cannot be called from a running event loop"
    response = await query_engine.aquery(request.question)
    return {"answer": str(response)}

# Health check
@app.get("/")
def read_root():
    return {"status": "Backend is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)