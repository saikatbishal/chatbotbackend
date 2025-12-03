from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import asyncio
import json
import re
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from pymongo import MongoClient
import gridfs
import os
import certifi 
from dotenv import load_dotenv

TIMEOUT_SECONDS =90
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

Settings.llm = GoogleGenAI(model_name="models/gemini-1.5-flash", api_key=GOOGLE_API_KEY)
Settings.embed_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004", api_key=GOOGLE_API_KEY)

query_engine = None

def initialize_index():
    global query_engine
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where()) 
    db = client["my_portfolio_db"]
    fs = gridfs.GridFS(db)

    if not os.path.exists("temp_pdfs"):
        os.makedirs("temp_pdfs")

    print("Downloading PDFs from MongoDB...")
    for grid_out in fs.find():
        filename = os.path.basename(grid_out.filename)
        with open(f"temp_pdfs/{filename}", "wb") as f:
            f.write(grid_out.read())
            
    print("Indexing documents...")
    documents = SimpleDirectoryReader("temp_pdfs").load_data()
    index = VectorStoreIndex.from_documents(documents)

    template = (
        "You act as Saikat's portfolio assistant. Call yourself 'Portfolio'.\n"
        "SYSTEM INSTRUCTION: You must strictly answer in valid JSON format. Do not add Markdown formatting (like ```json).\n\n"
        "Your JSON response must have these two fields:\n"
        "1. 'answer': Your conversational reply to the user.\n"
        "2. 'action': One of the following strings based on the user's intent:\n"
        "   - 'scroll_home': if the user greets you or wants to go to the top.\n"
        "   - 'scroll_about': if they ask about Saikat personally or his bio.\n"
        "   - 'scroll_education': if they ask about Saikat's education, schooling or college or anything related to studies.\n"
        "   - 'scroll_education': if they ask about Saikat's educational details, school or college or anything about studies.\n"
        "   - 'scroll_projects': if they ask to see work, projects, apps, or GitHub.\n"
        "   - 'scroll_experience': if they ask about job history, skills, or resume.\n"
        "   - 'scroll_contact': if they want to email, connect, or hire Saikat.\n"
        "   - 'none': for general questions where no scrolling is needed.\n\n"
        "GUIDELINES FOR THE 'answer' FIELD:\n"
        "- Reply politely and professionally.\n"
        "- Always answer in short and concise sentences.\n"
        "- Avoid mentioning that you are an AI model. Instead, say that you are Saikat's portfolio.\n"
        "- If you don't know the answer, say 'I'm sorry, I don't have that information. Please contact Saikat directly.' and set the action to 'scroll_contact'.\n"
        "- Ask if the user would like to know more about the specific topic you just answered.\n\n"
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, answer the query.\n"
        "Query: {query_str}\n"
        "Response (JSON):"
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
    allow_origins=[
        "http://localhost:5173",
        "https://portfolio-peach-five-31.vercel.app",
        "https://portfolio-peach-five-31.vercel.app/"
    ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.options("/chat")
async def chat_options(request: Request):
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    if not query_engine:
        raise HTTPException(status_code=503, detail="Index not ready yet")
    
    try:
        response = await asyncio.wait_for(query_engine.aquery(request.question), timeout=TIMEOUT_SECONDS)
        response_text = str(response).strip()

        # --- ROBUST CLEANING LOGIC ---
        # 1. Try to find JSON inside ```json ... ``` blocks
        json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        
        # 2. If not found, try generic ``` ... ``` blocks
        if not json_match:
            json_match = re.search(r"```\s*(.*?)\s*```", response_text, re.DOTALL)

        # 3. Extract the clean string
        if json_match:
            clean_json_str = json_match.group(1)
        else:
            # No code blocks found, assume the whole text is the JSON
            clean_json_str = response_text

        # 4. Parse the clean string
        try:
            response_data = json.loads(clean_json_str)
        except json.JSONDecodeError:
            # If parsing fails, return the text as a simple answer (fallback)
            # This prevents the "```json" string from showing in UI
            response_data = {
                "answer": clean_json_str, 
                "action": "none"
            }

        return response_data

    except asyncio.TimeoutError:
        print("Log: AI Request timed out.")
        return JSONResponse(status_code=503, content={
            "answer": "I'm thinking a bit too hard and got stuck. Please try asking again!",
            "error": "timeout"
        })
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(
            status_code=500,
            content={"answer": "An error occurred while processing your request.", "error": str(e)},
        )

@app.get("/")
def read_root():
    return {"status": "Backend is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)