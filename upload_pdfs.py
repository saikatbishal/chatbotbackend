from pymongo import MongoClient
import gridfs
import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["my_portfolio_db"]
fs = gridfs.GridFS(db)

# List your local PDF files here
pdf_files = ["./public/resume1.pdf", "./public/resume2.pdf", "./public/case_study.pdf"]

for file_name in pdf_files:
    with open(file_name, "rb") as f:
        # Check if file exists to avoid duplicates
        if not fs.exists({"filename": file_name}):
            fs.put(f, filename=file_name)
            print(f"Uploaded {file_name}")
        else:
            print(f"{file_name} already exists in DB")