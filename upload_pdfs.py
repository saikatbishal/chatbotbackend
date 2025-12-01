import sys
import os
import gridfs
from pymongo import MongoClient
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# 2. Setup Database Connection
client = MongoClient(MONGO_URI)
db = client["my_portfolio_db"]
fs = gridfs.GridFS(db)

# List your local PDF files here
pdf_files = ["./public/resume1.pdf", "./public/resume2.pdf", "./public/case_study.pdf"]

def upload_files(files, mode):
    """
    Handles the logic for uploading files based on the mode.
    """
    
    # --- LOGIC FOR REWRITE MODE ---
    if mode == "rewrite":
        print("Mode: REWRITE - Deleting all existing PDFs from the database...")
        # Find all files in the bucket and delete them
        for grid_file in fs.find({}):
            fs.delete(grid_file._id)
        print("Database cleared.\n")

    # --- UPLOAD LOOP (Applies to both modes) ---
    for file_path in files:
        try:
            with open(file_path, "rb") as f:
                
                # Check if file exists in DB
                existing_file = fs.find_one({"filename": file_path})

                # If we are in APPEND mode and the file exists, delete the OLD one first
                if mode == "append" and existing_file:
                    print(f"File '{file_path}' exists. Replacing...")
                    fs.delete(existing_file._id)
                
                # Upload the file (For 'rewrite', we just upload because DB is empty.
                # For 'append', we upload the new version after deleting the old one).
                fs.put(f, filename=file_path)
                print(f"Successfully uploaded: {file_path}")

        except FileNotFoundError:
            print(f"Error: Local file not found: {file_path}")
        except Exception as e:
            print(f"An error occurred with {file_path}: {e}")

# 3. Main Execution Block
if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python upload_pdf.py [append|rewrite]")
        sys.exit(1)

    arg_mode = sys.argv[1].lower()

    if arg_mode not in ["append", "rewrite"]:
        print("Invalid argument. Please use 'append' or 'rewrite'.")
        sys.exit(1)

    upload_files(pdf_files, arg_mode)