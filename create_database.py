import argparse
import os
import shutil
import glob
import warnings
import hashlib
import json
import io
from pathlib import Path
from contextlib import redirect_stderr
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Disable ChromaDB telemetry completely
import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_CLIENT_TELEMETRY'] = 'False'

# Suppress telemetry logging
import logging
logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.ERROR)

# Load environment variables
load_dotenv()

# Set up GCP project ID - you'll need to set this in your .env file
PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT_ID')
LOCATION = os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1')

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"
METADATA_FILE = "file_metadata.json"


def main():
    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--force", action="store_true", help="Force reprocess all files.")
    args = parser.parse_args()
    
    if args.reset:
        print("✨ Clearing Database")
        clear_database()
        # Also clear metadata when resetting
        if os.path.exists(METADATA_FILE):
            os.remove(METADATA_FILE)
            print(f"🗑️  Cleared file metadata: {METADATA_FILE}")

    # Load existing file metadata
    stored_metadata = load_file_metadata()
    
    # Check which files have changed
    print("🔍 Checking for file changes...")
    current_files = get_all_files()
    
    changed_files = []
    new_files = []
    
    for file_info in current_files:
        stored_info = stored_metadata.get(file_info['path'])
        
        if args.force or has_file_changed(file_info, stored_info):
            if stored_info:
                changed_files.append(file_info)
                print(f"📝 Changed: {file_info['path']}")
            else:
                new_files.append(file_info)
                print(f"✨ New: {file_info['path']}")
    
    if not changed_files and not new_files:
        print("✅ No file changes detected. Database is up to date.")
        return
    
    # Process changed and new files
    files_to_process = changed_files + new_files
    print(f"\n📚 Processing {len(files_to_process)} files...")
    
    # Remove chunks for changed files
    if changed_files:
        print("🗑️  Removing old chunks for changed files...")
        with redirect_stderr(io.StringIO()):
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
            
            for file_info in changed_files:
                remove_chunks_for_file(db, file_info['path'])
    
    # Load and process documents
    documents = load_documents_for_files([f['path'] for f in files_to_process])
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    
    # Update metadata for processed files
    for file_info in files_to_process:
        stored_metadata = update_file_metadata(stored_metadata, file_info)
    
    # Save updated metadata
    save_file_metadata(stored_metadata)
    print(f"💾 Updated file metadata: {METADATA_FILE}")


def get_embedding_function():
    """Get Vertex AI embedding function"""
    return VertexAIEmbeddings(
        model_name="text-embedding-005",  # Latest Vertex AI embedding model
        project=PROJECT_ID,
        location=LOCATION
    )


def get_file_hash(file_path):
    """Calculate SHA-256 hash of a file to detect changes"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def get_file_info(file_path):
    """Get file information for change detection"""
    stat = os.stat(file_path)
    return {
        'path': file_path,
        'size': stat.st_size,
        'modified': stat.st_mtime,
        'hash': get_file_hash(file_path)
    }


def get_all_files():
    """Get all files in the data directory with their info"""
    files_info = []
    
    # Get all supported file types
    for pattern in ["*.md", "*.txt", "*.pdf"]:
        files = glob.glob(os.path.join(DATA_PATH, pattern))
        for file_path in files:
            files_info.append(get_file_info(file_path))
    
    return files_info


def has_file_changed(file_info, stored_info):
    """Check if a file has changed compared to stored information"""
    if not stored_info:
        return True  # New file
    
    # Check if size, modification time, or hash changed
    return (file_info['size'] != stored_info.get('size') or
            file_info['modified'] != stored_info.get('modified') or
            file_info['hash'] != stored_info.get('hash'))


def remove_chunks_for_file(db, file_path):
    """Remove all chunks for a specific file from the database"""
    # Get all documents with this source
    results = db.get(where={"source": file_path})
    
    if results['ids']:
        print(f"🗑️  Removing {len(results['ids'])} old chunks for: {file_path}")
        db.delete(ids=results['ids'])
        return len(results['ids'])
    return 0


def load_file_metadata():
    """Load file metadata from JSON file"""
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"⚠️  Warning: Could not read {METADATA_FILE}, treating all files as new")
            return {}
    return {}


def save_file_metadata(metadata):
    """Save file metadata to JSON file"""
    try:
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
    except IOError as e:
        print(f"⚠️  Warning: Could not save file metadata: {e}")


def update_file_metadata(metadata, file_info):
    """Update metadata for a specific file"""
    metadata[file_info['path']] = {
        'size': file_info['size'],
        'modified': file_info['modified'],
        'hash': file_info['hash']
    }
    return metadata


def load_documents():
    """Load documents from multiple file types: PDF, Markdown, and Text"""
    documents = []
    
    # Load markdown files
    md_loader = DirectoryLoader(DATA_PATH, glob="*.md")
    md_documents = md_loader.load()
    documents.extend(md_documents)
    print(f"Loaded {len(md_documents)} markdown documents")
    
    # Load text files
    txt_loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    txt_documents = txt_loader.load()
    documents.extend(txt_documents)
    print(f"Loaded {len(txt_documents)} text documents")
    
    # Load PDF files
    pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
    pdf_documents = []
    for pdf_file in pdf_files:
        try:
            pdf_loader = PyPDFLoader(pdf_file)
            pdf_docs = pdf_loader.load()
            pdf_documents.extend(pdf_docs)
        except Exception as e:
            print(f"Error loading PDF {pdf_file}: {str(e)}")
            continue
    
    documents.extend(pdf_documents)
    print(f"Loaded {len(pdf_documents)} PDF documents")
    
    print(f"Total documents loaded: {len(documents)}")
    return documents


def load_documents_for_files(file_paths):
    """Load documents only for specified files"""
    documents = []
    
    md_count = 0
    txt_count = 0
    pdf_count = 0
    
    for file_path in file_paths:
        try:
            if file_path.endswith('.md'):
                # Load single markdown file
                md_loader = DirectoryLoader(os.path.dirname(file_path), glob=os.path.basename(file_path))
                md_docs = md_loader.load()
                documents.extend(md_docs)
                md_count += len(md_docs)
                
            elif file_path.endswith('.txt'):
                # Load single text file
                txt_loader = DirectoryLoader(os.path.dirname(file_path), glob=os.path.basename(file_path))
                txt_docs = txt_loader.load()
                documents.extend(txt_docs)
                txt_count += len(txt_docs)
                
            elif file_path.endswith('.pdf'):
                # Load single PDF file
                pdf_loader = PyPDFLoader(file_path)
                pdf_docs = pdf_loader.load()
                documents.extend(pdf_docs)
                pdf_count += len(pdf_docs)
                
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            continue
    
    print(f"Loaded {md_count} markdown documents, {txt_count} text documents, {pdf_count} PDF documents")
    print(f"Total documents loaded: {len(documents)}")
    return documents


def split_documents(documents: list[Document]):
    """Split documents into chunks with improved parameters"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Larger chunks for better context
        chunk_overlap=80,  # 10% overlap
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    # Show example chunk if available
    if len(chunks) > 0:
        document = chunks[0]
        print(f"Example chunk content: {document.page_content[:200]}...")
        print(f"Example chunk metadata: {document.metadata}")
    
    return chunks


def add_to_chroma(chunks: list[Document]):
    """Add or update documents in ChromaDB with incremental updates"""
    # Suppress stderr during ChromaDB operations to hide telemetry errors
    with redirect_stderr(io.StringIO()):
        # Load the existing database or create new one
        db = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=get_embedding_function()
        )

        # Calculate Page IDs for deduplication
        chunks_with_ids = calculate_chunk_ids(chunks)

        # Add or Update the documents
        existing_items = db.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids"])
    
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        
        # Suppress stderr for the add operation too
        with redirect_stderr(io.StringIO()):
            db.add_documents(new_chunks, ids=new_chunk_ids)
        
        print(f"✅ Added {len(new_chunks)} new chunks to {CHROMA_PATH}")
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):
    """Calculate unique IDs for each chunk to enable incremental updates"""
    # This will create IDs like "data/books/file.pdf:6:2"
    # Format: Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page", 0)  # Default to 0 for non-PDF files
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    """Clear the entire ChromaDB database"""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"🗑️  Database cleared: {CHROMA_PATH}")
    else:
        print("📁 Database directory does not exist")


if __name__ == "__main__":
    main() 