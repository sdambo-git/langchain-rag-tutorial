import argparse
import os
import shutil
import glob
import warnings
import hashlib
import json
import io
import fnmatch
from pathlib import Path
from contextlib import redirect_stderr
from datetime import datetime
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
    # Create argument parser with comprehensive help
    parser = argparse.ArgumentParser(
        description="🚀 RAG Database Creation and Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
📚 EXAMPLES:
  # Normal run - only processes changed/new files
  python create_database.py
  
  # Reset everything (clears database and metadata)
  python create_database.py --reset
  
  # Force reprocess all files (ignores change detection)  
  python create_database.py --force
  
  # Remove specific files from database
  python create_database.py --remove "*.txt"
  python create_database.py --remove "old_document.pdf"
  
  # Show database status
  python create_database.py --status
  
  # Get help
  python create_database.py --help

🔍 HOW IT WORKS:
  The script intelligently detects file changes using:
  • File size comparison
  • Modification time tracking  
  • Content hash verification (SHA-256)
  
  Only changed or new files are processed, making updates very fast!

📁 SUPPORTED FILE TYPES:
  • PDF files (*.pdf) - using PyPDFLoader
  • Markdown files (*.md) - using DirectoryLoader
  • Text files (*.txt) - using DirectoryLoader

💾 FILES CREATED:
  • chroma/ - Vector database directory
  • file_metadata.json - File change tracking metadata
        """
    )
    
    parser.add_argument(
        "--reset", 
        action="store_true", 
        help="🗑️  Reset the database completely. This will:\n"
             "   • Delete the entire vector database (chroma/)\n"
             "   • Remove file metadata (file_metadata.json)\n"
             "   • Force reprocessing of all files on next run"
    )
    
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="🔄 Force reprocess all files, ignoring change detection.\n"
             "   • Removes existing chunks for all files\n"
             "   • Reprocesses every file regardless of changes\n"
             "   • Updates file metadata\n"
             "   • Useful for testing or after changing chunk parameters"
    )
    
    parser.add_argument(
        "--status", 
        action="store_true", 
        help="📊 Show database status and file information.\n"
             "   • Display current database size\n"
             "   • Show tracked files and their metadata\n"
             "   • Exit without processing files"
    )
    
    parser.add_argument(
        "--remove", 
        type=str,
        metavar="PATTERN",
        help="🗑️  Remove files from database using glob pattern.\n"
             "   • Use glob patterns like '*.txt' or 'specific_file.pdf'\n"
             "   • Removes chunks from vector database\n"
             "   • Updates file metadata tracking\n"
             "   • Examples: --remove '*.txt' --remove 'old_*.pdf'"
    )
    
    args = parser.parse_args()
    
    # Handle status command
    if args.status:
        show_database_status()
        return
    
    # Handle remove command
    if args.remove:
        remove_files_from_database(args.remove)
        return
    
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


def show_database_status():
    """Display database status and file information"""
    print("📊 RAG Database Status")
    print("=" * 50)
    
    # Check if database exists
    if os.path.exists(CHROMA_PATH):
        try:
            with redirect_stderr(io.StringIO()):
                db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
                existing_items = db.get(include=[])
                total_chunks = len(existing_items["ids"])
            print(f"🗄️  Database: {CHROMA_PATH}")
            print(f"📄 Total chunks: {total_chunks}")
        except Exception as e:
            print(f"❌ Error reading database: {e}")
    else:
        print("❌ Database not found. Run without --status to create it.")
    
    # Show file metadata
    print(f"\n📁 File Tracking: {METADATA_FILE}")
    stored_metadata = load_file_metadata()
    
    if stored_metadata:
        print(f"📊 Tracked files: {len(stored_metadata)}")
        print("\nFile Details:")
        print("-" * 50)
        
        for file_path, metadata in stored_metadata.items():
            file_size_mb = metadata['size'] / (1024 * 1024)
            modified_time = datetime.fromtimestamp(metadata['modified']).strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"📄 {file_path}")
            print(f"   Size: {file_size_mb:.2f} MB")
            print(f"   Modified: {modified_time}")
            print(f"   Hash: {metadata['hash'][:16]}...")
            print()
    else:
        print("📝 No files tracked yet. Run without --status to process files.")
    
    # Show current files in directory
    print(f"\n📂 Files in {DATA_PATH}:")
    current_files = get_all_files()
    if current_files:
        for file_info in current_files:
            file_size_mb = file_info['size'] / (1024 * 1024)
            stored_info = stored_metadata.get(file_info['path'])
            
            if stored_info:
                if has_file_changed(file_info, stored_info):
                    status = "🔄 CHANGED"
                else:
                    status = "✅ UP TO DATE"
            else:
                status = "✨ NEW"
            
            print(f"   {status} {file_info['path']} ({file_size_mb:.2f} MB)")
    else:
        print("   No supported files found.")
    
    print(f"\n🔍 Supported file types: *.pdf, *.md, *.txt")
    print(f"📖 Use --help for more information")
    
    return


def remove_files_from_database(pattern):
    """Remove files matching the glob pattern from the database"""
    print(f"🔍 Searching for files matching pattern: '{pattern}'")
    
    # Load metadata to see what files are tracked
    stored_metadata = load_file_metadata()
    
    if not stored_metadata:
        print("❌ No files are tracked in the database.")
        return
    
    # Find files matching the pattern
    matching_files = []
    
    for file_path in stored_metadata.keys():
        # Check if the file matches the pattern
        if fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(os.path.basename(file_path), pattern):
            matching_files.append(file_path)
    
    if not matching_files:
        print(f"❌ No files found matching pattern: '{pattern}'")
        print("📋 Available files:")
        for file_path in stored_metadata.keys():
            print(f"   • {file_path}")
        return
    
    # Show what will be removed
    print(f"\n🗑️  Files to remove ({len(matching_files)}):")
    for file_path in matching_files:
        print(f"   • {file_path}")
    
    # Confirm removal
    try:
        confirm = input(f"\n❓ Remove {len(matching_files)} file(s) from database? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("❌ Operation cancelled.")
            return
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled.")
        return
    
    # Remove files from database
    removed_count = 0
    total_chunks_removed = 0
    
    try:
        with redirect_stderr(io.StringIO()):
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
            
            for file_path in matching_files:
                chunks_removed = remove_chunks_for_file(db, file_path)
                if chunks_removed > 0:
                    removed_count += 1
                    total_chunks_removed += chunks_removed
                    
                    # Remove from metadata
                    if file_path in stored_metadata:
                        del stored_metadata[file_path]
                        print(f"✅ Removed {file_path} ({chunks_removed} chunks)")
                else:
                    print(f"⚠️  No chunks found for {file_path}")
    
    except Exception as e:
        print(f"❌ Error removing files: {e}")
        return
    
    # Save updated metadata
    save_file_metadata(stored_metadata)
    
    print(f"\n🎉 Removal complete!")
    print(f"   • Files removed: {removed_count}")
    print(f"   • Chunks removed: {total_chunks_removed}")
    print(f"   • Metadata updated: {METADATA_FILE}")
    
    return


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