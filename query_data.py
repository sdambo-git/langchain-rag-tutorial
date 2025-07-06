import argparse
import warnings
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Disable ChromaDB telemetry completely BEFORE importing
import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_CLIENT_TELEMETRY'] = 'False'

# Suppress all logging and telemetry
import logging
import sys
import io
from contextlib import redirect_stderr

# Suppress telemetry logging before imports
logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.ERROR)

# Load environment variables
load_dotenv()

# For Google AI Studio, we need GOOGLE_API_KEY instead of GCP project settings
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print("Please set GOOGLE_API_KEY in your .env file")
    print("You can get an API key from: https://aistudio.google.com/app/apikey")
    exit(1)

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    
    print(f"🔍 Searching for: '{query_text}'")
    print("⏳ Processing query...")

    # Keep using Vertex AI embeddings as they work with GCP authentication
    # We'll need to use the previous GCP setup for embeddings
    import os
    PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT_ID')
    LOCATION = os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1')
    
    embedding_function = VertexAIEmbeddings(
        model_name="text-embedding-005",  # Latest Vertex AI embedding model
        project=PROJECT_ID,
        location=LOCATION
    )
    # Suppress stderr during ChromaDB operations to hide telemetry errors
    with redirect_stderr(io.StringIO()):
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        # Search the DB.
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    # Check if we found any results
    if len(results) == 0:
        print("❌ No results found in the database.")
        print("Try a more general query or check if the database contains relevant content.")
        return
    
    # Use a lower threshold (0.3) to be more permissive
    if results[0][1] < 0.3:
        print(f"Best match score ({results[0][1]:.4f}) is below threshold (0.3).")
        print("Try a more general query or check if the database contains relevant content.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Use Google AI Studio chat model (simpler than Vertex AI)
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro"  # This should work with Google AI Studio
    )
    response = model.invoke(prompt)
    
    # Extract just the content from the response
    response_text = response.content
    
    # Get unique sources
    sources = list(set([doc.metadata.get("source", "Unknown") for doc, _score in results]))
    
    # Format the output nicely
    print("🤖 Answer:")
    print("=" * 60)
    print(response_text)
    print("\n📚 Sources:")
    print("-" * 20)
    for i, source in enumerate(sources, 1):
        print(f"{i}. {source}")
    
    print(f"\n✅ Query processed successfully! Found {len(results)} relevant passages.")


if __name__ == "__main__":
    main()
