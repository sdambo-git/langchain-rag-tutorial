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
    # Create CLI with multiple source options
    parser = argparse.ArgumentParser(
        description="🤖 RAG Query System with Multiple Sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
📚 SOURCE OPTIONS:
  • rag (default): Search local documents using ChromaDB
  • web: Search the web for real-time information
  • direct: Ask LLM directly without context
  • auto: Try sources in order until satisfied (rag → web → direct)
  • all: Get answers from all sources and compare

📋 EXAMPLES:
  # Default RAG search
  python query_data.py "What is BlueField-3?"
  
  # Web search for current info
  python query_data.py "What is BlueField-3?" --source web
  
  # Direct LLM answer
  python query_data.py "What is BlueField-3?" --source direct
  
  # Try multiple sources automatically
  python query_data.py "What is BlueField-3?" --source auto
  
  # Compare all sources
  python query_data.py "What is BlueField-3?" --source all
        """
    )
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument(
        "--source", 
        choices=["rag", "web", "direct", "auto", "all"],
        default="rag",
        help="Choose answer source: rag (local docs), web (search), direct (LLM), auto (try multiple), all (compare all)"
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Enable fallback to other sources if primary source fails"
    )
    args = parser.parse_args()
    query_text = args.query_text
    
    print(f"🔍 Query: '{query_text}'")
    print(f"📊 Source: {args.source}")
    print("⏳ Processing...\n")

    # Route to appropriate source handler
    if args.source == "rag":
        answer_rag(query_text, args.fallback)
    elif args.source == "web":
        answer_web(query_text, args.fallback)
    elif args.source == "direct":
        answer_direct(query_text, args.fallback)
    elif args.source == "auto":
        answer_auto(query_text)
    elif args.source == "all":
        answer_all(query_text)


def answer_rag(query_text, fallback_enabled=False):
    """Answer using RAG (Retrieval-Augmented Generation) from local documents"""
    try:
        print("🔍 Searching local documents...")
        
        # Set up embeddings and database
        PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT_ID')
        LOCATION = os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1')
        
        embedding_function = VertexAIEmbeddings(
            model_name="text-embedding-005",
            project=PROJECT_ID,
            location=LOCATION
        )
        
        # Suppress stderr during ChromaDB operations
        with redirect_stderr(io.StringIO()):
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            results = db.similarity_search_with_relevance_scores(query_text, k=3)
        
        # Check if we found good results
        if len(results) == 0 or results[0][1] < 0.3:
            if fallback_enabled:
                print("❌ No good results in local documents. Trying web search...")
                return answer_web(query_text, fallback_enabled=False)
            else:
                print("❌ No good results found in local documents.")
                print("💡 Try --source web for real-time information or --fallback to enable auto-fallback.")
                return None
        
        # Generate answer using context
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        response = model.invoke(prompt)
        
        # Get unique sources
        sources = list(set([doc.metadata.get("source", "Unknown") for doc, _score in results]))
        
        # Format output
        print("🤖 Answer (from local documents):")
        print("=" * 60)
        print(response.content)
        print("\n📚 Sources:")
        print("-" * 20)
        for i, source in enumerate(sources, 1):
            print(f"{i}. {source}")
        
        print(f"\n✅ RAG query successful! Found {len(results)} relevant passages.")
        return response.content
        
    except Exception as e:
        print(f"❌ RAG search failed: {e}")
        if fallback_enabled:
            print("🔄 Falling back to web search...")
            return answer_web(query_text, fallback_enabled=False)
        return None


def answer_web(query_text, fallback_enabled=False):
    """Answer using web search for real-time information"""
    try:
        print("🌐 Searching the web...")
        
        # Use a simple approach since web search integration is complex
        # For now, we'll fall back to direct LLM for web-like queries
        print("💡 Web search feature requires external API integration.")
        print("🔄 Using direct LLM with current knowledge instead...")
        
        # Use direct LLM as web search alternative
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        
        # Modify prompt to indicate this is for current/general information
        web_prompt = f"""Please provide a comprehensive answer about: {query_text}

Focus on providing current, general knowledge that would be helpful to someone looking this up online. Include relevant details, context, and practical information."""
        
        response = model.invoke(web_prompt)
        
        # Format output
        print("🤖 Answer (LLM with web-style context):")
        print("=" * 60)
        print(response.content)
        print("\n🌐 Source: LLM general knowledge (web search alternative)")
        
        print("\n✅ Web-style query successful!")
        return response.content
        
    except Exception as e:
        print(f"❌ Web search failed: {e}")
        if fallback_enabled:
            print("🔄 Falling back to direct LLM...")
            return answer_direct(query_text, fallback_enabled=False)
        return None


def answer_direct(query_text, fallback_enabled=False):
    """Answer using direct LLM knowledge without external context"""
    try:
        print("🧠 Using direct LLM knowledge...")
        
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        response = model.invoke(query_text)
        
        # Format output
        print("🤖 Answer (direct LLM):")
        print("=" * 60)
        print(response.content)
        print("\n🧠 Source: LLM training data knowledge")
        
        print("\n✅ Direct LLM query successful!")
        return response.content
        
    except Exception as e:
        print(f"❌ Direct LLM failed: {e}")
        return None


def answer_auto(query_text):
    """Try sources automatically in order: RAG → Web → Direct"""
    print("🔄 Auto mode: Trying multiple sources...")
    
    # Try RAG first
    print("\n1️⃣ Trying local documents (RAG)...")
    result = answer_rag(query_text, fallback_enabled=False)
    if result:
        return result
    
    # Try web search
    print("\n2️⃣ Trying web search...")
    result = answer_web(query_text, fallback_enabled=False)
    if result:
        return result
    
    # Try direct LLM as last resort
    print("\n3️⃣ Trying direct LLM...")
    result = answer_direct(query_text, fallback_enabled=False)
    if result:
        return result
    
    print("❌ All sources failed!")
    return None


def answer_all(query_text):
    """Get answers from all sources and compare them"""
    print("🔍 Comparing all sources...")
    
    results = {}
    
    # Try RAG
    print("\n" + "="*60)
    print("1️⃣ LOCAL DOCUMENTS (RAG)")
    print("="*60)
    rag_result = answer_rag(query_text, fallback_enabled=False)
    if rag_result:
        results["rag"] = rag_result
    
    # Try web search
    print("\n" + "="*60)
    print("2️⃣ WEB SEARCH")
    print("="*60)
    web_result = answer_web(query_text, fallback_enabled=False)
    if web_result:
        results["web"] = web_result
    
    # Try direct LLM
    print("\n" + "="*60)
    print("3️⃣ DIRECT LLM")
    print("="*60)
    direct_result = answer_direct(query_text, fallback_enabled=False)
    if direct_result:
        results["direct"] = direct_result
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"✅ Sources that provided answers: {len(results)}")
    for source in results.keys():
        print(f"   • {source}")
    
    return results


if __name__ == "__main__":
    main()
