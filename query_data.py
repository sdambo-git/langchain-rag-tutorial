import argparse
import warnings
import sys
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import requests
import time
from bs4 import BeautifulSoup
from googleapiclient.discovery import build

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

# Google Custom Search API configuration
GOOGLE_SEARCH_API_KEY = os.environ.get('GOOGLE_SEARCH_API_KEY')
GOOGLE_SEARCH_ENGINE_ID = os.environ.get('GOOGLE_SEARCH_ENGINE_ID')
WEB_SEARCH_ENABLED = os.environ.get('WEB_SEARCH_ENABLED', 'true').lower() == 'true'
WEB_SEARCH_MAX_RESULTS = int(os.environ.get('WEB_SEARCH_MAX_RESULTS', '5'))
WEB_SEARCH_TIMEOUT = int(os.environ.get('WEB_SEARCH_TIMEOUT', '10'))

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

WEB_SEARCH_PROMPT_TEMPLATE = """
Answer the question based on the following web search results:

{context}

---

Please provide a comprehensive answer to this question: {question}

Use the search results to provide accurate, up-to-date information. If the search results don't contain enough information to answer the question, say so.
"""


def google_search(query, max_results=5):
    """Perform Google search using Custom Search API"""
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_ENGINE_ID:
        print("⚠️  Google Search API not configured. Please set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID in your .env file")
        return []
    
    try:
        # Build the search service
        service = build("customsearch", "v1", developerKey=GOOGLE_SEARCH_API_KEY)
        
        # Execute the search
        result = service.cse().list(
            q=query,
            cx=GOOGLE_SEARCH_ENGINE_ID,
            num=max_results
        ).execute()
        
        # Extract search results
        search_results = []
        if 'items' in result:
            for item in result['items']:
                search_results.append({
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'displayLink': item.get('displayLink', '')
                })
        
        return search_results
        
    except Exception as e:
        print(f"❌ Google Search API error: {e}")
        return []


def extract_web_content(url, timeout=10, verbose=True):
    """Extract content from a web page"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Limit text length to avoid overwhelming the LLM
        if len(text) > 2000:
            text = text[:2000] + "..."
        
        return text
        
    except Exception as e:
        if verbose:
            print(f"⚠️  Could not extract content from {url}: {e}")
        return ""


def perform_web_search(query, max_results=5, verbose=True):
    """Perform web search and extract content from results"""
    if verbose:
        print(f"🔍 Searching for: {query}")
    
    # Get search results
    search_results = google_search(query, max_results)
    
    if not search_results:
        return None, []
    
    if verbose:
        print(f"📋 Found {len(search_results)} search results")
    
    # Extract content from each result
    web_content = []
    for i, result in enumerate(search_results, 1):
        if verbose:
            print(f"📄 Processing result {i}/{len(search_results)}: {result['displayLink']}")
        
        content = extract_web_content(result['link'], WEB_SEARCH_TIMEOUT, verbose)
        if content:
            web_content.append({
                'title': result['title'],
                'url': result['link'],
                'snippet': result['snippet'],
                'content': content,
                'source': result['displayLink']
            })
        else:
            # If we can't extract content, use the snippet
            web_content.append({
                'title': result['title'],
                'url': result['link'],
                'snippet': result['snippet'],
                'content': result['snippet'],
                'source': result['displayLink']
            })
    
    # Create context from web content
    context_parts = []
    for item in web_content:
        context_parts.append(f"Source: {item['source']}\nTitle: {item['title']}\nContent: {item['content']}\n")
    
    context = "\n---\n\n".join(context_parts)
    
    return context, web_content


def evaluate_response_quality(query_text, response_text, source_type):
    """Evaluate the quality and relevance of a response to a query"""
    if not response_text or len(response_text.strip()) < 50:
        return 0  # Very short or empty responses get low score
    
    response_lower = response_text.lower()
    query_lower = query_text.lower()
    
    score = 5  # Base score
    
    # Check for insufficient information indicators
    insufficient_indicators = [
        "doesn't explain", "doesn't contain", "does not contain", "not included", "doesn't have", "does not have",
        "doesn't provide", "does not provide", "not available", "insufficient information",
        "more details", "not detailed", "doesn't describe", "does not describe", "lacks information",
        "no information", "not found", "not present", "doesn't cover", "does not cover",
        "i don't know", "i cannot", "i can't", "unable to", "cannot provide"
    ]
    
    # Penalty for insufficient information
    if any(indicator in response_lower for indicator in insufficient_indicators):
        score -= 3
    
    # Bonus for comprehensive responses
    if len(response_text) > 500:
        score += 1
    if len(response_text) > 1000:
        score += 1
    
    # Check if response directly answers the query
    query_words = set(query_lower.split())
    response_words = set(response_lower.split())
    overlap = len(query_words.intersection(response_words))
    if overlap >= len(query_words) * 0.5:  # 50% word overlap
        score += 1
    
    # Bonus for structured responses (lists, steps, etc.)
    if any(pattern in response_text for pattern in ['\n1.', '\n2.', '\n*', '\n-', '**', '###']):
        score += 1
    
    # Source-specific bonuses
    if source_type == "web" and "sources:" in response_lower:
        score += 1  # Web responses with sources are valuable
    elif source_type == "rag" and not any(indicator in response_lower for indicator in insufficient_indicators):
        score += 1  # Good RAG responses are valuable for technical queries
    elif source_type == "direct" and len(response_text) > 800:
        score += 1  # Comprehensive direct LLM responses
    
    return max(0, min(10, score))  # Clamp between 0-10


def smart_answer(query_text):
    """Run multiple sources and return the best response"""
    print("🧠 Smart Mode: Testing multiple sources to find the optimal answer...")
    
    # Get recommended source from smart analysis
    recommended_source, reasoning = analyze_query_intent(query_text)
    print(f"🎯 Primary recommendation: {recommended_source}")
    print(f"💡 Reasoning: {reasoning}")
    
    responses = {}
    
    # Always test direct LLM (fast and always available)
    print("\n🧠 Testing Direct LLM...")
    try:
        direct_response = get_direct_response(query_text)
        if direct_response:
            direct_score = evaluate_response_quality(query_text, direct_response, "direct")
            responses["direct"] = {
                "response": direct_response,
                "score": direct_score,
                "reasoning": f"Direct LLM knowledge (score: {direct_score}/10)"
            }
            print(f"✅ Direct LLM completed (score: {direct_score}/10)")
        else:
            print("❌ Direct LLM failed")
    except Exception as e:
        print(f"❌ Direct LLM error: {e}")
    
    # Test the recommended source
    print(f"\n🎯 Testing Recommended Source ({recommended_source})...")
    try:
        if recommended_source == "rag":
            rag_response = get_rag_response(query_text)
            if rag_response:
                rag_score = evaluate_response_quality(query_text, rag_response, "rag")
                responses["rag"] = {
                    "response": rag_response,
                    "score": rag_score,
                    "reasoning": f"Local documents (score: {rag_score}/10)"
                }
                print(f"✅ RAG completed (score: {rag_score}/10)")
            else:
                print("❌ RAG failed or no results")
        
        elif recommended_source == "web":
            web_response = get_web_response(query_text)
            if web_response:
                web_score = evaluate_response_quality(query_text, web_response, "web")
                responses["web"] = {
                    "response": web_response,
                    "score": web_score,
                    "reasoning": f"Web search (score: {web_score}/10)"
                }
                print(f"✅ Web search completed (score: {web_score}/10)")
            else:
                print("❌ Web search failed")
        
        # If recommended was direct, we already have it
        
    except Exception as e:
        print(f"❌ Recommended source ({recommended_source}) error: {e}")
    
    # If we don't have web results yet and other sources scored low, try web search
    if "web" not in responses and all(r["score"] < 6 for r in responses.values()):
        print("\n🌐 Scores are low, trying web search for additional information...")
        try:
            web_response = get_web_response(query_text)
            if web_response:
                web_score = evaluate_response_quality(query_text, web_response, "web")
                responses["web"] = {
                    "response": web_response,
                    "score": web_score,
                    "reasoning": f"Web search fallback (score: {web_score}/10)"
                }
                print(f"✅ Web search completed (score: {web_score}/10)")
        except Exception as e:
            print(f"❌ Web search error: {e}")
    
    if not responses:
        print("❌ All sources failed!")
        return None
    
    # Find the best response
    best_source = max(responses.keys(), key=lambda k: responses[k]["score"])
    best_response = responses[best_source]
    
    print(f"\n🏆 Best Response Selected: {best_source.upper()}")
    print(f"📊 Final Scores:")
    for source, data in sorted(responses.items(), key=lambda x: x[1]["score"], reverse=True):
        indicator = "🏆" if source == best_source else "📊"
        print(f"   {indicator} {source}: {data['score']}/10 - {data['reasoning']}")
    
    # Display the best response
    print(f"\n🤖 Best Answer (from {best_source}):")
    print("=" * 60)
    print(best_response["response"])
    
    if len(responses) > 1:
        print(f"\n💡 Note: Tested {len(responses)} sources and selected the highest-scoring response.")
    
    return best_response["response"]


def get_direct_response(query_text):
    """Get response from direct LLM"""
    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        response = model.invoke(query_text)
        return str(response.content) if response.content else None
    except Exception:
        return None


def get_rag_response(query_text):
    """Get response from RAG without printing output"""
    try:
        PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT_ID')
        LOCATION = os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1')
        
        embedding_function = VertexAIEmbeddings(
            model_name="text-embedding-005",
            project=PROJECT_ID,
            location=LOCATION
        )
        
        with redirect_stderr(io.StringIO()):
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            results = db.similarity_search_with_relevance_scores(query_text, k=3)
        
        if len(results) == 0 or results[0][1] < 0.3:
            return None
        
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        response = model.invoke(prompt)
        
        return str(response.content) if response.content else None
    except Exception:
        return None


def get_web_response(query_text):
    """Get response from web search without printing output"""
    try:
        if not WEB_SEARCH_ENABLED or not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_ENGINE_ID:
            return None
        
        context, web_content = perform_web_search(query_text, WEB_SEARCH_MAX_RESULTS, verbose=False)
        if not context:
            return None
        
        prompt_template = ChatPromptTemplate.from_template(WEB_SEARCH_PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context, question=query_text)
        
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        response = model.invoke(prompt)
        
        return str(response.content) if response.content else None
    except Exception:
        return None


def analyze_query_intent(query_text):
    """Analyze the query to determine the most appropriate source for NVIDIA/Red Hat OpenShift domain"""
    query_lower = query_text.lower()
    
    # Keywords that suggest web search is most appropriate (current/breaking info)
    web_keywords = [
        'latest', 'recent', 'current', 'today', 'now', 'breaking', 'news',
        '2024', '2025', 'this year', 'trending', 'happening', 'update',
        'announcement', 'event', 'conference', 'launch', 'release',
        'price', 'stock', 'earnings', 'financial', 'market',
        'new version', 'updates', 'patch', 'vulnerability', 'security alert',
        'gtc', 'red hat summit', 'kubecon', 'industry news',
        'comparison', 'vs', 'versus', 'benchmark', 'performance test'
    ]
    
    # Keywords that suggest local RAG is most appropriate (technical documentation)
    rag_keywords = [
        # NVIDIA specific terms
        'bluefield', 'dpu', 'nvidia', 'cuda', 'tensor', 'gpu', 'accelerator',
        'mellanox', 'infiniband', 'connectx', 'nvlink', 'nvrm', 'nvml',
        'drivers', 'toolkit', 'runtime', 'container toolkit', 'device plugin',
        'gpu operator', 'network operator', 'doca', 'dpdk', 'sr-iov',
        
        # Red Hat OpenShift specific terms  
        'openshift', 'kubernetes', 'k8s', 'operator', 'operatorhub', 'olm',
        'pod', 'deployment', 'service', 'route', 'ingress', 'pvc', 'pv',
        'configmap', 'secret', 'namespace', 'project', 'cluster',
        'node', 'worker', 'master', 'control plane', 'etcd',
        'crio', 'containerd', 'oc', 'kubectl', 'helm', 'kustomize',
        'machineconfig', 'machineconfigpool', 'scc', 'rbac',
        'operand', 'crd', 'custom resource', 'webhook', 'admission controller',
        
        # Technical/Infrastructure terms
        'configuration', 'setup', 'installation', 'deployment', 'troubleshooting',
        'error', 'issue', 'problem', 'fix', 'solution', 'debugging',
        'logs', 'monitoring', 'metrics', 'alerts', 'performance',
        'network', 'storage', 'security', 'rbac', 'authentication',
        'yaml', 'manifest', 'specification', 'api', 'cli', 'command',
        'documentation', 'guide', 'tutorial', 'how to', 'step by step',
        'architecture', 'design', 'implementation', 'best practices'
    ]
    
    # Keywords that suggest direct LLM is most appropriate (general concepts)
    direct_keywords = [
        'explain', 'define', 'concept', 'theory', 'basics', 'introduction',
        'what is', 'why', 'when', 'difference between', 'overview',
        'general', 'fundamental', 'principle', 'methodology',
        'history', 'evolution', 'comparison', 'pros and cons',
        'advantages', 'disadvantages', 'use case', 'scenario'
    ]
    
    # Count keyword matches
    web_score = sum(1 for keyword in web_keywords if keyword in query_lower)
    rag_score = sum(1 for keyword in rag_keywords if keyword in query_lower)
    direct_score = sum(1 for keyword in direct_keywords if keyword in query_lower)
    
    # Additional scoring based on query characteristics
    if any(year in query_lower for year in ['2024', '2025', '2023']):
        web_score += 2
    
    # NVIDIA/OpenShift specific scoring boosts
    if any(term in query_lower for term in ['nvidia', 'gpu', 'cuda', 'tensor', 'bluefield', 'dpu']):
        rag_score += 2  # NVIDIA terms strongly suggest local documentation
    
    if any(term in query_lower for term in ['openshift', 'kubernetes', 'operator', 'pod', 'cluster']):
        rag_score += 2  # OpenShift terms strongly suggest local documentation
    
    # Troubleshooting and technical queries favor RAG
    if any(term in query_lower for term in ['troubleshoot', 'error', 'issue', 'problem', 'fix', 'debug']):
        rag_score += 2
    
    # Installation/configuration queries favor RAG
    if any(term in query_lower for term in ['install', 'setup', 'configure', 'deploy', 'how to']):
        rag_score += 1
    
    # Long technical queries often benefit from RAG
    if len(query_lower.split()) > 10:
        rag_score += 1
    
    # Handle explanatory questions more intelligently
    if query_lower.startswith(('what is', 'define', 'explain')):
        if any(keyword in query_lower for keyword in ['latest', 'current', 'recent', 'new']):
            web_score += 2
        else:
            # For "what is" questions, prefer comprehensive sources
            # Technical terms get both web and direct boost, let other factors decide
            if any(term in query_lower for term in ['nvidia', 'openshift', 'gpu', 'operator', 'kubernetes', 'doca', 'cuda']):
                web_score += 1  # Web often has comprehensive explanations
                direct_score += 1  # LLM also good for explanations
                # Don't boost RAG here unless it's a very specific technical implementation query
            else:
                direct_score += 2  # Pure conceptual questions favor direct LLM
    
    # "How to" questions strongly favor RAG for technical implementation
    if any(phrase in query_lower for phrase in ['how to', 'how do i', 'how can i']):
        if any(term in query_lower for term in ['nvidia', 'openshift', 'gpu', 'operator', 'kubernetes', 'doca', 'cuda']):
            rag_score += 2  # Technical how-to should check docs first
        else:
            web_score += 1  # General how-to might need current info
    
    # Special handling for combined "what is X and how to use it" queries
    if ('what is' in query_lower and any(phrase in query_lower for phrase in ['how to', 'how do i', 'use it', 'using'])):
        # These need comprehensive information, prefer web for technical terms
        if any(term in query_lower for term in ['nvidia', 'openshift', 'gpu', 'operator', 'kubernetes', 'doca', 'cuda']):
            web_score += 2  # Web likely has comprehensive guides
            # Reduce RAG bonus since local docs might not have complete info
            rag_score = max(0, rag_score - 1)
    
    # Determine the best source
    if web_score > rag_score and web_score > direct_score:
        return 'web', f"Time-sensitive/current information detected (web: {web_score}, rag: {rag_score}, direct: {direct_score})"
    elif rag_score > direct_score:
        return 'rag', f"Technical/documentation query detected (rag: {rag_score}, web: {web_score}, direct: {direct_score})"
    else:
        return 'direct', f"General knowledge/conceptual query detected (direct: {direct_score}, rag: {rag_score}, web: {web_score})"


def interactive_mode():
    """Interactive mode for easy querying without command line arguments"""
    print("🤖 RAG Query System - Interactive Mode")
    print("=" * 50)
    print("💡 Tips:")
    print("  • Just type your question and hit Enter")
    print("  • AI will automatically choose the best source")
    print("  • Type 'quit' or 'exit' to leave")
    print("  • Type 'help' for source options")
    print()
    
    while True:
        try:
            query = input("🔍 Enter your query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'help':
                print("\n📚 Available commands:")
                print("  • Just type your question for smart source selection (recommended)")
                print("  • Add :rag, :web, :direct, or :all to force a specific source")
                print()
                print("🎯 When to use each mode:")
                print("  • Default (smart): Best answer automatically - DEFAULT mode for daily queries")
                print("  • :web → Force web search for current events/breaking news")
                print("  • :direct → Force LLM knowledge for general concepts/theory")
                print("  • :rag → Force local docs when you know info is in your files")
                print("  • :all → Research mode to compare all sources side-by-side")
                print()
                print("📋 Examples:")
                print("    'What is AI?' → Smart picks best source")
                print("    'latest news:web' → Force web search")
                print("    'explain physics:direct' → Force LLM knowledge")
                print("    'GPU troubleshooting:rag' → Force local docs")
                print("    'machine learning:all' → Compare all sources")
                print()
                continue
            
            if not query:
                continue
            
            # Check for source specification
            source = "smart"
            if ':' in query:
                query, source_spec = query.rsplit(':', 1)
                if source_spec.lower() in ['rag', 'web', 'direct', 'smart', 'all']:
                    source = source_spec.lower()
                    query = query.strip()
            
            print(f"\n🔍 Query: '{query}'")
            print(f"📊 Source: {source}")
            
            # Handle smart source selection
            if source == "smart":
                smart_answer(query)
            else:
                print("⏳ Processing...\n")
                
                # Route to specified source
                if source == "rag":
                    answer_rag(query, fallback_enabled=True)
                elif source == "web":
                    answer_web(query, fallback_enabled=True)
                elif source == "direct":
                    answer_direct(query, fallback_enabled=True)
                elif source == "all":
                    answer_all(query)
            
            print("\n" + "="*50 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again.\n")


def main():
    # Check if script is run without arguments for interactive mode
    if len(sys.argv) == 1:
        interactive_mode()
        return
    
    # Create CLI with multiple source options
    parser = argparse.ArgumentParser(
        description="🤖 RAG Query System with Multiple Sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
📚 SOURCE OPTIONS:
  • rag: Search local documents using ChromaDB
  • web: Search the web for real-time information
  • direct: Ask LLM directly without context
  • smart (DEFAULT): AI tests multiple sources and picks the best response
  • all: Get answers from all sources and compare

🎯 WHEN TO USE EACH MODE:
  • smart: Daily use - want the best answer quickly (DEFAULT & recommended)
  • rag: Force local docs only (when you know info is in your files)
  • web: Force web search only (for latest news/current events)
  • direct: Force LLM knowledge only (for general concepts/theory)
  • all: Research mode - compare different perspectives side-by-side

📋 EXAMPLES:
  # Interactive mode (no arguments)
  python query_data.py
  
  # Smart mode: Best answer automatically (RECOMMENDED)
  python query_data.py "What is BlueField-3?"
  
  # Web search: Current/breaking information
  python query_data.py "latest AI news 2024" --source web
  
  # Direct LLM: General concepts and explanations
  python query_data.py "explain quantum physics" --source direct
  
  # All sources: Research and comparison
  python query_data.py "cryptocurrency pros and cons" --source all
  
  # RAG: Technical documentation (when you know it's in your docs)
  python query_data.py "OpenShift operator troubleshooting" --source rag
        """
    )
    parser.add_argument(
        "query_text", 
        type=str, 
        nargs='?',  # Make query optional
        default="How do I troubleshoot NVIDIA GPU Operator installation issues on OpenShift?",
        help="The query text (default: NVIDIA GPU Operator troubleshooting question)"
    )
    parser.add_argument(
        "--source", 
        choices=["rag", "web", "direct", "smart", "all"],
        default="smart",
        help="Choose answer source: rag (local docs), web (search), direct (LLM), smart (DEFAULT - AI tests multiple sources, picks best), all (compare all)"
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
    
    # Handle smart source selection
    if args.source == "smart":
        smart_answer(query_text)
    else:
        print("⏳ Processing...\n")
        
        # Route to appropriate source handler
        if args.source == "rag":
            answer_rag(query_text, args.fallback)
        elif args.source == "web":
            answer_web(query_text, args.fallback)
        elif args.source == "direct":
            answer_direct(query_text, args.fallback)
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
            # For "what is" questions about technical terms, automatically try web search
            if any(phrase in query_text.lower() for phrase in ['what is', 'define', 'explain']) and \
               any(term in query_text.lower() for term in ['nvidia', 'openshift', 'gpu', 'operator', 'kubernetes', 'doca', 'cuda']):
                print("❌ No good results in local documents for this explanatory query.")
                print("🔄 Automatically trying web search for comprehensive information...")
                return answer_web(query_text, fallback_enabled=False)
            elif fallback_enabled:
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
        
        # Check if the response indicates insufficient information
        response_text = str(response.content) if response.content else ""
        response_lower = response_text.lower()
        insufficient_indicators = [
            "doesn't explain", "doesn't contain", "does not contain", "not included", "doesn't have", "does not have",
            "doesn't provide", "does not provide", "not available", "insufficient information",
            "more details", "not detailed", "doesn't describe", "does not describe", "lacks information",
            "no information", "not found", "not present", "doesn't cover", "does not cover"
        ]
        
        has_insufficient_info = any(indicator in response_lower for indicator in insufficient_indicators)
        
        # For explanatory or troubleshooting questions about technical terms, try web search if local docs are insufficient
        is_explanatory = any(phrase in query_text.lower() for phrase in ['what is', 'define', 'explain'])
        is_troubleshooting = any(phrase in query_text.lower() for phrase in ['troubleshoot', 'error', 'issue', 'problem', 'fix', 'debug', 'not working', 'fails', 'failed'])
        has_technical_terms = any(term in query_text.lower() for term in ['nvidia', 'openshift', 'gpu', 'operator', 'kubernetes', 'doca', 'cuda'])
        
        if has_insufficient_info and (is_explanatory or is_troubleshooting) and has_technical_terms:
            print("🤖 Initial Answer (from local documents):")
            print("=" * 60)
            print(response.content)
            print("\n⚠️ Local documents don't have comprehensive information.")
            print("🔄 Automatically searching the web for better coverage...")
            web_result = answer_web(query_text, fallback_enabled=False)
            return web_result
        
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
        
        # Check if web search is enabled and configured
        if not WEB_SEARCH_ENABLED or not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_ENGINE_ID:
            print("⚠️  Web search not configured. Using LLM with web-style context instead...")
            
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
        
        # Perform real web search
        context, web_content = perform_web_search(query_text, WEB_SEARCH_MAX_RESULTS)
        
        if not context:
            print("❌ No web search results found.")
            if fallback_enabled:
                print("🔄 Falling back to direct LLM...")
                return answer_direct(query_text, fallback_enabled=False)
            else:
                print("💡 Try --fallback to enable auto-fallback to direct LLM.")
                return None
        
        # Generate answer using web search context
        prompt_template = ChatPromptTemplate.from_template(WEB_SEARCH_PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context, question=query_text)
        
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        response = model.invoke(prompt)
        
        # Format output
        print("🤖 Answer (from web search):")
        print("=" * 60)
        print(response.content)
        print("\n🌐 Sources:")
        print("-" * 20)
        for i, item in enumerate(web_content, 1):
            print(f"{i}. {item['title']} ({item['source']})")
            print(f"   {item['url']}")
        
        print(f"\n✅ Web search successful! Found {len(web_content)} sources.")
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
