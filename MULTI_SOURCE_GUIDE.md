# 🚀 Multi-Source RAG Query System

Your RAG system now supports **multiple answer sources** with intelligent fallback and comparison capabilities!

## 📊 Available Sources

### 1. 🤖 **Smart Mode** - *DEFAULT & Recommended*
- AI tests multiple sources and picks the best answer automatically
- Uses intelligent scoring to select optimal response
- Always includes Direct LLM + recommended source based on query analysis
- **Best for**: Daily use - always gives optimal results

### 2. 🔍 **RAG (Local Documents)** - *Force Local Only*
- Searches your local ChromaDB vector database
- Provides answers based on your uploaded documents
- Most accurate for document-specific questions
- **Best for**: When you know the info is in your local files

### 3. 🌐 **Web Search** - *Force Web Only*
- Real web search using Google Custom Search API
- Provides current, real-time information from the internet
- **Best for**: Breaking news, current events, latest developments

### 4. 🧠 **Direct LLM** - *Force LLM Only*
- Direct query to the language model without external context
- Uses LLM's training data knowledge
- **Best for**: General concepts, explanations, theory

### 5. 🔀 **All Sources** - *Research Mode*
- Gets answers from all available sources (RAG, Web, Direct)
- Shows different perspectives side-by-side for comparison
- **Best for**: Research when you want to compare different viewpoints

## 🛠️ Usage Examples

### Basic Usage
```bash
# Default Smart mode (AI picks best source automatically)
python query_data.py "What is BlueField-3?"

# Force specific source
python query_data.py "What is BlueField-3?" --source rag
python query_data.py "What is BlueField-3?" --source web
python query_data.py "What is BlueField-3?" --source direct

# Compare all sources side-by-side
python query_data.py "What is BlueField-3?" --source all
```

### Advanced Features
```bash
# Enable fallback for single source
python query_data.py "What is machine learning?" --source rag --fallback

# Get help
python query_data.py --help
```

## 🎯 When to Use Each Source

| Query Type | Recommended Source | Why |
|------------|-------------------|-----|
| **Daily queries** | `smart` (default) | **AI picks optimal source automatically** |
| Document-specific questions | `rag` | Force local docs when you know info is there |
| Breaking news, current events | `web` | Force web search for real-time information |
| General concepts, theory | `direct` | Force LLM knowledge for explanations |
| Research, comparison | `all` | See all perspectives side-by-side |

## 🔧 Technical Features

### Smart Intelligence System
- **AI Analysis**: Analyzes your query to recommend best source
- **Multi-Source Testing**: Tests multiple sources simultaneously
- **Objective Scoring**: Rates responses 0-10 for quality and relevance
- **Automatic Selection**: Picks highest-scoring answer

### Source Identification
- Clear labeling of which source provided the answer
- Source attribution and URLs for web/RAG queries
- Scoring explanations for transparency

### Performance Optimization
- Suppressed telemetry for clean output
- Efficient database queries
- Parallel source testing for speed

## 📋 Example Scenarios

### Scenario 1: Smart Mode (Daily Use)
```bash
# Let AI pick the best source automatically
python query_data.py "What is NVIDIA DOCA and how to use it?"
# → AI tests multiple sources, picks web for comprehensive explanation
```

### Scenario 2: Technical Documentation Query
```bash
# Force local documents when you know info is there
python query_data.py "How does BlueField-3 handle DMA transfers?" --source rag
# → Returns specific technical details from your documentation
```

### Scenario 3: Current Technology Trends
```bash
# Force web search for breaking news
python query_data.py "What are the latest AI developments in 2024?" --source web
# → Returns current, real-time information from web
```

### Scenario 4: Research and Comparison
```bash
# Compare all sources side-by-side
python query_data.py "What is DOCA framework?" --source all
# → Shows local docs vs. web vs. LLM knowledge differences
```

## 🚨 Important Notes

### Web Search Implementation
- Real Google Custom Search API integration
- Extracts content from actual web pages
- Provides current, real-time information
- Requires API configuration (see GOOGLE_SEARCH_SETUP.md)

### Smart Mode Intelligence
- Always tests Direct LLM as baseline
- Tests AI-recommended source based on query analysis
- May test additional sources if scores are low
- Uses objective 0-10 scoring system

### Source Quality & Use Cases
- **Smart**: Best overall results - use for daily queries
- **RAG**: Highest accuracy for your specific documents
- **Web**: Best for current events and real-time information
- **Direct**: Good for general concepts and explanations
- **All**: Best for research when you need multiple perspectives

## 🎉 Benefits

1. **Intelligence**: AI automatically picks the best source for your query
2. **Flexibility**: Choose specific sources when needed
3. **Reliability**: Smart testing ensures high-quality answers
4. **Comparison**: See different perspectives on the same topic
5. **Efficiency**: Optimal source selection and parallel processing
6. **Transparency**: Clear scoring and source attribution

---

## Quick Reference

```bash
# Source Options (5 total)
--source smart    # AI picks best source automatically (DEFAULT)
--source rag      # Force local documents only
--source web      # Force web search only
--source direct   # Force LLM knowledge only
--source all      # Compare all sources side-by-side

# Additional Options
--fallback        # Enable fallback for single source
--help           # Show all options
```

