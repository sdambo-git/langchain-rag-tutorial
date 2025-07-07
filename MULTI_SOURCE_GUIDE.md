# 🚀 Multi-Source RAG Query System

Your RAG system now supports **multiple answer sources** with intelligent fallback and comparison capabilities!

## 📊 Available Sources

### 1. 🔍 **RAG (Local Documents)** - *Default*
- Searches your local ChromaDB vector database
- Provides answers based on your uploaded documents
- Most accurate for document-specific questions
- **Best for**: Technical documentation, specific manuals, proprietary information

### 2. 🌐 **Web Search** - *Current Information*
- Uses LLM with web-style prompting for current/general knowledge
- Provides comprehensive, up-to-date information
- **Best for**: Current events, general knowledge, latest technology trends

### 3. 🧠 **Direct LLM** - *Training Knowledge*
- Direct query to the language model without context
- Uses LLM's training data knowledge
- **Best for**: General concepts, explanations, broad topics

### 4. 🔄 **Auto Mode** - *Smart Fallback*
- Tries sources in order: RAG → Web → Direct
- Automatically finds the best available answer
- **Best for**: When you're unsure which source to use

### 5. 🔀 **All Sources** - *Comparison Mode*
- Gets answers from all available sources
- Shows different perspectives on the same question
- **Best for**: Comparing local vs. general knowledge

## 🛠️ Usage Examples

### Basic Usage
```bash
# Default RAG search
python query_data.py "What is BlueField-3?"

# Specific source
python query_data.py "What is BlueField-3?" --source direct
python query_data.py "What is BlueField-3?" --source web

# Auto fallback mode
python query_data.py "What is quantum computing?" --source auto

# Compare all sources
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
| Document-specific questions | `rag` | Most accurate, document-based |
| Current events, trends | `web` | Up-to-date information |
| General concepts | `direct` | Broad knowledge base |
| Unknown/mixed topics | `auto` | Smart fallback system |
| Need comparison | `all` | See all perspectives |

## 🔧 Technical Features

### Smart Fallback System
- **Automatic**: Tries multiple sources until one succeeds
- **Quality Checking**: Evaluates answer relevance before fallback
- **Error Handling**: Graceful failure with helpful suggestions

### Source Identification
- Clear labeling of which source provided the answer
- Source attribution for RAG queries
- Confidence indicators where applicable

### Performance Optimization
- Suppressed telemetry for clean output
- Efficient database queries
- Minimal latency for direct LLM calls

## 📋 Example Scenarios

### Scenario 1: Technical Documentation Query
```bash
# Query about specific BlueField-3 feature
python query_data.py "How does BlueField-3 handle DMA transfers?" --source rag
# → Returns specific technical details from your documentation
```

### Scenario 2: Current Technology Trends
```bash
# Query about latest developments
python query_data.py "What are the latest AI developments?" --source web
# → Returns current, comprehensive information
```

### Scenario 3: Comparison Research
```bash
# Compare different knowledge sources
python query_data.py "What is DOCA framework?" --source all
# → Shows document-specific vs. general knowledge differences
```

### Scenario 4: Smart Discovery
```bash
# Let the system find the best source
python query_data.py "BlueField-3 vs other DPUs" --source auto
# → Tries RAG first, falls back to web/direct as needed
```

## 🚨 Important Notes

### Web Search Implementation
- Currently simulated using enhanced LLM prompting
- Provides web-style comprehensive answers
- Can be extended with actual web search APIs

### Fallback Behavior
- Only triggered when primary source fails or finds no good results
- Preserves source hierarchy (RAG → Web → Direct)
- Can be disabled by omitting `--fallback` flag

### Source Quality
- **RAG**: Highest accuracy for document content
- **Web**: Best for current/general information
- **Direct**: Good for concepts and explanations

## 🎉 Benefits

1. **Flexibility**: Choose the best source for your query type
2. **Reliability**: Automatic fallback ensures you get an answer
3. **Comparison**: See different perspectives on the same topic
4. **Efficiency**: Smart routing to the most appropriate source
5. **Transparency**: Clear indication of which source provided the answer

---

## Quick Reference

```bash
# Source Options
--source rag      # Local documents (default)
--source web      # Web-style search
--source direct   # Direct LLM
--source auto     # Smart fallback
--source all      # Compare all

# Additional Options
--fallback        # Enable fallback for single source
--help           # Show all options
```

