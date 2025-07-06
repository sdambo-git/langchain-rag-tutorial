# 🚀 LangChain RAG System - Enhanced Multi-Source System

A comprehensive **Retrieval-Augmented Generation (RAG) system** with intelligent multi-source querying, advanced file management, and professional-grade features.

## 🎯 **System Overview**

This RAG system provides:
- **Multi-source querying** with intelligent fallback
- **Smart file change detection** and incremental updates
- **Multiple document format support** (PDF, Markdown, Text)
- **Professional database management** with glob pattern operations
- **Clean, production-ready output** with telemetry suppression

## 📁 **Project Structure**

```
langchain-rag-tutorial/
├── 🤖 query_data.py              # Multi-source query system
├── 🗄️  create_database.py        # Database management tool
├── 🔧 compare_embeddings.py      # Embedding testing utility
├── 📋 requirements.txt           # Python dependencies
├── 📖 MULTI_SOURCE_GUIDE.md     # Detailed usage guide
├── 📄 README.md                 # This file
├── 🔑 .env                      # Environment configuration
├── 📁 data/books/               # Document storage
├── 🗄️  chroma/                  # Vector database
└── 📊 file_metadata.json       # File tracking metadata
```

---

## 🤖 **Query System** (`query_data.py`)

### **Multi-Source Intelligence**

Choose from **5 different answer sources** with automatic fallback:

| Source | Description | Best For |
|--------|-------------|----------|
| 🔍 `rag` | Local documents via ChromaDB | Technical docs, manuals |
| 🌐 `web` | Web-style comprehensive answers | Current trends, general knowledge |
| 🧠 `direct` | Pure LLM knowledge | Concepts, explanations |
| 🔄 `auto` | Smart fallback (RAG→Web→Direct) | Unknown topic types |
| 🔀 `all` | Compare all sources | Research, multiple perspectives |

### **Usage Examples**

```bash
# 🔍 Default RAG search (local documents)
python query_data.py "What are the key features of BlueField-3?"

# 🌐 Web-style comprehensive answers
python query_data.py "What is the latest in AI technology?" --source web

# 🧠 Direct LLM knowledge
python query_data.py "Explain quantum computing" --source direct

# 🔄 Smart auto-fallback mode
python query_data.py "Machine learning concepts" --source auto

# 🔀 Compare all sources
python query_data.py "What is DOCA framework?" --source all

# 🛡️ Enable fallback for reliability
python query_data.py "Unrelated topic" --source rag --fallback
```

### **Query Options**

```bash
python query_data.py [QUERY] [OPTIONS]

Arguments:
  QUERY                     Your question or search term

Options:
  --source {rag,web,direct,auto,all}
                            Choose answer source (default: rag)
  --fallback               Enable fallback to other sources if primary fails
  -h, --help               Show help and examples
```

---

## 🗄️ **Database Management** (`create_database.py`)

### **Intelligent File Management**

Advanced database creation with **smart change detection** and **incremental updates**:

- ✅ **File change detection** (size, time, content hash)
- ✅ **Incremental processing** (only changed/new files)
- ✅ **Glob pattern removal** for bulk operations
- ✅ **Professional status reporting**
- ✅ **Multi-format support** (PDF, Markdown, Text)

### **Usage Examples**

```bash
# 📚 Normal processing (smart incremental updates)
python create_database.py

# 📊 Show database status and file tracking
python create_database.py --status

# 🔄 Force reprocess all files (ignore change detection)
python create_database.py --force

# 🗑️ Reset everything (database + metadata)
python create_database.py --reset

# 🗂️ Remove specific files or patterns
python create_database.py --remove "*.txt"
python create_database.py --remove "old_document.pdf"
python create_database.py --remove "temp_*"
```

### **Database Options**

```bash
python create_database.py [OPTIONS]

Options:
  --status                 Show database status and file information
  --reset                  Reset database completely (clears everything)
  --force                  Force reprocess all files (ignores change detection)
  --remove PATTERN         Remove files using glob patterns
  -h, --help              Show comprehensive help and examples
```

### **Smart Features**

- **🔍 Change Detection**: SHA-256 hash + modification time + file size
- **📊 Status Monitoring**: Database size, file tracking, change status
- **🗑️ Bulk Operations**: Remove multiple files with glob patterns
- **💾 Metadata Tracking**: JSON-based file state management
- **🔄 Incremental Updates**: Only process what changed

---

## 🔧 **Utility Tools**

### **Embedding Comparison** (`compare_embeddings.py`)

Test and compare embeddings for different terms:

```bash
# Test embedding generation and similarity
python compare_embeddings.py

# Compare similarity between words
# Example output: Comparing (apple, iphone): 0.7234
```

---

## ⚙️ **Setup and Configuration**

### **1. Install Dependencies**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install requirements
pip install -r requirements.txt
```

### **2. Configure Environment**

Create `.env` file with your credentials:

```bash
# Google Cloud Platform (for Vertex AI embeddings)
GOOGLE_CLOUD_PROJECT_ID=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1

# Google AI Studio (for chat model)
GOOGLE_API_KEY=your-google-ai-studio-key
```

### **3. Authenticate with Google Cloud**

```bash
# Install Google Cloud CLI
gcloud auth application-default login

# Set your project
gcloud config set project your-gcp-project-id
```

### **4. Prepare Your Documents**

```bash
# Add documents to the data directory
mkdir -p data/books
# Copy your PDF, Markdown, or Text files to data/books/
```

### **5. Create the Database**

```bash
# Process all documents
python create_database.py

# Check status
python create_database.py --status
```

---

## 📋 **File Format Support**

| Format | Extension | Loader | Features |
|--------|-----------|--------|----------|
| 📄 PDF | `.pdf` | PyPDFLoader | Page-level metadata |
| 📝 Markdown | `.md` | DirectoryLoader | Full text processing |
| 📃 Text | `.txt` | DirectoryLoader | Simple text files |

### **Document Processing**

- **📏 Chunk Size**: 800 characters with 80-character overlap
- **🔗 Deduplication**: Unique chunk IDs prevent duplicates
- **🏷️ Metadata**: Source file, page numbers, modification tracking
- **⚡ Performance**: Only processes changed files

---

## 🎯 **Common Workflows**

### **Daily Usage Workflow**

```bash
# 1. Check what's in your database
python create_database.py --status

# 2. Add new documents to data/books/
# (The system will auto-detect them)

# 3. Update database (only processes new/changed files)
python create_database.py

# 4. Query your documents
python query_data.py "Your question here"
```

### **Research Workflow**

```bash
# Compare multiple sources for comprehensive research
python query_data.py "Your research topic" --source all

# Get document-specific details
python query_data.py "Specific feature question" --source rag

# Get current/general information
python query_data.py "Current trends question" --source web
```

### **Maintenance Workflow**

```bash
# Clean up old files
python create_database.py --remove "old_*"
python create_database.py --remove "*.tmp"

# Reset and rebuild (if needed)
python create_database.py --reset
python create_database.py

# Force reprocess after changing settings
python create_database.py --force
```

---

## 🔍 **Advanced Features**

### **Multi-Source Intelligence**
- **Smart Routing**: Automatically choose the best source
- **Fallback System**: Never leave you without an answer
- **Source Attribution**: Always know where answers come from
- **Quality Thresholds**: Relevance scoring for better results

### **Professional Database Management**
- **File Change Detection**: SHA-256 + timestamp + size verification
- **Incremental Processing**: Lightning-fast updates
- **Glob Pattern Operations**: Bulk file management
- **Metadata Tracking**: Complete audit trail
- **Error Recovery**: Graceful handling of file issues

### **Production Features**
- **Telemetry Suppression**: Clean, professional output
- **Error Handling**: Comprehensive exception management
- **Performance Optimization**: Efficient vector operations
- **Scalable Architecture**: Handles large document collections

---

## 🎓 **Getting Started Guide**

### **Quick Start (5 minutes)**

1. **Clone and setup**:
   ```bash
   git clone <your-repo>
   cd langchain-rag-tutorial
   pip install -r requirements.txt
   ```

2. **Configure credentials** (create `.env` file with your keys)

3. **Add a document**:
   ```bash
   # Copy a PDF to data/books/
   cp your_document.pdf data/books/
   ```

4. **Create database**:
   ```bash
   python create_database.py
   ```

5. **Ask questions**:
   ```bash
   python query_data.py "What is this document about?"
   ```

### **Learning Path**

1. 📖 **Read**: [MULTI_SOURCE_GUIDE.md](MULTI_SOURCE_GUIDE.md) for detailed examples
2. 🧪 **Experiment**: Try different sources and compare results
3. 🔧 **Customize**: Adjust chunk sizes and embedding models
4. 🚀 **Scale**: Add more documents and explore advanced features

---

## 🛠️ **Technical Specifications**

- **🔗 LangChain**: 0.3.x with latest integrations
- **🧠 LLM**: Google Gemini 1.5 Pro via AI Studio
- **📊 Embeddings**: Vertex AI text-embedding-005
- **🗄️ Vector Store**: ChromaDB with persistence
- **🐍 Python**: 3.8+ compatibility
- **☁️ Cloud**: Google Cloud Platform integration

---

## 📚 **Additional Resources**

- **[MULTI_SOURCE_GUIDE.md](MULTI_SOURCE_GUIDE.md)**: Comprehensive usage guide
- **[requirements.txt](requirements.txt)**: Full dependency list
- **Google Cloud AI**: [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- **LangChain**: [Official Documentation](https://docs.langchain.com/)

---

## 🎉 **Features Summary**

✅ **Multi-source querying** with 5 different answer sources  
✅ **Smart file change detection** with SHA-256 verification  
✅ **Incremental database updates** for lightning-fast processing  
✅ **Glob pattern file management** for bulk operations  
✅ **Professional telemetry suppression** for clean output  
✅ **Comprehensive error handling** and fallback systems  
✅ **Multiple document format support** (PDF, Markdown, Text)  
✅ **Production-ready architecture** with metadata tracking  
✅ **Intelligent source routing** with quality thresholds  
✅ **Complete audit trail** with file tracking and status  

Your RAG system is now a **professional-grade, multi-source knowledge platform**! 🚀
