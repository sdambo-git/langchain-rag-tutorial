# 🔍 Google Search API Setup Guide

This guide will help you set up real web search functionality for your RAG system using Google Custom Search API.

## 📋 Prerequisites

- Google Cloud Platform account
- Google AI Studio account (for Gemini LLM)
- Basic understanding of API configuration

## 🛠️ Step 1: Create a Google Custom Search Engine

1. **Go to Google Custom Search Engine**
   - Visit: https://cse.google.com/cse/
   - Sign in with your Google account

2. **Create a New Search Engine**
   - Click "Add" or "Create"
   - Enter a name for your search engine (e.g., "RAG Web Search")
   - In "Sites to search", enter `*` (asterisk) to search the entire web
   - Click "Create"

3. **Configure Your Search Engine**
   - After creation, click on "Control Panel"
   - Go to "Setup" > "Basics"
   - Make sure "Search the entire web" is enabled
   - Note down your **Search engine ID** (you'll need this)

## 🔑 Step 2: Get Google Search API Key

1. **Go to Google Cloud Console**
   - Visit: https://console.cloud.google.com/
   - Select your project or create a new one

2. **Enable Custom Search API**
   - Go to "APIs & Services" > "Library"
   - Search for "Custom Search API"
   - Click on it and press "Enable"

3. **Create API Key**
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "API Key"
   - Copy the generated API key
   - (Optional) Restrict the key to "Custom Search API" for security

## ⚙️ Step 3: Configure Environment Variables

Create a `.env` file in your project root with the following configuration:

```env
# Google Cloud Platform Configuration
GOOGLE_CLOUD_PROJECT_ID=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1

# Google AI Studio API Key (for Gemini LLM)
GOOGLE_API_KEY=your-google-ai-studio-api-key

# Google Custom Search API Configuration
GOOGLE_SEARCH_API_KEY=your-google-search-api-key
GOOGLE_SEARCH_ENGINE_ID=your-custom-search-engine-id

# Optional: Web Search Configuration
WEB_SEARCH_ENABLED=true
WEB_SEARCH_MAX_RESULTS=5
WEB_SEARCH_TIMEOUT=10
```

### 🔧 Configuration Details

- **GOOGLE_SEARCH_API_KEY**: The API key from Google Cloud Console
- **GOOGLE_SEARCH_ENGINE_ID**: The Search engine ID from Custom Search Engine
- **WEB_SEARCH_ENABLED**: Set to `true` to enable web search, `false` to disable
- **WEB_SEARCH_MAX_RESULTS**: Number of search results to retrieve (1-10)
- **WEB_SEARCH_TIMEOUT**: Timeout in seconds for web page content extraction

## 🧪 Step 4: Test Your Setup

1. **Install Dependencies**
   ```bash
   source activate_venv.sh
   pip install -r requirements.txt
   ```

2. **Test Web Search**
   ```bash
   python query_data.py "latest news about AI" --source web
   ```

3. **Test with Fallback**
   ```bash
   python query_data.py "what is quantum computing" --source web --fallback
   ```

## 🚀 Usage Examples

### Basic Web Search
```bash
# Search for current information
python query_data.py "latest developments in quantum computing" --source web

# Search with fallback to other sources
python query_data.py "NVIDIA BlueField-3 DPU" --source web --fallback
```

### Compare Multiple Sources
```bash
# Compare local documents vs web search
python query_data.py "AI safety guidelines" --source all

# Use auto mode (tries RAG → Web → Direct)
python query_data.py "latest AI regulations" --source auto
```

## 🔧 Troubleshooting

### Common Issues and Solutions

1. **"Google Search API not configured" Error**
   - Check that `GOOGLE_SEARCH_API_KEY` and `GOOGLE_SEARCH_ENGINE_ID` are set in `.env`
   - Verify the API key is valid and not expired

2. **"Custom Search API" Not Enabled**
   - Go to Google Cloud Console > APIs & Services > Library
   - Search for "Custom Search API" and enable it

3. **Search Results Empty**
   - Check if your search query is too specific
   - Verify your Custom Search Engine is configured to search the entire web
   - Check API quotas in Google Cloud Console

4. **Web Content Extraction Fails**
   - Some websites block automated content extraction
   - The system will fall back to using search snippets
   - Try different search terms or enable fallback mode

### API Limits and Quotas

- **Free Tier**: 100 search queries per day
- **Paid Tier**: Up to 10,000 queries per day
- **Rate Limits**: 1 query per second for free tier

## 💡 Tips for Better Results

1. **Use Specific Search Terms**
   - Instead of: "AI"
   - Use: "artificial intelligence machine learning 2024"

2. **Enable Fallback Mode**
   - Use `--fallback` flag for automatic fallback to other sources
   - Useful when search results are insufficient

3. **Combine with Local RAG**
   - Use `--source all` to compare local documents with web results
   - Use `--source auto` for intelligent source selection

4. **Monitor API Usage**
   - Check Google Cloud Console for API usage statistics
   - Set up billing alerts if using paid tier

## 🔐 Security Best Practices

1. **Protect Your API Keys**
   - Never commit `.env` file to version control
   - Use environment variables in production
   - Restrict API keys to specific services

2. **Configure API Restrictions**
   - Limit API key usage to specific IP addresses
   - Set up usage quotas and alerts
   - Regularly rotate API keys

## 📚 Additional Resources

- [Google Custom Search API Documentation](https://developers.google.com/custom-search/v1/introduction)
- [Google Cloud Console](https://console.cloud.google.com/)
- [Google Custom Search Engine](https://cse.google.com/cse/)
- [Google AI Studio](https://aistudio.google.com/)

## 🆘 Need Help?

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all API keys and IDs are correctly configured
3. Test with fallback mode enabled
4. Check Google Cloud Console for API errors and quotas

Happy searching! 🔍✨ 