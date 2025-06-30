from langchain_google_vertexai import VertexAIEmbeddings
from dotenv import load_dotenv
import os
import numpy as np

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()

# Set up GCP project ID - you'll need to set this in your .env file
PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT_ID')
LOCATION = os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1')  # Default location

def main():
    # Get embedding for a word using Vertex AI
    embedding_function = VertexAIEmbeddings(
        model_name="textembedding-gecko@001",  # Vertex AI embedding model
        project=PROJECT_ID,
        location=LOCATION
    )
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector}")
    print(f"Vector length: {len(vector)}")

    # Compare vector of two words
    words = ("apple", "iphone")
    vector1 = embedding_function.embed_query(words[0])
    vector2 = embedding_function.embed_query(words[1])
    
    # Calculate cosine similarity
    similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    print(f"Comparing ({words[0]}, {words[1]}): {similarity:.4f}")


if __name__ == "__main__":
    main()
