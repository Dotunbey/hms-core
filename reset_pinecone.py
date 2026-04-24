import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = "hmscore"
print(f"Checking for existing index '{index_name}'...")

try:
    if index_name in pc.list_indexes().names():
        print(f"Deleting existing index '{index_name}' with old dimensions...")
        pc.delete_index(index_name)
        print("Index deleted successfully!")
    else:
        print(f"Index '{index_name}' not found. You're good to go!")
except Exception as e:
    print(f"Error: {e}")
