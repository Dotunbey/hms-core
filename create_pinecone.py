import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = "hmscore"
dimension = 3072

print(f"Creating Pinecone index '{index_name}' with dimension {dimension}...")
try:
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Index created successfully!")
    else:
        print("Index already exists!")
except Exception as e:
    print(f"Error creating index: {e}")
