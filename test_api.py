import requests
import json

url = "http://localhost:8000/api/memory/ingest"

# Open the file in binary mode
with open("data/dummy.pdf", "rb") as f:
    files = {
        "file": ("dummy.pdf", f, "application/pdf")
    }
    data = {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "metadata_json": json.dumps({"source": "HR"})
    }
    
    print("Sending request to API...")
    response = requests.post(url, files=files, data=data)
    
    print(f"Status Code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)
