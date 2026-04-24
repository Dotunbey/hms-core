import requests
import json

url = "http://localhost:8000/api/memory/query"

data = {
    "query": "write an email to john doe about the quarterly results",
    "include_graph": True
}

print(f"Sending query: '{data['query']}'")
response = requests.post(url, json=data)

print(f"Status Code: {response.status_code}")
try:
    print(json.dumps(response.json(), indent=2))
except:
    print(response.text)
