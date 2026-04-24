import requests
import json
import sys

url = "http://localhost:8000/api/agent/ask"

query = "What are the rules regarding playtesting according to the handbook?"
if len(sys.argv) > 1:
    query = sys.argv[1]

data = {
    "query": query,
    "actor_id": "test_user_01"
}

print(f"Asking Agent: '{data['query']}'")
response = requests.post(url, json=data)

print(f"Status Code: {response.status_code}")
try:
    print(json.dumps(response.json(), indent=2))
except:
    print(response.text)
