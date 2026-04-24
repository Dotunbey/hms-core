import os
from dotenv import load_dotenv
import requests

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

response = requests.get(url)
data = response.json()

print("Available generation models:")
if 'models' in data:
    for m in data['models']:
        if 'generateContent' in m.get('supportedGenerationMethods', []):
            print(m['name'])
else:
    print(data)
