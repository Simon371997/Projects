import requests
import json

print("Testing Ollama local server...")

url = "http://localhost:11434/api/chat"

payload = {
    "model": "qwen3:latest",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
}
response = requests.post(url, json=payload, stream=True)

if response.status_code == 200:
    print("Streaming from Ollama:")
    for chunk in response.iter_lines(decode_unicode=True):
        if chunk:
            try:
                data = json.loads(chunk)
                if "message" in data and "content" in data["message"]:
                    print(data["message"]["content"], end="")
            except json.JSONDecodeError:
                print("Error decoding JSON:", chunk)
    print()
else:
    print(f"Error: {response.status_code} - {response.text}")
