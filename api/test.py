import requests

# URL of your running chatbot API
url = "http://127.0.0.1:8000/chat/"

# Your input message to the chatbot
data = {
    "message": "tell me more about Online Web Posting "
}

# Send a POST request to the chatbot
response = requests.post(url, json=data)

# Print the chatbot's reply
if response.status_code == 200:
    print("Bot response:", response.json()["response"])
else:
    print("Error:", response.status_code, response.text)