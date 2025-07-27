import requests
import base64

# OpenAI API Key (store securely, replace with environment variable or secure method)
api_key = "OpenAPI"

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "D:/Project/Mr. Sandhu/Game_Stats/ss.png"

# Get the base64 encoded image
base64_image = encode_image(image_path)

# Set up headers for the request
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Since OpenAI GPT model doesn't handle image URLs directly, 
# you'll need to provide textual data or an external analysis of the image.

# Sample payload (Text-based for chat model)
payload = {
    "model": "gpt-3.5-turbo",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Describe the content of an image that has been analyzed."}
    ],
    "max_tokens": 100
}

# Make the request
response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

# Print the response
print(response.json())
