import requests
import json
from base64 import b64encode
import re

# Function to prepare the image data for the request
def make_image_data(img_path):
    with open(img_path, 'rb') as f:
        img_content = b64encode(f.read()).decode()
    img_request = {
        'requests': [{
            'image': {
                'content': img_content
            },
            'features': [{
                'type': 'TEXT_DETECTION'
            }]
        }]
    }
    return json.dumps(img_request).encode()

# Function to send request to Google Vision API
def request_ocr(api_key, img_path):
    endpoint_url = 'https://vision.googleapis.com/v1/images:annotate'
    img_data = make_image_data(img_path)
    response = requests.post(endpoint_url, 
                             data=img_data, 
                             params={'key': api_key}, 
                             headers={'Content-Type': 'application/json'})
    return response.json()

# Function to detect and extract text from an image
def detect_text_from_image(api_key, img_path):
    result = request_ocr(api_key, img_path)
    
    # Handle response and extract text
    if 'error' in result:
        raise Exception(f"API Error: {result['error']['message']}")
    
    # Extract text from the result
    text_annotations = result['responses'][0].get('textAnnotations', [])
    
    if text_annotations:
        print("Detected Text:")
        full_text = ""
        for annotation in text_annotations:
            text = annotation['description']
            full_text += f"{text} "
        return full_text.strip()  # Return the concatenated detected text
    else:
        print("No text detected.")
        return None

# Debugging function to print full text for inspection
def print_detected_text(detected_text):
    print("\nFull Detected Text for Debugging:")
    print(detected_text)
    print("\n---------------------")

# Function to extract Rocket League game-specific information
def extract_game_info(detected_text):
    print_detected_text(detected_text)  # Print full detected text for debugging
    
    # Updated regex patterns to avoid incorrect matches
    patterns = {
        'Winning Team': r'WINNER\s+([A-Z\s]+)',  # Extract winning team name
        'MVP Player': r'MVP\s+([A-Za-z0-9_]+)',  # Extract MVP player name
        'Player 1': r'([A-Za-z0-9_]+)\s+HERO',  # General pattern to capture player 1's name near HERO
        'Player 2': r'HERO\s+MVP\s+([A-Za-z0-9_]+)',  # General pattern to capture player 2 near MVP
    }

    game_info = {}

    # Extract Winning Team
    match = re.search(patterns['Winning Team'], detected_text)
    if match:
        game_info['Winning Team'] = match.group(1).strip()

    # Extract MVP Player
    match = re.search(patterns['MVP Player'], detected_text)
    if match:
        game_info['MVP Player'] = match.group(1).strip()

    # Extract Player 1
    match = re.search(patterns['Player 1'], detected_text)
    if match:
        game_info['Player 1'] = match.group(1).strip()

    # Extract Player 2
    match = re.search(patterns['Player 2'], detected_text)
    if match:
        game_info['Player 2'] = match.group(1).strip()

    if game_info:
        print("\nExtracted Game Information:")
        for key, value in game_info.items():
            print(f"{key}: {value}")
    else:
        print("No game information could be extracted.")

# Set your API key and image path
api_key = "AIzaSyA0ANGz6o2bGb2fnxwU5mQpk5TgsG6AOhc"

img_path = "D:/Project/Mr. Sandhu/Game_Stats/rocket_league.jpg"

# Detect text and extract game info
detected_text = detect_text_from_image(api_key, img_path)
if detected_text:
    extract_game_info(detected_text)
