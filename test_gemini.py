
import os
from dotenv import load_dotenv
from google import genai

# 1. Read the .env file
load_dotenv()

# 2. Get the API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("API key kidaikkala! .env file check pannu.")
else:
    # 3. Connect to Gemini
    client = genai.Client(api_key=api_key)

    # 4. Send a test question
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="Say hello in Tamil and English."
    )

    # 5. Display the AI answer
    print("Gemini response:")
    print(response.text)
