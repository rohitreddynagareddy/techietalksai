import os
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

def generate_chat_response_b4(conversation_history, api_key):
    """Generate a chat response using the OpenAI API."""
    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": conversation_history,
        "temperature": 0.7
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=120)

    if response.status_code == 200:
        data = response.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
    else:
        print(f"Error in chat response: {response.status_code}, {response.text}")
        return None

def generate_chat_response(conversation_history, api_key):
    """Generate a chat response using the OpenAI API."""
	from openai import OpenAI
	client = OpenAI()

	completion = client.chat.completions.create(
	  model="gpt-4o-mini",
	  messages = conversation_history,
	  # messages=[
	  #   {"role": "system", "content": "You are a helpful assistant."},
	  #   {"role": "user", "content": "Hello!"}
	  # ]
	)

	return completion.choices[0].message

if __name__ == "__main__":
    # Configuration
    api_key = os.getenv("OPENAI_API_KEY")
    query = "Why is sky blue?"
    conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]
    conversation_history.append({"role": "user", "content": query})

    # Generate chat response
    response = generate_chat_response(conversation_history, api_key)
    print("\nFinal Answer:")
    print(response)
