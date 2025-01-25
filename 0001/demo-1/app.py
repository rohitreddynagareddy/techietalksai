import os
from dotenv import load_dotenv
import requests
from pprint import pprint
import json
# Load environment variables from .env file
load_dotenv()
"""
{
    "id": "chatcmpl-Abri92ZlEh3N09ad1MXs59Q62A2uu",
    "choices": [
        {
            "finish_reason": "stop",
            "index": "0",
            "logprobs": "None",
            "message": {
                "content": "The sky appears blue due to a phenomenon called Rayleigh scattering. When sunlight enters the Earth's atmosphere, it encounters molecules and small particles in the air. Sunlight, or white light, is made up of a spectrum of colors, each corresponding to different wavelengths. Blue light has a shorter wavelength compared to other colors like red or yellow.\n\nWhen sunlight hits the molecules in the atmosphere, the shorter wavelengths (blue and violet) are scattered in all directions more effectively than the longer wavelengths (like red). Although violet light is scattered even more than blue, our eyes are more sensitive to blue light and also because some of the violet light is absorbed by the ozone layer. As a result, when we look up at the sky during the day, we see it as predominantly blue.\n\nThe intensity and shade of blue can change depending on various factors, including the position of the sun, atmospheric conditions, and the presence of particles or pollutants in the air. For example, during sunrise and sunset, the sun's light has to travel through more of the atmosphere, scattering the shorter wavelengths out of our line of sight and allowing the longer wavelengths (reds and oranges) to dominate the sky's color.",
                "refusal": "None",
                "role": "assistant",
                "audio": "None",
                "function_call": "None",
                "tool_calls": "None"
            }
        }
    ],
    "created": "1733587965",
    "model": "gpt-4o-mini-2024-07-18",
    "object": "chat.completion",
    "service_tier": "None",
    "system_fingerprint": "fp_bba3c8e70b",
    "usage": {
        "completion_tokens": "236",
        "prompt_tokens": "22",
        "total_tokens": "258",
        "completion_tokens_details": {
            "accepted_prediction_tokens": "0",
            "audio_tokens": "0",
            "reasoning_tokens": "0",
            "rejected_prediction_tokens": "0"
        },
        "prompt_tokens_details": {
            "audio_tokens": "0",
            "cached_tokens": "0"
        }
    },
    "_request_id": "req_95f3fb7f58d6ef6e6f6eca1112a2daea"
}

"""

# i = 1
def pretty_print_completion(completion):
    # global i
    # i = 1
    def serialize(obj):
        global i
        # If the object has a __dict__ attribute, convert it to a dictionary
        if hasattr(obj, "__dict__"):
            # print("DICT")

            return {key: serialize(value) for key, value in obj.__dict__.items()}
        # If the object is a list, serialize each item
        elif isinstance(obj, list):
            # print("LIST")
            return [serialize(item) for item in obj]
        # If the object is a dictionary, serialize its keys and values
        elif isinstance(obj, dict):
            # print("INSTANCE DICT")
            return {key: serialize(value) for key, value in obj.items()}
        # For other objects, return their string representation
        else:
            # print("OBJ")
		    # Recursively serialize the ChatCompletion object
            # print(f"STRINGS: {i}: " + str(obj))
            # i = i + 1 
            return str(obj)

    # Recursively serialize the ChatCompletion object
    serialized_completion = serialize(completion)

    # Pretty-print the serialized completion
    print(json.dumps(serialized_completion, indent=4))


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
      temperature = 0.7
    )
    pretty_print_completion(completion)
    # print(json.dumps(completion, indent=4))
    # pprint(completion)
    # if completion.status_code == 200:
    #     data = completion.json()
    #     return data  # Return the full JSON response as a Python dictionary
    # else:
    #     print(f"Error in chat response: {completion.status_code}, {completion.text}")
    #     return None
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
    if response:
        # Pretty-print JSON response
        # print(json.dumps(response, indent=4))
        # pprint(response.content)
        # print("RESPONSE")
        # print(response)       
        print("RESPONSE-1")
        pretty_print_completion(response)
        print("RESPONSE.CONTENT")
        print(response.content)
        print(response.tool_calls)
        print(response.function_call)
 
    else:
        print("No response received.")
