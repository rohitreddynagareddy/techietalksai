from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="computer-use-preview",
    tools=[{
        "type": "computer_use_preview",
        "display_width": 1024,
        "display_height": 768,
        "environment": "browser" # other possible values: "mac", "windows", "ubuntu"
    }],
    input=[
        {
            "role": "user",
            "content": "Check the latest OpenAI news on chrome.com."
        }
        # Optional: include a screenshot of the initial state of the environment
        # {
        #     type: "input_image",
        #     image_url: f"data:image/png;base64,{screenshot_base64}"
        # }
    ],
    reasoning={
        "generate_summary": "concise",
    },
    truncation="auto"
)

print(response.output)