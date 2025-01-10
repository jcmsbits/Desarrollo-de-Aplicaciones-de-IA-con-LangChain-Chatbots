import os
from openai import OpenAI
from getpass import getpass

client = OpenAI(
    api_key=getpass("API:"),  # This is the default and can be omitted
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-3.5",
)