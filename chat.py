import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],   # Make sure HF_TOKEN is exported
)

messages = []

print("🤖 Chat Agent")
print("Type 'exit' to quit.\n")

while True:
    user = input("You: ")

    if user.lower() == "exit":
        break

    messages.append({"role": "user", "content": user})

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
    )

    answer = response.choices[0].message.content

    print(f"\nAI: {answer}\n")

    messages.append({"role": "assistant", "content": answer})
