import os
from openai import OpenAI

# ------------------------------------
# Hugging Face Router Client
# ------------------------------------
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

# ------------------------------------
# Available Models
# ------------------------------------
MODELS = {
    "1": {
        "name": "GPT-OSS 20B",
        "id": "openai/gpt-oss-20b",
    },
    "2": {
        "name": "DeepSeek V3",
        "id": "deepseek-ai/DeepSeek-V3",
    },
    "3": {
        "name": "Qwen 2.5 7B",
        "id": "Qwen/Qwen2.5-7B-Instruct",
    },
    "4": {
        "name": "Llama 3.3 70B",
        "id": "meta-llama/Llama-3.3-70B-Instruct",
    },
    "5": {
        "name": "Mistral 7B",
        "id": "mistralai/Mistral-7B-Instruct-v0.3",
    },
}

# ------------------------------------
# Model Selection
# ------------------------------------
print("=" * 40)
print("🤖 AI Playground")
print("=" * 40)

print("\nAvailable Models:\n")

for key, value in MODELS.items():
    print(f"{key}. {value['name']}")

choice = input("\nChoose a model (1-5): ")

if choice not in MODELS:
    print("❌ Invalid choice.")
    exit()

MODEL_NAME = MODELS[choice]["name"]
MODEL_ID = MODELS[choice]["id"]

print(f"\n✅ Using: {MODEL_NAME}")
print(f"Model ID: {MODEL_ID}")

print("\nType 'exit' to quit.\n")

# ------------------------------------
# Chat Loop
# ------------------------------------
messages = []

while True:

    user = input("You: ")

    if user.lower() == "exit":
        print("\nGoodbye! 👋")
        break

    messages.append(
        {
            "role": "user",
            "content": user,
        }
    )

    try:

        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
        )

        answer = response.choices[0].message.content

        print(f"\n🤖 {answer}\n")

        messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

    except Exception as e:

        print("\n❌ Error while calling the model.")
        print(e)
        break
