import os
import json
import requests

from openai import OpenAI

# ------------------------------------
# Hugging Face Client
# ------------------------------------

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

MODEL = "deepseek-ai/DeepSeek-V3"

# ------------------------------------
# Weather Service 
# ------------------------------------

def get_weather(city):
    # Step 1: Find latitude and longitude of the city
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"

    geo_response = requests.get(
        geo_url,
        params={
            "name": city,
            "count": 1,
        },
    )

    geo_data = geo_response.json()

    if "results" not in geo_data:
        return f"Sorry, I couldn't find '{city}'."

    location = geo_data["results"][0]

    latitude = location["latitude"]
    longitude = location["longitude"]
    city_name = location["name"]
    country = location["country"]

    # Step 2: Fetch weather using latitude & longitude
    weather_url = "https://api.open-meteo.com/v1/forecast"

    weather_response = requests.get(
        weather_url,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
            ],
        },
    )

    weather_data = weather_response.json()

    current = weather_data["current"]

    return {
        "city": city_name,
        "country": country,
        "temperature": current["temperature_2m"],
        "humidity": current["relative_humidity_2m"],
        "wind_speed": current["wind_speed_10m"],
    }


# ------------------------------------
# Tool Definition
# ------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Example: Mumbai"
                    }
                },
                "required": ["city"],
            },
        },
    }
]

# ------------------------------------
# Conversation
# ------------------------------------

messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful AI assistant."
            " Use the get_weather tool whenever the user asks about weather."
        ),
    }
]

print("=" * 50)
print("🤖 Weather Agent")
print("=" * 50)
print("Type 'exit' to quit.\n")

while True:

    user = input("You: ")

    if user.lower() == "exit":
        break

    messages.append(
        {
            "role": "user",
            "content": user,
        }
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
    )

    message = response.choices[0].message

    # ------------------------------------
    # Did the model call a tool?
    # ------------------------------------

    if message.tool_calls:

        tool_call = message.tool_calls[0]

        function_name = tool_call.function.name

        arguments = json.loads(tool_call.function.arguments)

        if function_name == "get_weather":

            city = arguments["city"]

            print(f"\n🛠 Calling get_weather({city})")

            result = get_weather(city) 

            print("\n🌦 Weather Data")
            print(json.dumps(result, indent=4))

            messages.append(message)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                }
            )

            final_response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
            )

            answer = final_response.choices[0].message.content

            print("🤖", answer, "\n")

            messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

    else:

        answer = message.content

        print("\n🤖", answer, "\n")

        messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )
