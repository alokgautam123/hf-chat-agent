import os
import json
import ast
import operator
import requests
from ddgs import DDGS

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
# Safe Calculator
# ------------------------------------

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def evaluate(node):
    if isinstance(node, ast.Constant):
        return node.value

    elif isinstance(node, ast.BinOp):
        left = evaluate(node.left)
        right = evaluate(node.right)

        op = type(node.op)

        if op not in OPERATORS:
            raise ValueError("Operator not supported.")

        return OPERATORS[op](left, right)

    else:
        raise ValueError("Invalid expression.")

def calculator(expression):
    tree = ast.parse(expression, mode="eval")
    return evaluate(tree.body)

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

def search_web(query):

    with DDGS() as ddgs:
        results = list(
            ddgs.text(
                query,
                max_results=5,
            )
        )

    return results

# ------------------------------------
# Tool Registry
# ------------------------------------
tool_functions = {
    "calculator": calculator,
    "get_weather": get_weather,
    "search_web": search_web,
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
     },
     {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate mathematical expressions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Example: 25*8+100/5"
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for recent or factual information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
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
            " Use the calculator tool for mathematical calculations."
            " Use the search_web tool whenever the user asks for current events, news, or information that requires searching the web."
            " When a tool returns information, base your answer on the tool output rather than your own memory."
        ),
    }
]

print("=" * 50)
print("🤖 Multi-Tool Agent")
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

        tool = tool_functions.get(function_name)
        if tool is None:
            print(f"Unknown tool requested: {function_name}")
            continue

        print(f"\n🛠 Calling {function_name} with {arguments}")
        result = tool(**arguments)
        print("\n📦 Tool Result")
        if isinstance(result, (dict, list)):
            print(json.dumps(result, indent=4))
        else:
            print(result)
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
