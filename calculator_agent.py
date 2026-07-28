import os
import json
import ast
import operator

from openai import OpenAI

# ------------------------------------
# Hugging Face Client
# ------------------------------------

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)
# LLM used for tool calling
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


# ------------------------------------
# Tool Definition
# ------------------------------------

tools = [
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
            " Use the calculator tool whenever mathematical computation is needed."
        ),
    }
]

print("=" * 50)
print("🤖 Function Calling Agent")
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
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )
    except Exception as e:
        print(e)

    message = response.choices[0].message

    # ------------------------------------
    # Did the model call a tool?
    # ------------------------------------

    if message.tool_calls:

        tool_call = message.tool_calls[0]

        function_name = tool_call.function.name

        arguments = json.loads(tool_call.function.arguments)

        if function_name == "calculator":

            expression = arguments["expression"]

            print(f"\n🛠 Calling calculator({expression})")

            result = calculator(expression)

            print(f"🧮 Result = {result}\n")

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
