# HF Chat Agent

A step-by-step AI Agent project built while learning modern Large Language Model (LLM) application development.

The goal of this repository is to understand how AI agents are built from first principles by implementing one concept at a time instead of relying on high-level frameworks.

Each script introduces a new capability, gradually evolving from a simple chatbot into a multi-tool AI agent capable of performing calculations, weather lookup, web search.

This repository is part of my hands-on journey to understand how modern AI agents work by implementing each capability from scratch before moving to more advanced topics such as Retrieval-Augmented Generation (RAG).

---

# Features

* Basic chat application using the Hugging Face Router
* Support for multiple LLMs
* Calculator tool
* Weather tool using Open-Meteo APIs
* Web Search tool
* Multi-tool AI Agent

---

# Project Structure

```
hf-chat-agent/

chat.py
    Basic chat application

chat_multiple_models.py
    Chat with multiple LLMs

calculator_agent.py
    Function calling example with calculator tool

weather_agent.py
    Weather tool using Open-Meteo API

multitool_agent.py
    Calculator + Weather tools

multitool_agent_calc_weather_search.py
    Calculator + Weather + Web Search tools

weather_test.py
    Weather API testing

```

---

# Technologies Used

* Python 3.11
* Hugging Face Router
* OpenAI Python SDK
* DeepSeek V3
* GPT-OSS
* Qwen
* Llama
* Open-Meteo API
* DDGS (DuckDuckGo Search)

---

# Installation

Clone the repository

```bash
git clone https://github.com/alokgautam123/hf-chat-agent.git
cd hf-chat-agent
```

Create a virtual environment

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Set your Hugging Face token before running the examples.

Linux/macOS

```bash
export HF_TOKEN=your_huggingface_token
```

Windows

```cmd
set HF_TOKEN=your_huggingface_token
```

---

# Running the Examples

Basic Chat

```bash
python chat.py
```

Multiple Models

```bash
python chat_multiple_models.py
```

Calculator Agent

```bash
python calculator_agent.py
```

Weather Agent

```bash
python weather_agent.py
```

Multi-Tool Agent

```bash
python multitool_agent.py
```

Multi-Tool Agent with Web Search

```bash
python multitool_agent_calc_weather_search.py
```
---

# Learning Journey

This repository was intentionally built incrementally.

The progression followed is:

1. Basic Chat Agent
2. Multiple Model Support
3. Calculator Agent (Function Calling) 
4. Weather Agent 
5. Multi-Tool Agent
6. Multi-Tool Agent including Web Search Tool

Each step introduces one new concept while keeping the code easy to understand.

---

# Acknowledgements

This project uses:

* Hugging Face Router
* OpenAI Python SDK
* Open-Meteo API
* ChromaDB
* Sentence Transformers
* DDGS

Special thanks to the open-source community for providing excellent tools and libraries that make learning AI application development accessible.

---

# License

This project is intended for learning and educational purposes.

