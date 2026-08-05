# KnowledgeForge

## Personal Knowledge Base powered by AI

KnowledgeForge is an AI-powered Personal Knowledge Base designed to help users store, search, and interact with their own knowledge using Large Language Models (LLMs).

The vision is to build an AI Second Brain where users can upload their documents, technical notes, research material, books, and personal knowledge, and then ask questions in natural language to retrieve meaningful answers from their own information.

KnowledgeForge is being built as a learning journey towards building production-grade AI applications, with the long-term goal of evolving into a useful and monetizable AI knowledge assistant.

---

# Current Status

## KnowledgeForge v0.1 — Working RAG Knowledge Assistant

The first milestone has been completed: an end-to-end Retrieval-Augmented Generation (RAG) pipeline.

The system can:

* Load documents
* Split documents into meaningful chunks
* Generate semantic embeddings
* Store embeddings in a vector database
* Retrieve relevant information using similarity search
* Use retrieved context to generate accurate answers using an LLM

The current implementation has been tested with RFC documents and successfully answers questions based only on the indexed knowledge base.

Example:

```
Question:
What is Time to Live?

Answer:
Time to Live (TTL) is an indication of an upper bound on the lifetime
of an internet datagram...
```

---

# Architecture

```
                 Documents
                    |
                    |
                    v
             Document Loader
                    |
                    |
                    v
              Text Chunking
                    |
                    |
                    v
             Embedding Model
                    |
                    |
                    v
              ChromaDB Vector Store
                    |
                    |
                    v
              Semantic Retrieval
                    |
                    |
                    v
              Context + Question
                    |
                    |
                    v
                 LLM
                    |
                    |
                    v
              Generated Answer
```

---

# Current Features

## Document Processing

Supported:

* PDF documents
* Markdown files

Pipeline:

* Document loading
* Text splitting
* Embedding generation
* Vector indexing

---

## Semantic Search

KnowledgeForge uses vector embeddings to understand the meaning of queries instead of relying only on keyword matching.

Example:

A user can ask:

```
Explain IP packet lifetime
```

and retrieve information related to:

```
Time To Live (TTL)
```

even if the exact words are different.

---

## Retrieval-Augmented Generation (RAG)

The system follows a RAG architecture:

1. Retrieve relevant knowledge from the user's documents
2. Provide the retrieved context to the LLM
3. Generate an answer grounded in the provided information

This reduces hallucination and keeps answers connected to the user's knowledge base.

---

# Technology Stack

## Language

* Python 3.11

## AI / ML

* Large Language Models (LLMs)
* Sentence Transformers
* Hugging Face Router

## Vector Database

* ChromaDB

## APIs / Libraries

* OpenAI Python SDK
* Transformers
* PyTorch

---

# Project Structure

```
knowledgeforge/

├── docs/
│   └── User documents

├── ingest.py
│   └── Document ingestion pipeline

├── chat.py
│   └── Query and answer generation

├── embeddings.py
│   └── Embedding generation

├── vectordb.py
│   └── ChromaDB integration

├── config.py
│   └── Application configuration

└── chroma_db/
    └── Generated vector database
```

---

# Running KnowledgeForge

## Create environment

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Configure environment variables

Set Hugging Face token:

```bash
export HF_TOKEN=<your_token>
```

## Add documents

Place documents inside:

```
docs/
```

## Index documents

```bash
python ingest.py
```

## Ask questions

```bash
python chat.py
```

---

# Roadmap

## KnowledgeForge v0.2

Improve reliability and user experience:

* Source citations
* Document metadata
* Page-level references
* Better retrieval strategies
* Improved prompts
* Support more document formats

---

## KnowledgeForge v0.3

Move towards a real application:

* Web interface
* Document upload
* User accounts
* Conversation history
* Multiple knowledge bases
* Background document processing

---

## Future Vision

KnowledgeForge aims to become a personal and professional AI knowledge assistant.

Potential use cases:

### Individuals

* Personal notes assistant
* Research assistant
* Book/document understanding
* Personal AI second brain

### Engineers

* Search technical documentation
* Understand RFCs
* Query architecture documents
* Maintain engineering knowledge

### Teams

* Internal documentation assistant
* Product knowledge base
* Engineering support assistant

---

# Product Vision

The long-term goal is to build a private AI assistant that helps users access and use their accumulated knowledge efficiently.

Instead of searching through hundreds of documents manually, users should be able to simply ask:

```
What do I know about this topic?
```

and receive a reliable answer from their own knowledge base.

---

# License

This project is currently developed as an experimental AI product prototype.

