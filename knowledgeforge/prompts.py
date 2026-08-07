def build_qa_prompt(question, context):
    return f"""
You are an expert assistant.

Answer using ONLY the provided context.

For every answer, include citations in this format:

Sources:
- <filename>
- Page <page number>

If information is not present, reply:

I couldn't find that information in the provided documents.

Context:

{context}

Question:

{question}

Answer:
"""
