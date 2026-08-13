def build_qa_prompt(question, context):
    return f"""
You are an expert assistant.

Answer using ONLY the provided context.

If the information needed to answer the question is NOT present in the context,
reply with exactly:

INSUFFICIENT_CONTEXT

Do not include citations or any other text when returning INSUFFICIENT_CONTEXT.

For valid answers, include citations in this format:

Sources:
- <filename>
- Page <page number>

Context:

{context}

Question:

{question}

Answer:
"""
