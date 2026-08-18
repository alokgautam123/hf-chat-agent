from services.qa import answer_question

question = input("Ask a question: ")
result = answer_question(question, "legacy")

print("\nAnswer:\n")
print(result["answer"])
