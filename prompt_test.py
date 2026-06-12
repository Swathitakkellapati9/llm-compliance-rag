query = "How long should customer records be stored?"

context = """
Customer records must be retained for 7 years.
Employees must complete compliance training annually.
"""

prompt = f"""
You are a compliance assistant.

Answer only from the provided context.

Context:
{context}

Question:
{query}
"""

print(prompt)