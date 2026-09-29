def get_rag_prompt(question, context):
    return f"""You are an expert Financial Fraud and Compliance Assistant. 
Use the following pieces of retrieved context to answer the user's question. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

if __name__ == "__main__":
    print("Prompt module created.")
