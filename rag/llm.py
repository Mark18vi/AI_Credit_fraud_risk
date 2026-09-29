from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_llm():
    # Fetch API key from environment or .env
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Please add it to your .env file or environment variables.")
        
    return ChatOpenAI(
        openai_api_key=api_key,
        model="gpt-4o-mini", 
        temperature=0.5,
        max_tokens=500
    )

def generate_answer(question, context):
    try:
        llm = get_llm()
        
        # Construct the prompt
        system_prompt = (
            "You are an expert Financial Fraud and Compliance Assistant. "
            "Use the provided context to answer the question accurately. "
            "If the answer isn't in the context, state that you don't know. "
            "Keep the answer professional and concise."
        )
        
        full_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=full_prompt)
        ]
        
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"API Error: {str(e)}"

if __name__ == "__main__":
    print("LLM module created.")
