from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
template = """
Answer the questions below.
Here is the conversation history: {context}
Question: {question}
Answer: {answer}
"""
model = OllamaLLM(model="llama3.3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
def handle_conversation():
    context = ""
    print("Welcome, How may I help you?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break  
        # Fix: Proper dictionary passed to chain.invoke
        result = chain.invoke({"context":context, "question": user_input})
        print("Bot:", result)
        # Append to conversation history
        context += f"\nUser: {user_input}\nAI: {result}"
if __name__ == "__main__":
    handle_conversation()