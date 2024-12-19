from flask import Flask, request, jsonify
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Initialize Flask app
app = Flask(__name__)

# Initialize AI model and chain
template = """
Answer the questions below.

Here is the conversation history: {context}

Question: {question}

Answer: {answer}
"""
model = OllamaLLM(model="llama3.3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Keep conversation context
context = ""

@app.route("/ask", methods=["POST"])
def ask_question():
    global context
    data = request.json
    user_question = data.get("question", "")
    
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    # Get the AI response
    result = chain.invoke({"context": context, "question": user_question})
    context += f"\nUser: {user_question}\nAI: {result}"

    return jsonify({"answer": result})

if __name__ == "__main__":
    app.run(port=5000)
