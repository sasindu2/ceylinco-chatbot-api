from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

app = Flask(__name__)
CORS(app)

# Load PDF
loader = PyPDFLoader("data/ceylinco2024.pdf")
pages = loader.load_and_split()

# Create vector DB
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
texts = text_splitter.split_documents(pages)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

# Create chain
llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_template(
    "Answer the question using the following context:\n\n{context}\n\nQuestion: {question}"
)
chain = create_stuff_documents_chain(llm, prompt)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "")

    if question.lower() in ["hi", "hello"]:
        return jsonify({"response": "👋 Hello! I'm your Ceylinco AI assistant. Ask me anything about the 2024 annual report!"})

    docs = db.similarity_search(question)
    result = chain.invoke({"context": docs, "question": question})
    return jsonify({"response": result})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
