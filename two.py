import os
import sys

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not set. Add it to your .env file.")

hf_repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
DB_FAISS_PATH = "vectorstore/db_faiss"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def load_llm(hf_repo_id):
    endpoint = HuggingFaceEndpoint(
        repo_id=hf_repo_id,
        task="conversational",
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN,
        provider="auto"
    )
    return ChatHuggingFace(llm=endpoint, max_tokens=512, temperature=0.5)

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate.from_template(custom_prompt_template)
    return prompt

if not os.path.exists(DB_FAISS_PATH):
    raise FileNotFoundError(
        f"FAISS vector store not found at '{DB_FAISS_PATH}'. Run the ingestion script first."
    )

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(hf_repo_id),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

user_query = input("Write Query Here: ")
if not user_query.strip():
    print("Query cannot be empty.")
    sys.exit(1)

response = qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])
