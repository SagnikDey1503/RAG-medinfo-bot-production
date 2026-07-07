from langchain_community.document_loaders import PyPDFLoader , DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
#loading raw pdfs

path ="data/"
def load_pdfs(path):
    loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdfs(path)
print(f"Loaded {len(documents)} documents.")
# print(documents[0])
#creat chunks for limited context of llm
def create_chunks (extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(extracted_data)
    return chunks

text_chunks = create_chunks(extracted_data=documents)
print(f"Created {len(text_chunks)} text chunks.")
#embedding the chunks

def get_embedding_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    return embedding_model
# 364 dimensional vector and understand context too

embedding_model = get_embedding_model()


#store embeddings in vector database  FAISS

db_path="vectorstore/db_faiss"

db=FAISS.from_documents(text_chunks, embedding_model)
db.save_local(db_path)
print(f"Embeddings stored in FAISS vector database at {db_path}.")