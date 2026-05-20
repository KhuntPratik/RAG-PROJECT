from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(page_content="This is the first document.", metadata={"source": "doc1"}),
    Document(page_content="This is the second document.", metadata={"source": "doc2"}),
    Document(page_content="This is the third document.", metadata={"source": "doc3"})
]

# Embedding model
embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)


result = vectorstore.similarity_search("What is the content of the first document?", k=2)

retriver = vectorstore.as_retriever()

docs = retriver.invoke("What is the content of the first document?", k=2)

