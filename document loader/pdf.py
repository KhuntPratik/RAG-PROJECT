from langchain_community.document_loaders import PyPDFLoader


data = PyPDFLoader(r"E:\Gen Ai\RAG PROJECT\document loader\GRU.pdf")

docs = data.load()

print(docs[0])