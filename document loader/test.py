from langchain_community.document_loaders import TextLoader

data = TextLoader(r"E:\Gen Ai\RAG PROJECT\document loader\notes.txt")

docs = data.load() 

print(docs)