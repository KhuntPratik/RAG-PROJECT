from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv

load_dotenv()


data = PyPDFLoader(r"E:\Gen Ai\RAG PROJECT\document loader\GRU.pdf")

docs = data.load()

template = ChatPromptTemplate.from_messages([("system", "You are a summarize teat."),("human", "{data}")])  

llm = ChatMistralAI(model = "mistral-small-2506")

prompt = template.format_prompt(data = docs)

result = llm.invoke(prompt)

print(result.content)