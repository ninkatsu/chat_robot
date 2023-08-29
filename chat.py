import os
import openai
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
#数据集目录
directory = os.getenv("DIRECTORY_PATH")
#pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
index_name = os.getenv("INDEX_NAME")
#openai
openai_api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("MODEL_NAME")


def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

documents = load_docs(directory)
# print("文档内容"+ documents)
# print("文档长度："+len(documents))

docs = split_docs(documents)
# print(len(docs))


os.environ["http_proxy"] = "http://127.0.0.1:7078"
os.environ["https_proxy"] = "http://127.0.0.1:7078"
# os.environ["OPENAI_API_TYPE"] = "azure"
# os.environ["OPENAI_API_BASE"] = "https://<your-endpoint.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = openai_api_key
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# query_result = embeddings.embed_query("元灵数智")
# print(len(query_result))


pinecone.init(
    api_key=pinecone_api_key,
    environment=pinecone_environment
)

# index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
# if you already have an index, you can load it like this
index = Pinecone.from_existing_index(index_name, embeddings)

def get_similiar_docs(query, k=2, score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query, k=k)
  else:
    similar_docs = index.similarity_search(query, k=k)
  return similar_docs

llm = OpenAI(model_name=model_name)

chain = load_qa_chain(llm, chain_type="stuff")

def get_answer(query):
  similar_docs = get_similiar_docs(query)
  answer = chain.run(input_documents=similar_docs, question=query)
  return answer


# query = "介绍任罡的学术背景"
# answer = get_answer(query)
# print(query)
# print(answer)
print("----------------------------------------------------------------")

query = "我想知道任罡的完整工作经历"
answer = get_answer(query)
print(query)
print(answer)


