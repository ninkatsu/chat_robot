import pinecone,os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
#数据集目录
directory = os.getenv("DIRECTORY_PATH")
#pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
index_name = os.getenv("INDEX_NAME")

pinecone.init(
    api_key=pinecone_api_key,
    environment=pinecone_environment
)
print(pinecone.list_indexes())
