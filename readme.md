智能对话项目 chat.py
该项目是一个使用 OpenAI GPT-3.5 语言模型进行问答的智能对话系统。
可以将新的领域知识/content/data（pdf格式），将其转化为向量存入带有索引的向量数据库，此过程考虑到token长度，将其进行chunk化并设置重叠长度（向量间相关度）
当用户提出问题时，采用同样的openai提供的embedding方式将其带入数据库检索，并返回相应的向量答案灌入GPT生成回答。

以下是该项目的说明：

    依赖项安装：

    Python 3.x
    openai 
    pinecone 
    langchain 
    dotenv 
    可以使用以下命令安装这些依赖项：

    pip install -r requirements.txt

    配置环境变量
    在运行代码之前，需要配置一些环境变量。可以创建一个名为 .env 的文件，并在其中添加以下内容：

    plaintext
    Copy
    DIRECTORY_PATH=/path/to/dataset/directory
    PINECONE_API_KEY=<your-pinecone-api-key>
    PINECONE_ENVIRONMENT=<your-pinecone-environment>
    INDEX_NAME=<your-index-name>
    OPENAI_API_KEY=<your-openai-api-key>
    MODEL_NAME=<your-model-name>
    确保将上述 <your-...> 替换为相应的值。

    加载数据集
    在代码中，首先从指定的目录中加载数据集。使用 load_docs 函数加载数据集，并使用 split_docs 函数将文档拆分为较小的块。

    初始化 OpenAI 模型和 Pinecone 索引
    使用 OpenAIEmbeddings 类初始化 OpenAI 文本嵌入模型，并使用 Pinecone 类初始化 Pinecone 索引。确保在代码中设置正确的 OpenAI 模型和 Pinecone 索引的名称。

    获取相似文档
    使用 get_similiar_docs 函数可以根据查询获取相似的文档。可以指定返回的文档数量和是否返回相似度分数。

    运行问答链
    使用 load_qa_chain 函数加载问答链，并使用 get_answer 函数根据查询获取答案。可以将查询作为输入，然后得到与查询相关的文档，并使用问答链生成答案。

    运行代码
    在代码的最后，可以取消注释并运行 get_answer 函数来获取答案。

    python
    Copy
    query = "我想知道xx的完整工作经历"
    answer = get_answer(query)
    print(query)
    print(answer)

注意事项

在运行代码之前，确保已经正确配置了环境变量，并且提供了正确的数据集目录、Pinecone API 密钥、Pinecone 环境、索引名称、OpenAI API 密钥和模型名称。
请根据项目实际需求对代码进行适当的修改和调整。
可以根据需要取消注释并打印其他变量和结果，以便进行调试和验证。