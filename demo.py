#加载环境变量
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# 为了查询聊天模型GPT-3.5-turbo或GPT-4，导入聊天消息和ChatOpenAI的模式（schema）。
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
messages = [
    SystemMessage(content="你是一个专业的数据科学家"),
    HumanMessage(content="写一个Python脚本，用模拟数据训练一个神经网络")
]

response = chat(messages)
print(response.content, end='\n')

# 实际上，SystemMessage为GPT-3.5-turbo模块提供了每个提示-完成对的上下文信息。HumanMessage是指您在ChatGPT界面中输入的内容，也就是您的提示。
# 但是对于一个自定义知识的聊天机器人，我们通常会将提示中重复的部分抽象出来。例如，如果我要创建一个推特生成器应用程序，
# 我不想一直输入“给我写一条关于…的推特”。因此，让我们来看看如何使用提示模板（PromptTemplates）来将这些内容抽象出来。

# 导入提示并定义PromptTemplate
from langchain.prompt_templates import PromptTemplate

template = """您是一位专业的数据科学家，擅长构建深度学习模型。用几行话解释{concept}的概念"""
prompt = PromptTemplate(input_variables=["concept"], template=template)

# 用PromptTemplate运行LLM
llm(prompt.format(concept="autoencoder"))
llm(prompt.format(concept="regularization"))

# 导入LLMChain并定义一个链，用语言模型和提示作为参数。
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)  # 只指定输入变量来运行链。
print(chain.run("autoencoder"))

# 除此之外，顾名思义，我们还可以把这些链连起来，创建更大的组合。
# 比如，我可以把一个链的结果传递给另一个链。在这个代码片段里，Rabbitmetrics把第一个链的完成结果传递给第二个链，让它用500字向一个五岁的孩子解释。

# 定义一个第二个提示
second_prompt = PromptTemplate(input_variables=["ml_concept"], template="把{ml_concept}的概念描述转换成用500字向我解释，就像我是一个五岁的孩子一样")

chain_two = LLMChain(llm=llm, prompt=second_prompt)

# 用上面的两个链定义一个顺序链：第二个链把第一个链的输出作为输入
from langchain.chains import SimpleSequential, Chain

overall_chain = SimpleSequential(Chain(chains=[chain, chain_two], verbose=True))

# 只指定第一个链的输入变量来运行链。
explanation = overall_chain.run("autoencoder")
print(explanation)

# 导入分割文本的工具，并把上面给出的解释分成文档块。
# 分割文本需要两个参数：每个块有多大（chunksize）和每个块有多少重叠（chunkoverlap）。
# 让每个块之间有重叠是很重要的，可以帮助识别相关的相邻块。
# 每个块都可以这样获取：texts[0].page_content

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
