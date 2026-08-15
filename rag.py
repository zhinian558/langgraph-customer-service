import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_text_splitters import CharacterTextSplitter

# ---------- 1. 加载知识库文档 ----------
def load_knowledge_base(file_path: str = "knowledge_base/return_policy.txt"):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    # 按段落切分，每段最大 200 字符，不重叠
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=0
    )
    return text_splitter.split_documents(documents)

# ---------- 2. 向量库接口（预留切换 Milvus） ----------
def get_vectorstore(embedding_model=None):
    """
    返回向量存储实例。
    当前使用 Chroma 内存模式。
    如需切换 Milvus，只需替换此函数实现：
    from langchain_milvus import Milvus
    return Milvus.from_documents(..., connection_args={"host": "localhost", "port": "19530"})
    """
    if embedding_model is None:
        # 使用 HuggingFace 轻量 embedding，首次运行会自动下载模型
        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5"
        )

    # Chroma 内存模式：数据存在当前进程内，重启后消失，适合 MVP 演示
    docs = load_knowledge_base()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        collection_name="return_policy",
        persist_directory=None,  # None 表示不持久化，纯内存
    )
    return vectorstore

# ---------- 3. 定义检索工具 ----------
_vectorstore = get_vectorstore()  # 全局初始化一次，避免重复加载

@tool
def retrieve_return_policy(query: str) -> str:
    """
    检索退货政策知识库工具。
    输入：query，用户关于退货政策的问题。
    输出：最相关的政策片段。
    """
    results = _vectorstore.similarity_search(query, k=1)
    if results:
        return results[0].page_content
    return "未找到相关的退货政策。"