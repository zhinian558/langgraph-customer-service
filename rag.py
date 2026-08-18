import os
import json
from typing import List
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_text_splitters import CharacterTextSplitter
from rank_bm25 import BM25Okapi
import jieba
from sentence_transformers import CrossEncoder

# ---------- 1. 加载知识库文档 ----------
def load_knowledge_base(file_path: str = "knowledge_base/return_policy.txt"):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=0
    )
    return text_splitter.split_documents(documents)

# ---------- 2. 向量库接口（预留切换 Milvus） ----------
def get_vectorstore(embedding_model=None):
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5"
        )
    docs = load_knowledge_base()
    # 使用持久化目录
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        collection_name="return_policy",
        persist_directory="./chroma_db",  # 持久化
    )
    return vectorstore, docs

# ---------- 3. 全局初始化（模块加载时执行一次） ----------
_vectorstore, _docs = get_vectorstore()

# 提取文档文本列表（用于 BM25）
_doc_texts = [doc.page_content for doc in _docs]

# 使用 jieba 分词构建 BM25 索引
_tokenized_docs = [list(jieba.cut(text)) for text in _doc_texts]
_bm25 = BM25Okapi(_tokenized_docs)

# 加载重排序模型（Cross-Encoder）
_reranker = CrossEncoder(
    "BAAI/bge-reranker-base",
    max_length=512
)

# ---------- 4. 混合检索 + 重排序 ----------
def _hybrid_search(query: str, top_k: int = 5) -> List[str]:
    """
    执行混合检索：向量检索 + BM25，合并后去重，返回候选文档文本列表。
    """
    # 向量检索
    vector_results = _vectorstore.similarity_search(query, k=top_k)
    vector_texts = [doc.page_content for doc in vector_results]

    # BM25 检索
    tokenized_query = list(jieba.cut(query))
    bm25_scores = _bm25.get_scores(tokenized_query)
    # 获取 top_k 索引
    top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    bm25_texts = [_doc_texts[i] for i in top_indices]

    # 合并去重（保持顺序）
    combined = []
    for text in vector_texts + bm25_texts:
        if text not in combined:
            combined.append(text)
    return combined

def _rerank_documents(query: str, candidate_texts: List[str]) -> str:
    """
    使用 Cross-Encoder 对候选文档进行重排序，返回最相关的文档文本。
    """
    if not candidate_texts:
        return "未找到相关的退货政策。"

    # 构造 (query, doc) 对
    pairs = [(query, doc) for doc in candidate_texts]
    scores = _reranker.predict(pairs)

    # 取得分最高的索引
    best_idx = scores.argmax()
    return candidate_texts[best_idx]

# ---------- 5. 定义检索工具 ----------
@tool
def retrieve_return_policy(query: str) -> str:
    """
    检索退货政策知识库工具。
    输入：query，用户关于退货政策的问题。
    输出：最相关的政策片段。
    """
    candidates = _hybrid_search(query, top_k=5)
    best_doc = _rerank_documents(query, candidates)
    return best_doc