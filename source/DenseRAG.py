import os
import json
import numpy as np
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
from sklearn.metrics.pairwise import cosine_similarity


class VllmDenseRAGPipeline:
    def __init__(self,documents,language,cache_path,vllm_url="http://localhost:8000/v1",max_worker=100):
        """
        初始化 基于 vLLM 的 Dense RAG Pipeline
        :param documents: 文档列表
        :param language: 语言类型 ('python', 'en', 'zh')
        :param cache_path: 缓存文件路径 (如: 'cache_codenet.pkl')，非常重要！不同数据集请使用不同名字
        :param vllm_url: vLLM 提供服务的 url
        :param max_worker: 并发线程数
        """
        self.language=language
        self.cache_path=cache_path
        self.url=vllm_url
        self.max_worker=max_worker

        # 核心逻辑：先看有没有算好的缓存，有的话直接秒速读取，没有才计算
        if os.path.exists(self.cache_path):
            with open(self.cache_path,'rb') as f:
                cache_data=pickle.load(f)
            self.chunks=cache_data['chunks']
            self.chunk_embeddings=cache_data['embeddings']
        else:
            # --- 1. 文本切分模块（保留你的原逻辑） ---
            if language=="python":
                self.text_splitter=RecursiveCharacterTextSplitter.from_language(
                    language=Language.PYTHON,
                    chunk_size=200,
                    chunk_overlap=0
                )
            elif language=="en":
                self.text_splitter=RecursiveCharacterTextSplitter(
                    chunk_size=300,
                    chunk_overlap=50,
                    length_function=len,
                    separators=["\n\n","\n",".",";",","," ",""]
                )
            elif language=="zh":
                self.text_splitter=RecursiveCharacterTextSplitter(
                    chunk_size=300,
                    chunk_overlap=50,
                    length_function=len,
                    separators=["\n\n","\n","。","；","，"," ",""]
                )
            else:
                raise ValueError("Language Error")

            raw_chunks=[]
            for doc in documents:
                chunks=self.text_splitter.split_text(doc)
                raw_chunks+=chunks

            # 去重处理
            self.chunks=list(set(raw_chunks))
            # --- 2. 批量计算并保存向量 ---
            embeddings_list=self.vllm_embedding(self.chunks)
            self.chunk_embeddings=np.array(embeddings_list)  # 转换为 numpy array 方便后续矩阵运算

            # 保存到缓存文件
            with open(self.cache_path,'wb') as f:
                pickle.dump({
                    'chunks':self.chunks,
                    'embeddings':self.chunk_embeddings
                },f)
            print(f"Embeddings computed and saved to {self.cache_path}.")

    def vllm_embedding(self,text_list):
        """调用本地 vLLM 获取文本 list 的 embedding"""
        from openai import OpenAI
        from concurrent.futures import ThreadPoolExecutor

        client=OpenAI(base_url=self.url,api_key="none")
        model_name="Qwen3-Embedding-8B"  # 确保与你启动的 --served-model-name 一致

        def get_embedding(text):
            try:
                response=client.embeddings.create(
                    model=model_name,
                    input=text,
                    encoding_format="float"
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"[Warn] vLLM Request Failed for text block. Err: {e}")
                # 遇到错误返回一个全零向量占位，防止整个流程崩溃（假设维度为4096或8192）
                return None

        with ThreadPoolExecutor(max_workers=self.max_worker) as executor:
            embedding_list=list(executor.map(get_embedding,text_list))

        # 安全检查：检查是否有请求失败（None），如果有则补齐0向量（或者抛出错误）
        dim=len([emb for emb in embedding_list if emb is not None][0])
        safe_embedding_list=[emb if emb is not None else [0.0]*dim for emb in embedding_list]

        return safe_embedding_list

    def query(self,question: str,top_k: int = 3):
        """进行检索查询响应"""
        # 1. 对 query 直接计算一次 embedding，由于只有一条文本，速度极快
        query_emb_list=self.vllm_embedding([question])
        # 转换为二维数组形状 (1, D) 以适配 sklearn 计算要求
        query_vec=np.array(query_emb_list).reshape(1,-1)

        # 2. 计算与所有 database chunks 的相似度
        similarities=cosine_similarity(query_vec,self.chunk_embeddings).flatten()

        # 3. 提取最高分索引
        top_indices=similarities.argsort()[-top_k:][::-1]

        return [self.chunks[idx] for idx in top_indices]


if __name__=="__main__":

    # 用你现有的数据测试一把
    with open(r"/nucleus/data/hmUs/reference.json","r",encoding="utf-8") as file:
        documents=json.load(file)

    # 注意：为不同数据集建立不同的 cache_path! 这很重要！
    # 第一次运行此代码会比较慢，因为在调用 vLLM
    # 第二次及以后运行，会直接加载 cache_peopledaily_en.pkl，瞬间完成
    rag=VllmDenseRAGPipeline(
        documents,
        language="zh",  # 中文使用 zh 规则切分
        cache_path="cache_hmUs_zh.pkl",
        vllm_url="http://localhost:8000/v1",
        max_worker=50
    )

    result=rag.query(
        "胆囊壁毛糙，胆囊内可见数个强回声，其一大小约12×10mm，后伴声影，可随体位改变移动。",
        top_k=5
    )

    for i,item in enumerate(result):
        print(f"--- Top {i+1} ---")
        print(item)