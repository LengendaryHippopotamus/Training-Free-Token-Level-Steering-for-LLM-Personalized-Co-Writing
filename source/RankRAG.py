import os
import json
import numpy as np
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
from sklearn.metrics.pairwise import cosine_similarity
# 新增：导入用于重排的交叉编码器
from sentence_transformers import CrossEncoder


class VllmRankRAGPipeline:
    def __init__(self,documents,language,cache_path,vllm_url="http://localhost:8000/v1",max_worker=100,
        use_reranker=True,
        reranker_model="/llm/bge-reranker-v2-m3/"
        ):
        """
        初始化 基于 vLLM 的 Dense RAG Pipeline (支持 Rerank 精排)
        :param documents: 文档列表
        :param language: 语言类型 ('python', 'en', 'zh')
        :param cache_path: 缓存文件路径
        :param vllm_url: vLLM 提供服务的 url
        :param max_worker: 并发线程数
        :param use_reranker: 是否启用重排机制 (应对审稿人要求)
        :param reranker_model: 重排模型的 HuggingFace 路径或本地路径
        """
        self.language=language
        self.cache_path=cache_path
        self.url=vllm_url
        self.max_worker=max_worker

        # --- 新增：初始化 Reranker 模型 ---
        self.use_reranker=use_reranker
        if self.use_reranker:
            # print(f"Loading Reranker model: {reranker_model} ...")
            # 自动调用 GPU (如果可用)，对中文医疗数据推荐 bge-reranker-v2-m3
            self.reranker=CrossEncoder(reranker_model,max_length=8192)
            # print("Reranker loaded successfully.")

        # 核心逻辑：先看有没有算好的缓存，有的话直接秒速读取，没有才计算
        if os.path.exists(self.cache_path):
            with open(self.cache_path,'rb') as f:
                cache_data=pickle.load(f)
            self.chunks=cache_data['chunks']
            self.chunk_embeddings=cache_data['embeddings']
        else:
            # --- 1. 文本切分模块 ---
            if language=="python":
                self.text_splitter=RecursiveCharacterTextSplitter.from_language(
                    language=Language.PYTHON,chunk_size=200,chunk_overlap=0
                )
            elif language=="en":
                self.text_splitter=RecursiveCharacterTextSplitter(
                    chunk_size=300,chunk_overlap=50,length_function=len,
                    separators=["\n\n","\n",".",";",","," ",""]
                )
            elif language=="zh":
                self.text_splitter=RecursiveCharacterTextSplitter(
                    chunk_size=300,chunk_overlap=50,length_function=len,
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
            self.chunk_embeddings=np.array(embeddings_list)

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
        model_name="Qwen3-Embedding-8B"

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
                return None

        with ThreadPoolExecutor(max_workers=self.max_worker) as executor:
            embedding_list=list(executor.map(get_embedding,text_list))

        dim=len([emb for emb in embedding_list if emb is not None][0])
        safe_embedding_list=[emb if emb is not None else [0.0]*dim for emb in embedding_list]

        return safe_embedding_list

    def query(self,question: str,top_k: int = 3,recall_k: int = 20):
        """
        进行检索查询响应 (支持两阶段检索)
        :param question: 用户查询
        :param top_k: 最终返回的精准文档数量
        :param recall_k: 第一阶段粗排召回的文档数量 (必须大于或等于 top_k)
        """
        # --- 阶段 1：Dense Retrieval 粗排 ---
        query_emb_list=self.vllm_embedding([question])
        query_vec=np.array(query_emb_list).reshape(1,-1)

        similarities=cosine_similarity(query_vec,self.chunk_embeddings).flatten()

        # 提取前 recall_k 个候选文本 (例如前 20 个)
        top_indices=similarities.argsort()[-recall_k:][::-1]
        recalled_chunks=[self.chunks[idx] for idx in top_indices]

        # --- 阶段 2：Rerank 精排 ---
        if self.use_reranker:
            # 构造 (Query, Document) 对
            cross_inp=[[question,chunk] for chunk in recalled_chunks]

            # 使用 Cross-encoder 计算精准相似度得分
            rerank_scores=self.reranker.predict(cross_inp)

            # 根据 Reranker 的得分重新排序
            rerank_indices=np.argsort(rerank_scores)[::-1][:top_k]

            # 返回精排后的 top_k 结果
            final_results=[recalled_chunks[idx] for idx in rerank_indices]
            return final_results
        else:
            # 如果不使用 reranker，退化为原始的一阶段检索
            return recalled_chunks[:top_k]


if __name__=="__main__":

    with open(r"/nucleus/data/hmUs/reference.json","r",encoding="utf-8") as file:
        documents=json.load(file)

    rag=VllmRankRAGPipeline(
        documents,
        language="zh",
        cache_path="cache_hmUs_zh.pkl",
        vllm_url="http://localhost:8000/v1",
        max_worker=50,
        use_reranker=True  # 开启重排增强 Baseline
    )

    # 第一阶段先召回 20 个，第二阶段重排后精准输出前 5 个
    result=rag.query(
        "胆囊壁毛糙，胆囊内可见数个强回声，其一大小约12×10mm，后伴声影，可随体位改变移动。",
        top_k=5,
        recall_k=20
    )

    for i,item in enumerate(result):
        print(f"--- Top {i+1} ---")
        print(item)