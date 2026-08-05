import os
import json
import numpy
import torch
from source.Experiment import Haic_component

class haic_qwen_eval():
    def __init__(
        self,
        file_path,
        sampling_num,
        guide_method,
        repeat=1,
        url="http://localhost:8000/v1",
        max_worker=2048
    ):
        self.file_path=file_path
        self.sampling_num=sampling_num
        # self.sampling_num=100
        self.guide_method=guide_method
        self.repeat=repeat
        self.url=url
        self.max_worker=max_worker
        if self.guide_method=="CoTdecode":
            pdiff=torch.load(file_path+'CoTdecodelog.pt',map_location=torch.device('cpu')).reshape(self.sampling_num,self.repeat)
            _,self.indices=torch.cummax(pdiff,dim=1)

    def save_score(self):
        numpy.savez(self.file_path+'score.npz',**self.score)
        print(self.score['Qwen3embd'])
    def load_socre(self):
        if not os.path.exists(self.file_path+"score.npz"):
            self.score=dict()
        else:
            data=numpy.load(self.file_path+'score.npz',allow_pickle=True)
            self.score=dict(data)
            data.close()

    def Qwen_embedding_line(self):
        self.load_socre()
        with open(self.file_path+'Qwen3embd_log.json','r',encoding='utf-8') as f:
            data=json.load(f)  # 读取 JSON 数据，假设是一个 list
        # data=data[0:100]
        print(len(data))
        embedding_list=self.vllm_embedding(data)
        similarity_scores=self.calculate_similarity_scores(embedding_list,self.sampling_num,self.repeat)
        self.score['Qwen3embd']=similarity_scores.numpy()
        print(self.score['Qwen3embd'])
        self.score_stat('Qwen3embd')
        self.save_score()

    def score_stat(self,metric,data=None):
        if self.guide_method=="CoTdecode":
            self.score[metric]=Haic_component.get_z_from_cummax(self.indices,self.score[metric])
        elif self.guide_method=="origin":
            self.score[metric]=Haic_component.score_at_k_vectorized(self.score[metric],self.sampling_num,self.repeat)
        else:
            assert self.repeat==1

        self.score[metric]=self.score[metric].mean(axis=0)

    def vllm_embedding(self,text_list):
        from openai import OpenAI
        from concurrent.futures import ThreadPoolExecutor
        # 客户端初始化不变
        client=OpenAI(base_url=self.url,api_key="none")
        model_name="Qwen3-Embedding-8B"  # 与启动命令中的--served-model-name一致

        def get_embedding(text):
            """调用服务获取单个文本的向量"""
            response=client.embeddings.create(  # 关键：使用embeddings接口
                model=model_name,
                input=text,  # input可以是字符串或字符串列表
                encoding_format="float"  # 指定返回格式，默认为float
            )
            return response.data[0].embedding

        # text_list = ["今天天气真好", "机器学习很有趣","今天天气真好", "机器学习很有趣","今天天气真好", "机器学习很有趣","今天天气真好", "机器学习很有趣"]
        # 使用线程池并行获取向量
        with ThreadPoolExecutor(max_workers=self.max_worker) as executor:
            embedding_list=list(executor.map(get_embedding,text_list))
        return embedding_list


    def calculate_similarity_scores(self,all_embeddings,b,k):
        import torch
        import torch.nn.functional as F

        # 将embedding列表转换为张量
        embeddings=torch.tensor(all_embeddings,dtype=torch.float32)  # 形状: [b*(k+1), m]
        # 将embedding分割为参考句子和备选句子
        # 重塑为(b, k+1, m)的形状
        embeddings_reshaped=embeddings.view(b,k+1,-1)
        # 提取参考句子和备选句子的embedding
        ref_embeddings=embeddings_reshaped[:,0,:]  # 形状: (b, m)
        cand_embeddings=embeddings_reshaped[:,1:,:]  # 形状: (b, k, m)
        # 使用torch的余弦相似度函数
        # 需要将ref_embeddings扩展维度以便广播计算
        ref_embeddings_expanded=ref_embeddings.unsqueeze(1)  # 形状: (b, 1, m)
        # 计算余弦相似度
        # dim=-1表示在最后一个维度（embedding维度）上计算相似度
        similarity_scores=F.cosine_similarity(cand_embeddings,ref_embeddings_expanded,dim=-1)  # 形状: (b, k)
        return similarity_scores