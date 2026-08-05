import os
import json
import re
import math
import numpy
from typing import Set

import torch
import torch.nn.functional as F

def wordchange(word1,word2,len1,len2):
    # 给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数。
    # 你可以对一个单词进行如下三种操作：插入一个字符、删除一个字符、替换一个字符。
    log=numpy.zeros((len1+1,len2+1),dtype=int)
    for i in range(len1+1):
        log[i,0]=i
    for j in range(len2+1):
        log[0,j]=j
    for i in range(1,len1+1):
        for j in range(1,len2+1):
            if word1[i-1]==word2[j-1]:
                log[i,j]=log[i-1][j-1]
            else:
                log[i,j]=min(log[i-1,j].item()+1,log[i,j-1].item()+1,log[i-1][j-1].item()+1)
    return log
# print(wordchange("张三","李三四",2,3))

# pip install pypinyin
from pypinyin import lazy_pinyin, Style
# def get_pinyin_length(char):
#     pinyin_list = lazy_pinyin(char, style=Style.NORMAL)
#     print(pinyin_list)
#     return len(pinyin_list[0])
# print(get_pinyin_length("张三32ars"))

def word_input_len_count(word,add_input):
    # 用于计算每个字符的修改次数
    word_input_len=[]
    if add_input=="auto":
        for item in word:
            item_len=lazy_pinyin(item, style=Style.NORMAL)[0]
            # print(item,item_len)
            if item_len==item:
                word_input_len.append(1)
            else:
                word_input_len.append(len(item_len))
    else:
        word_input_len=[add_input for _ in range(len(word))]
    return [0]+word_input_len

# print(word_input_len_count("李三四",1))
# print(word_input_len_count("李三四","auto"))

def wordinput(
    word1,word2,len1,len2,
    word1_input=1,
    word2_input="auto"
):
    log=numpy.zeros((len1+1,len2+1),dtype=int)
    word1_input_len=word_input_len_count(word1,word1_input)
    word2_input_len=word_input_len_count(word2,word2_input)

    for i in range(1,len1+1):
        log[i,0]=log[i-1,0]+word1_input_len[i]
    for j in range(1,len2+1):
        log[0,j]=log[0,j-1]+word2_input_len[j]
    for i in range(1,len1+1):
        for j in range(1,len2+1):
            if word1[i-1]==word2[j-1]:
                log[i,j]=log[i-1,j-1]
            else:
                log[i,j]=min(
                    log[i-1,j].item()+word1_input_len[i],
                    log[i,j-1].item()+word2_input_len[j],
                    log[i-1,j-1]+word1_input_len[i]+word2_input_len[j]
                )
    return log
# print(wordinput("张三","三四五",2,3))
# print(wordinput("abc c","abccc c",3,5))

# pip install jieba
import jieba
jieba.initialize() # 初始化jieba
def Jaccard_tokenize_text(text: str,language: str = 'zh') -> Set[str]:
    """
    根据语言对文本进行分词
    Args:
        text: 输入文本
        language: 语言类型，'zh'表示中文，'en'表示英文
    Returns:
        词语集合
    """
    # 移除多余空格和特殊字符
    text=text.strip()
    if language=='zh':
        # 中文分词 - 使用结巴分词
        # 如果需要更精确的分词，可以加载自定义词典
        # jieba.load_userdict("my_dict.txt")
        words=jieba.lcut(text)
        # 过滤空字符和标点符号
        words=[word for word in words if word.strip() and not re.match(r'^[^\w\u4e00-\u9fff]+$',word)]
    elif language=='en':
        # 英文分词 - 不转换为小写，按单词分割
        # 使用正则表达式分割，保留字母数字和连字符
        words=re.findall(r'\b[a-zA-Z0-9]+\b',text)
    # print(words)
    return words

def Jaccard_similarity(str_a,str_b,language):
    # language="zh"
    tokens_a=Jaccard_tokenize_text(str_a,language)
    tokens_b=Jaccard_tokenize_text(str_b,language)
    tokens_a=set(tokens_a[:-1])# 转换为集合
    tokens_b=set(tokens_b[:-1])
    # 计算Jaccard相似度
    if not tokens_a and not tokens_b:
        similarity=1.0  # 两个空集合视为完全相似
    else:
        intersection=tokens_a.intersection(tokens_b)
        union=tokens_a.union(tokens_b)
        similarity=len(intersection)/len(union)
    return similarity

# Jaccard_similarity("人工智能模","人工智障模型","zh")

def compute_cosine_similarity(X,Y):
    """
    计算余弦相似度矩阵
    参数:
        X: torch.Tensor, 形状 (n, d)
        Y: torch.Tensor, 形状 (m, d)
    返回:
        C: torch.Tensor, 形状 (n, m)
    """
    # 步骤1: 计算A和B
    n=X.size(0)
    m=Y.size(0)
    # 计算累积和
    A_cumsum=torch.cumsum(X,dim=0)  # 形状 (n, d)
    B_cumsum=torch.cumsum(Y,dim=0)  # 形状 (m, d)
    # 创建索引向量 [1, 2, ..., n] 和 [1, 2, ..., m]
    indices_A=torch.arange(1,n+1,dtype=torch.float32,device=X.device).view(-1,1)  # 形状 (n, 1)
    indices_B=torch.arange(1,m+1,dtype=torch.float32,device=Y.device).view(-1,1)  # 形状 (m, 1)
    # 计算平均值 A_i = (1/i) * sum_{k=1}^i X_k
    A=A_cumsum/indices_A  # 广播除法: (n, d) / (n, 1) → (n, d)
    B=B_cumsum/indices_B  # 广播除法: (m, d) / (m, 1) → (m, d)
    # 步骤2: 将A和B的每一行模长化为1（归一化）
    A_norm=F.normalize(A,p=2,dim=1)  # L2归一化，形状: (n, d)
    B_norm=F.normalize(B,p=2,dim=1)  # L2归一化，形状: (m, d)
    # 步骤3: 计算余弦相似度矩阵C 使用矩阵乘法计算所有向量对之间的点积
    C=torch.mm(A_norm,B_norm.T)  # 形状: (n, m)
    return C

# import torch
# from transformers import AutoModel
# # 1. 加载模型和分词器
# model_path = "/private/mwh/llm/Qwen2.5-0.5B/"  # 本地模型路径
# model = AutoModel.from_pretrained(model_path)
# # 2. 获取嵌入层权重
# # 方法A：直接访问（通用方法）
# input_embeddings = model.get_input_embeddings()
# embedding_weights = input_embeddings.weight.data  # [vocab_size, hidden_dim]
# # 方法B：如果模型是Qwen特定结构
# # if hasattr(model, 'embed_tokens'):
# #     embedding_weights = model.embed_tokens.weight.data
# # print(f"嵌入矩阵形状: {embedding_weights.shape}")
# torch.save(embedding_weights, "qwen_embeddings.pt")


def Qwen_embedding_score(str_a,str_b,tokenizer,embedding_weights):
    # print(tokenizer(str_a,return_tensors="pt")["input_ids"])
    # print(tokenizer(str_a,return_tensors="pt")["input_ids"].shape)
    input_ids_a=tokenizer(str_a,return_tensors="pt")["input_ids"][:,:-1]
    if input_ids_a.shape[1]==0:
        return 0.0
    # print(input_ids_a.shape)
    embedding_a=embedding_weights[input_ids_a].squeeze(0)
    # embedding_a=embedding_weights[input_ids_a]
    # print(embedding_weights[input_ids_a].shape)
    input_ids_b=tokenizer(str_b,return_tensors="pt")["input_ids"][:,:-1]
    if input_ids_b.shape[1]==0:
        return 0.0
    embedding_b=embedding_weights[input_ids_b].squeeze(0)
    # print(embedding_a.shape)
    # print(embedding_a.mean(dim=0,keepdim=True).shape)
    # print(embedding_a.mean(dim=0).shape)

    cos_sim=torch.cosine_similarity(embedding_a.mean(dim=0,keepdim=True),embedding_b.mean(dim=0,keepdim=True),dim=1)
    # print(cos_sim)
    # print(cos_sim.squeeze().item())
    # tmp=cos_sim.squeeze().item()
    # if math.isnan(tmp):
    #     print(tmp)
    #     print([str_a])
    #     print(tokenizer(str_a,return_tensors="pt")["input_ids"])
        # print(embedding_a.mean(dim=0,keepdim=True))
        # print(embedding_b.mean(dim=0,keepdim=True))
    return cos_sim.squeeze().item()





