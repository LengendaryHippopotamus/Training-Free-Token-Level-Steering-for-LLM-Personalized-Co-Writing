import json
import jieba
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Pipeline:
    def __init__(self,documents,language):
        # 初始化分词器
        if language=="python":
            self.text_splitter=RecursiveCharacterTextSplitter.from_language(
                    language=Language.PYTHON,
                    chunk_size=200,  # 目标chunk大小
                    chunk_overlap=0  # 重叠部分
                )
        elif language=="en":
            self.text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=300,  # 目标chunk大小
                chunk_overlap=50,  # 重叠部分
                length_function=len,  # 长度函数
                separators=["\n\n","\n",".",";",","," ",""]  # 分割符优先级
            )
        elif language=="zh":
            self.text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=300,  # 目标chunk大小
                chunk_overlap=50,  # 重叠部分
                length_function=len,  # 长度函数
                separators=["\n\n","\n","。","；","，"," ",""]  # 分割符优先级
            )
        else:
            print("Language Error")
        self.chunks=[]
        for doc in documents:
            chunks=self.text_splitter.split_text(doc)  #切片
            self.chunks+=chunks

        self.chunks=set(self.chunks)
        self.chunks=list(self.chunks)

        # 初始化TF-IDF向量化检索库
        if language in ["en","python"]:
            self.vectorizer=TfidfVectorizer(
                max_features=5000,  # 限制特征数
                stop_words=None,  # 中文可自定义停用词
                token_pattern=r'(?u)\b\w+\b'
            )
        elif language=="zh":
            self.vectorizer=TfidfVectorizer(
                max_features=5000,
                stop_words=None,
                tokenizer=self.jieba_tokenizer,  # 使用jieba分词器
                token_pattern=None  # 禁用默认的token_pattern
            )
        else:
            print("Language Error")
        self.tfidf_matrix=self.vectorizer.fit_transform(self.chunks)  # 训练TF-IDF

    def jieba_tokenizer(self,text):
        """使用jieba进行中文分词，并过滤停用词"""
        # 使用jieba进行分词
        words=jieba.cut(text)
        # 过滤停用词和非中文字符
        filtered_words=[]
        for word in words:
            word=word.strip()
            if (len(word)>0 and  # 非空
                    # word not in self.stopwords and  # 不在停用词表中
                    not word.isspace()):  # 不是空白字符
                filtered_words.append(word)
        return filtered_words

    def query(self,question: str,top_k: int = 3):
        query_vec=self.vectorizer.transform([question])
        similarities=cosine_similarity(query_vec,self.tfidf_matrix).flatten()
        top_indices=similarities.argsort()[-top_k:][::-1]
        return [self.chunks[idx] for idx in top_indices]

if __name__=="__main__":
    # with open(r"F:\code\healthcare nucleus generation\data\CodeNet\reference.json","r",encoding="utf-8") as file:
    #     documents=json.load(file)
    # rag = Pipeline(documents,"python")
    # result=rag.query("returns i such that all(val<x for val in a[lo:i]) and all(val>-=x for val in a[i:hi]).\n\n# from collections import deque", top_k=3)

    with open(r"reference.json","r",encoding="utf-8") as file:
        documents=json.load(file)
    rag=Pipeline(documents,"en")
    result=rag.query(
        "把保障和改善民生紧紧抓在手上，切实托住这个底，使民生改善和经济发展有效对接、良性循环、相得益彰\n要更加突出就业优先导向，确保重点群体就业稳定，加力推动就业形势持续好转，以更加充分、更高质量就业带动劳动者增收，让人民群众享有实实在在的获得感",
        top_k=5)

    for item in result:
        print([item])
