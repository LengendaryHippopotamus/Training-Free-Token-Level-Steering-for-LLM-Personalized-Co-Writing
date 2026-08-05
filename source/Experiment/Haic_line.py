import os
import json
import numpy
import torch
from source.Experiment import Haic_metric,Haic_component

class haic_eval():
    def __init__(
        self,
        file_path,
        dataset_name,
        sampling_num,
        guide_method,
        generate_string_length=40,
        reference_length=30,
        metirc_log=[],
        repeat=1,
        reflash=False,
        mode="abstract",
        length_label=False,
        haic_node_simplify=True,
        token_level=True,
        original_prompt_label=False,
        tokenizer1=None,
        tokenizer2=None,
        data_path="",
    ):
        self.file_path=file_path
        self.dataset_name=dataset_name
        self.sampling_num=sampling_num
        self.guide_method=guide_method
        self.generate_string_length=generate_string_length
        self.reference_length=reference_length
        self.metirc_log=metirc_log
        self.repeat=repeat

        self.reflash=reflash
        self.mode=mode
        self.length_label=length_label
        self.haic_node_simplify=haic_node_simplify
        self.token_level=token_level
        self.original_prompt_label=original_prompt_label

        self.tokenizer1=tokenizer1
        self.tokenizer2=tokenizer2
        self.data_path=data_path

        self.Qwen3embd_log=[]
        # if self.repeat>1:
        self.tmp_Levenshtein=numpy.zeros((
                self.repeat,
                self.generate_string_length+1,
                self.reference_length+1
            ),dtype=float)
        self.tmp_pinyin=numpy.zeros((
                self.repeat,
                self.generate_string_length+1,
                self.reference_length+1
            ),dtype=float)

        if self.guide_method=="CoTdecode":
            pdiff=torch.load(file_path+'CoTdecodelog.pt',map_location=torch.device('cpu')).reshape(self.sampling_num,self.repeat)
            _,self.indices=torch.cummax(pdiff,dim=1)

        # print(self.guide_method)

        self.idex=0

    def save_score(self):
        assert self.idex==self.sampling_num
        if self.mode=="abstract":
            numpy.savez(self.file_path+'score.npz',**self.score)
        elif self.mode=="detail":
            numpy.savez(self.file_path+'score_detail.npz',**self.score)
        if "Qwen3embd_1" in self.metirc_log:
            with open(self.file_path+'Qwen3embd_log.json','w',encoding='utf-8') as f:
                json.dump(self.Qwen3embd_log,f,ensure_ascii=False,indent=4)

    def load_socre(self):
        if self.mode=="abstract":
            if not os.path.exists(self.file_path+"score.npz") or self.reflash:
                self.score=dict(
                    Levenshtein=numpy.zeros((self.repeat,self.generate_string_length+1,self.reference_length+1),dtype=float),
                    pinyin=numpy.zeros((self.repeat,self.generate_string_length+1,self.reference_length+1),dtype=float),
                    Jaccard=numpy.zeros((self.sampling_num,self.repeat),dtype=float),
                    AvgWordVec=numpy.zeros((self.sampling_num,self.repeat),dtype=float),
                    Qwen3embd=None
                )
            else:
                data=numpy.load(self.file_path+'score.npz',allow_pickle=True)
                self.score=dict(data)
                data.close()
        elif self.mode=="detail":
            if not os.path.exists(self.file_path+"score_detail.npz") or self.reflash:
                self.score=dict(
                    Levenshtein=numpy.zeros((self.sampling_num,self.repeat),dtype=float),
                    pinyin=numpy.zeros((self.sampling_num,self.repeat),dtype=float),
                    Jaccard=numpy.zeros((self.sampling_num,self.repeat),dtype=float),
                    AvgWordVec=numpy.zeros((self.sampling_num,self.repeat),dtype=float),
                    Qwen3embd=None
                )
            else:
                data=numpy.load(self.file_path+'score_detail.npz',allow_pickle=True)
                self.score=dict(data)
                data.close()

        if self.metirc_log=="auto":
            self.metirc_log=[]
            if self.score["Levenshtein"].sum()==0:
                self.metirc_log.append("Levenshtein")
            if self.score["pinyin"].sum()==0:
                self.metirc_log.append("pinyin")
            if self.score["Jaccard"].sum()<0.01:
                self.metirc_log.append("Jaccard")
            if self.score["AvgWordVec"].sum()<0.01:
                self.metirc_log.append("AvgWordVec")
        else:
            for item in self.metirc_log:
                if item in ["Levenshtein","pinyin"]:
                    self.score[item]=numpy.zeros((self.repeat,self.generate_string_length+1,self.reference_length+1),dtype=float)
                elif item in ["Jaccard","AvgWordVec"]:
                    self.score[item]=numpy.zeros((self.sampling_num,self.repeat),dtype=float)

    def haic_result_line(self):
        self.load_socre()
        if self.metirc_log!=[]:
            self.haic_eval_trubo()
        if self.score["Jaccard"].shape[0]==self.sampling_num:
            self.score_stat("Jaccard")
        if self.score["AvgWordVec"].shape[0]==self.sampling_num:
            self.score_stat("AvgWordVec")
        if self.metirc_log!=[]:
            self.save_score()

    def haic_eval_trubo(self):
        length_log=[]
        # if (not os.path.exists(self.file_path+"score1.npy")) or self.reflash:
        with open(self.file_path+"results.json",'r',encoding='utf-8') as f:
            data=json.load(f)  # 读取 JSON 数据，假设是一个 list
        # num=len(data)
        if self.original_prompt_label==False:
            # with open(r"F:\code\healthcare nucleus generation\data\\"+self.dataset_name+"\question.json","r",encoding="utf-8") as file:
            with open(self.data_path+self.dataset_name+"/question.json","r",encoding="utf-8") as file:
                test_question=json.load(file)
        # data=data[0:2]
        for item in data:
            if self.original_prompt_label==True:
                prompt=item["prompt"]
            else:
                prompt=test_question[item["idex"]]
            text=item["text"]
            haic_node=item["haic_node"]
            if self.haic_node_simplify:
                haic_node=[x for x in range(haic_node[0],haic_node[1],haic_node[2])]
            assert len(text)==self.repeat*len(haic_node)
            if self.token_level==True:
                haic_node=Haic_component.batch_truncate_and_decode(
                    text=prompt,
                    truncation_lengths=haic_node,
                    tokenizer=self.tokenizer1
                )
            for i,haic_node_i in enumerate(haic_node):

                if "Qwen3embd_1" in self.metirc_log:
                    self.Qwen3embd_log.append(prompt[haic_node_i:haic_node_i+self.reference_length])
                for repeat_i in range(self.repeat):
                    if "Qwen3embd_1" in self.metirc_log:
                        self.Qwen3embd_log.append(text[i*self.repeat+repeat_i][0:self.generate_string_length])

                    # print(haic_node_i)
                    # print([prompt[0:haic_node_i]])
                    # print([text[i*self.repeat+repeat_i][0:self.generate_string_length]])
                    # print([prompt[haic_node_i:haic_node_i+self.reference_length]])

                    self.evaluate(
                        repeat_i,
                        text[i*self.repeat+repeat_i][0:self.generate_string_length],
                        prompt[haic_node_i:haic_node_i+self.reference_length],
                        self.generate_string_length,
                        self.reference_length
                    )
                    if self.length_label:
                        length_log.append(len(text[i*self.repeat+repeat_i]))
                if "Levenshtein" in self.metirc_log or "pinyin" in self.metirc_log:
                    self.batch_evaluate()
                self.idex+=1


        if self.length_label:
            self.score.length_log=numpy.array(length_log)

    def detail_eval_trubo(self):
        self.mode="detail"
        self.load_socre()
        if self.metirc_log!=[]:
            self.haic_eval_trubo()
        if self.score["Jaccard"].shape[0]==self.sampling_num:
            self.score_stat("Jaccard")
        if self.score["AvgWordVec"].shape[0]==self.sampling_num:
            self.score_stat("AvgWordVec")
        if self.metirc_log!=[]:
            self.save_score()

        score=numpy.column_stack((
            self.score["Levenshtein"],
            self.score["pinyin"],
            self.score["Jaccard"],
            self.score["AvgWordVec"]
        ))
        with open(self.file_path+"results.json",'r',encoding='utf-8') as f:
            data=json.load(f)  # 读取 JSON 数据，假设是一个 list
        tmp_len=0
        ret=[]
        for item in data:
            haic_node=item["haic_node"]
            if self.haic_node_simplify:
                t_len=len([x for x in range(haic_node[0],haic_node[1],haic_node[2])])
                ret.append(score[tmp_len:tmp_len+t_len,:])
                tmp_len+=t_len
        return ret


    def evaluate(
        self,
        repeat_i,
        generate_text,
        standard_text,
        generate_len,
        standard_len
    ):
        if standard_len!=len(standard_text):
            print("standard text length error")
            if standard_len<len(standard_text):
                standard_text=standard_text[0:standard_len]
            else:
                while(standard_len>len(standard_text)):
                    standard_text+=" "
        if generate_len<len(generate_text):
            generate_text=generate_text[0:generate_len]
        elif generate_len>len(generate_text):
            print("generate text length error",len(generate_text))
            while(generate_len>len(generate_text)):
                generate_text+=" "
        if "Levenshtein" in self.metirc_log:
            score1=Haic_metric.wordchange(generate_text,standard_text,generate_len,standard_len)
            # self.score["Levenshtein"][repeat_i]=score1
            self.tmp_Levenshtein[repeat_i]=score1
        if "pinyin" in self.metirc_log:
            score2=Haic_metric.wordinput(generate_text,standard_text,generate_len,standard_len,1,"auto")
            # self.score["pinyin"][repeat_i]=score2
            self.tmp_pinyin[repeat_i]=score2
        if "Jaccard" in self.metirc_log:
            if self.dataset_name in ["CDN"]:
                language="en"
            else:
                language="zh"
            # print(Haic_metric.Jaccard_similarity(generate_text,standard_text,language))
            self.score["Jaccard"][self.idex,repeat_i]=(Haic_metric.Jaccard_similarity(generate_text,standard_text,language))
        if "AvgWordVec" in self.metirc_log:
            self.score["AvgWordVec"][self.idex,repeat_i]=(Haic_metric.Qwen_embedding_score(generate_text,standard_text,self.tokenizer2,self.embedding_weights))

    def batch_evaluate(self):
        tmp_L=0
        tmp_p=0
        if self.guide_method=="CoTdecode":
            tmp_indices=self.indices[self.idex]
            # 获取累积最大值索引
            _,indices=torch.cummax(tmp_indices,dim=0)
            indices_list=indices.cpu().numpy()
            tmp_L=self.tmp_Levenshtein[indices_list]
            tmp_p=self.tmp_pinyin[indices_list]

        elif self.guide_method=="origin" and self.repeat>1:
            # 使用向量化版本
            Levenshtein_areas=Haic_component.A_calc(self.tmp_Levenshtein)
            pinyin_areas=Haic_component.A_calc(self.tmp_pinyin)
            for i in range(self.repeat):
                tmp_L=Haic_component.score_at_k_table(Levenshtein_areas,self.tmp_Levenshtein,self.repeat,i+1)
                tmp_p=Haic_component.score_at_k_table(pinyin_areas,self.tmp_pinyin,self.repeat,i+1)
        else:
            assert self.repeat==1
            tmp_L=self.tmp_Levenshtein
            tmp_p=self.tmp_pinyin

        if self.mode=="abstract":
            self.score["Levenshtein"]+=tmp_L.squeeze()
            self.score["pinyin"]+=tmp_p.squeeze()

        elif self.mode=="detail":
            self.score["Levenshtein"][self.idex]=Haic_component.A_calc(tmp_L)
            self.score["pinyin"][self.idex]=Haic_component.A_calc(tmp_p)

    def score_stat(self,metric,data=None):
        # 由于历史原因，Levenshtein和pinyin两个指标在循环过程中就已经进行整理(batch_evaluate函数)
        # 而其他指标开始时未整理，因此编写stat函数对其进行整理。
        # 进而之后的实验也使用这个方法进行整理
        # 同样类似batch_evaluate分3部分
        assert metric in ["Jaccard","AvgWordVec",'Qwen3embd']
        if self.guide_method=="CoTdecode":
            self.score[metric]=Haic_component.get_z_from_cummax(self.indices,self.score[metric])
        elif self.guide_method=="origin" and self.repeat>1:
            self.score[metric]=Haic_component.score_at_k_vectorized(self.score[metric],self.sampling_num,self.repeat)
        else:
            assert self.repeat==1
            pass
        if self.mode=="abstract":
            self.score[metric]=self.score[metric].mean(axis=0)
        elif self.mode=="detail":
            pass


