import sys
import os


import time
import copy
import math
import json
import torch
import transformers

from source import (
    Environment,
    Load_data,
    Load_model,
    Create_string,
    Prompt,
    SimpleRAG,
    Nucleus3,
    LogitsProcessor
)


class Mainpart():
    def __init__(self,setting_combination):
        # self.setting_combination=setting_combination
        super().__init__()
        self.experiment_setting=setting_combination.experiment_setting
        self.generate_setting=setting_combination.generate_setting
        self.logits_setting=setting_combination.logits_setting
        self.adapter_setting=setting_combination.adapter_setting

        self.logf=setting_combination.logf
        self.start_time=setting_combination.start_time
        self.model=None
        self.tokenizer=None
        self.question=None
        self.answer=None
        self.few_shot_prompt=None

        if self.experiment_setting.guide_method[0:5]=="Cloud":
            self.device="cpu"
            # self.device="cuda"
        else:
            self.device="cuda"

        self.core_time=time.time()-time.time()
        self.token_count=[0,0]

        for setting in [self.experiment_setting,self.generate_setting,self.logits_setting,self.adapter_setting]:
            for key,value in setting.__dict__.items():
                while (len(key))<24:
                    key+=" "
                print(key,value)
            print("")

        print("transformers.generation.utils.__file__",transformers.generation.utils.__file__)

        original_stdout=sys.stdout
        with open(self.experiment_setting.output_path+"environment.txt","w") as f:
            sys.stdout=f
            Environment.get_edition_all()
        sys.stdout=original_stdout

    def load(self):
        (model,tokenizer,
         self.logits_setting.tokenizer_vocab_size,
         self.logits_setting.eot_token_id
         )=Load_model.load_model(
            self.experiment_setting.model_path,
            load_model=self.experiment_setting.guide_method[0:5]!='Cloud'
        )
        tokenizer.pad_token=tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
        self.logits_setting.tokenizer=tokenizer

        (question,
         answer,
         self.adapter_setting.state_vector_tensor,
         self.adapter_setting.token_id_tensor,
         self.adapter_setting.length_list,
         few_shot_prompt
         )=Load_data.prepare_data(
            model,
            tokenizer,
            self.experiment_setting
        )

        # 非数据集型数据加载
        # question=["大语言模型是一种新型人工智能方法，通过大规模神经网络学习人类自然语言文本中的隐式关系"]
        # question=["大语言模型是一种新型人工智能方法，通过大规模神经网络学习人类自然语言文本中的隐式关系。续写20个字就停止。"]

        self.model=model
        self.tokenizer=tokenizer
        self.question=question
        self.answer=answer
        self.few_shot_prompt=few_shot_prompt

    def flush(self):
        if self.experiment_setting.log_label:
            self.logf.flush()
            os.fsync(self.logf.fileno())

    def cap_init(self,current_length,idex=None):
        if self.experiment_setting.guide_label:
            custom_adapter_processor=LogitsProcessor.CustomLogitsProcessor(
                logits_setting=self.logits_setting,
                adapter_setting=self.adapter_setting,
                interactive_mode=self.experiment_setting.interactive_mode
            )
            if self.experiment_setting.mask_mode=="self":
                custom_adapter_processor.adapter_setting.mask=idex
                custom_adapter_processor.mask_tensor()
        else:
            if self.experiment_setting.guide_method=="CoTdecode":
                custom_adapter_processor=LogitsProcessor.CoTdecodeLogitsProcessor(
                    self.tokenizer,
                    self.logits_setting.tokenizer_vocab_size,
                    self.logits_setting.eot_token_id,
                    current_length,
                    self.logits_setting.paranum
                )
            else:
                # custom_adapter_processor=None
                custom_adapter_processor=LogitsProcessor.OriginLogitsProcessor(
                    self.tokenizer,
                    self.logits_setting.tokenizer_vocab_size,
                    self.logits_setting.eot_token_id,
                    current_length,
                    self.logits_setting.paranum
                )
        return custom_adapter_processor

    def ncg(self,inputs):
        time1=time.time()

        self.experiment_setting.actual_batch_size=inputs.input_ids.shape[0]
        # current_text=[self.few_shot_prompt+c_i for c_i in current_text]
        # inputs=self.tokenizer(current_text,return_tensors="pt",padding=True,padding_side='left').to(self.model.device)

        # print(inputs)

        current_length=inputs.input_ids.shape[1]

        self.generate_setting.remaining_length=self.model.config.max_position_embeddings-current_length
        self.logits_setting.paranum=self.experiment_setting.actual_batch_size*self.generate_setting.num_beams
        self.logits_setting.current_length=current_length

        custom_adapter_processor=self.cap_init(current_length)
        outputs=Nucleus3.nucleus_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            inputs=inputs,
            custom_adapter_processor=custom_adapter_processor,
            generate_setting=self.generate_setting,
        )
        output_text=self.tokenizer.batch_decode(
            [seq[current_length:] for seq in outputs],
            skip_special_tokens=True
        )

        if self.experiment_setting.guide_method=="CoTdecode":
            self.experiment_setting.CoTdecodelog.append(custom_adapter_processor.log)

        if self.logits_setting.plot in ["guide","origin"]:
            self.experiment_setting.plot_hs_log.append(torch.stack(custom_adapter_processor.plot_hs, dim=0))
            self.experiment_setting.plot_df_log.append(torch.stack(custom_adapter_processor.plot_df, dim=0))

        time2=time.time()
        self.core_time=self.core_time+time2-time1

        return output_text

    def worker(self):

        idex=0 #第几个实验
        # question_idex=[] #数据集中的第几个问题
        # repeat_idex=0 #实验的第几次重复
        # question_num=0 #测试问题总数
        results_log=[]

        question_num=min(len(self.question),self.experiment_setting.experiment_num)
        question_idex=[i for i in range(question_num)]
        if self.experiment_setting.experiment_idex is not None:
            question_num=len(self.experiment_setting.experiment_idex)
            question_idex=self.experiment_setting.experiment_idex

        print("question_num",question_num)

        print_idex=0
        if self.experiment_setting.prompt_mode in ["RAG"]:
            language=Prompt.language_log(self.experiment_setting.dataset_name)
            rag=SimpleRAG.Pipeline(self.few_shot_prompt[2:],language)

        while(idex<question_num):

            if print_idex>=self.experiment_setting.experiment_num:
                break

            # print(idex)
            output_text=[]
            haic_node_log=[] #伴写模式的测试点
            question_input=self.tokenizer(self.question[question_idex[idex]],return_tensors="pt",padding=True,padding_side='left').to(self.device)
            total_legth=question_input.input_ids.shape[1]

            if self.experiment_setting.dataset_name in self.experiment_setting.haic_dataset:
                for haic_node_i in range(self.experiment_setting.initial_length,total_legth,self.experiment_setting.step_length):
                    if self.experiment_setting.ensure_word_complete:
                        checked_haic_node_i=Create_string.create_safe_string(self.question[question_idex[idex]],haic_node_i,self.tokenizer)
                    else:
                        checked_haic_node_i=haic_node_i
                    haic_node_log.append(checked_haic_node_i)
            else:
                haic_node_log=[total_legth]

            if self.experiment_setting.prompt_mode!="RAG":
                prompt_input=self.tokenizer(self.few_shot_prompt,return_tensors="pt",padding=True,padding_side='left').to(self.device)

            for haic_node_i in haic_node_log:
                if print_idex>=self.experiment_setting.experiment_num:
                    break

                print_idex+=1
                if print_idex%50==0:
                    print("idex",idex,"time",time.time()-self.start_time,"\t\tcore time",self.core_time)
                    self.flush()

                if self.experiment_setting.prompt_mode=="RAG":
                    result=rag.query(self.tokenizer.decode(question_input["input_ids"][0,0:haic_node_i],skip_special_tokens=True),top_k=5)
                    tmp_prompt=[result_i.strip('\n') for result_i in result]
                    prompt=self.few_shot_prompt[0]+"\n\n".join(tmp_prompt)+self.few_shot_prompt[1]
                    prompt_input=self.tokenizer(prompt,return_tensors="pt",padding=True,padding_side='left').to(self.device)


                # current_text=[]
                batch_idex=0
                repeat_idex=0
                while_i=0
                try:
                    while(1):
                        # current_text.append(self.question[question_idex[idex]][0:haic_node_i])
                        batch_idex+=1
                        repeat_idex+=1
                        while_i+=1
                        if batch_idex>=self.experiment_setting.batch_size or repeat_idex>=self.generate_setting.repeat_num:
                            tmp_inputs=copy.deepcopy(question_input)
                            tmp_inputs['input_ids']=torch.cat([prompt_input.input_ids,tmp_inputs.input_ids[:,0:haic_node_i]],dim=-1).repeat(while_i,1)
                            tmp_inputs['attention_mask'] =torch.cat([prompt_input.attention_mask,tmp_inputs.attention_mask[:,0:haic_node_i]],dim=-1).repeat(while_i,1)

                            if self.experiment_setting.guide_method[0:5]=="Cloud":
                                # print(tmp_inputs['input_ids'].shape)
                                token_ids=tmp_inputs['input_ids'][0,:]  # [0]表示取第一个序列
                                self.token_count[0]+=tmp_inputs['input_ids'][0,:].shape[0]
                                self.token_count[1]+=50
                                output_text.append(self.tokenizer.decode(token_ids,skip_special_tokens=True))
                            else:
                                output_text+=self.ncg(tmp_inputs)
                            batch_idex=0
                            while_i=0
                            if repeat_idex==self.generate_setting.repeat_num:
                                break

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) and self.experiment_setting.batch_size>1:
                        self.experiment_setting.batch_size=min(
                            self.experiment_setting.batch_size-1,
                            math.ceil(self.generate_setting.repeat_num/(math.ceil(self.generate_setting.repeat_num/self.experiment_setting.batch_size)+1))
                        )
                        print("batch_size   ",self.experiment_setting.batch_size)
                    else:
                        assert 1==0,e

            if self.experiment_setting.dataset_name[0:5]=="GSM8k":
                results_log.append({
                    "idex":idex,
                    "answer":self.answer[idex],
                    "text":output_text
                })
            elif self.experiment_setting.dataset_name=="MATH500":
                results_log.append({
                    "idex":idex,
                    "prompt":self.question[idex],
                    "answer":self.answer[idex],
                    "text":output_text
                })
            elif self.experiment_setting.dataset_name in self.experiment_setting.haic_dataset:
                results_log.append({
                    "idex":question_idex[idex],
                    # "haic_node":(self.experiment_setting.initial_length,total_legth,self.experiment_setting.step_length),
                    "haic_node":haic_node_log if self.experiment_setting.ensure_word_complete else (self.experiment_setting.initial_length,total_legth,self.experiment_setting.step_length),
                    "text":output_text
                })
            else:
                results_log.append({
                    "prompt":self.question[idex],
                    "response":output_text
                })

            idex+=1


        if self.experiment_setting.dataset_name[0:5]=="GSM8k" or self.experiment_setting.dataset_name in self.experiment_setting.haic_dataset:
            with open(self.experiment_setting.output_path+"results.json","a",encoding="utf-8") as file:
                json.dump(results_log,file,indent=4,ensure_ascii=False)
        else:
            with open(self.experiment_setting.output_path+"results.jsonl",'a',encoding='utf-8') as file:
                for data in results_log:
                    json_line=json.dumps(data,ensure_ascii=False)
                    file.write(json_line+'\n')
        if self.experiment_setting.guide_method=="CoTdecode":
            big_tensor=torch.cat(self.experiment_setting.CoTdecodelog,dim=0)
            torch.save(big_tensor,self.experiment_setting.output_path+'CoTdecodelog.pt')
        if self.logits_setting.plot in ["guide","origin"]:
            plot_hs=torch.stack(self.experiment_setting.plot_hs_log,dim=0)
            plot_df=torch.stack(self.experiment_setting.plot_df_log,dim=0)
            torch.save(plot_hs,self.experiment_setting.output_path+'plot_hs.pt')
            torch.save(plot_df,self.experiment_setting.output_path+'plot_df.pt')
            print(plot_hs.shape)
            print(plot_df.shape)

        print("---experiment summary---")
        print("question_num ",question_num)
        print("sampling_num ",print_idex)
        print("batch_size   ",self.experiment_setting.batch_size)
        print("tokne count  ",self.token_count)
        print("------------------------")

        return

def mainline(
    setting_combination
):
    mainpart=Mainpart(setting_combination)
    mainpart.load()
    mainpart.worker()


