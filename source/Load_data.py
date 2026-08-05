import os
import json
import torch
import pyarrow.parquet

from source import CausalLM,Prompt

def prepare_data(
    model,
    tokenizer,
    experiment_setting
):
    reference_texts=None
    test_question=None
    test_answer=None
    few_shot_prompt=""
    state_vector_tensor=None
    token_id_tensor=None
    length_list=[]

    if experiment_setting.interactive_mode!="interactive":#非交互式，读取测试数据集
        if experiment_setting.dataset_name in experiment_setting.haic_dataset:
            with open(experiment_setting.dataset_path+"question.json","r",encoding="utf-8") as file:
                test_question=json.load(file)

    # if experiment_setting.guide_label:
    #     if experiment_setting.few_shot_num>0 or not os.path.isfile(experiment_setting.state_vector_path+"state_vector_tensor.pth"):
    if experiment_setting.few_shot_num>0 or (experiment_setting.guide_label and not os.path.isfile(experiment_setting.state_vector_path+"state_vector_tensor.pth")) or experiment_setting.prompt_mode in ["RAG","RAGS"]:
        question_length=None

        if experiment_setting.reference_dataset_name in experiment_setting.haic_dataset:
            with open(experiment_setting.dataset_path+"reference.json","r",encoding="utf-8") as file:
                reference_texts=json.load(file)

        # cross_validation_num=3
        # cross_validation_idex=int(experiment_setting.dataset_name[-1])
        cross_validation_num=experiment_setting.cross_validation_num
        if cross_validation_num>1:
            cross_validation_idex=experiment_setting.cross_validation_idex
            data_length=len(test_question)

            cut_idex=[0]
            for i in range(cross_validation_num-1):
                cut_idex.append((data_length//cross_validation_num)*(i+1))
            cut_idex.append(data_length)

            # test_question=test_question[cut_idex[cross_validation_idex-1]:cut_idex[cross_validation_idex]]
            reference_texts=reference_texts[0:cut_idex[cross_validation_idex-1]]+reference_texts[cut_idex[cross_validation_idex]:data_length]
            question_length=question_length[0:cut_idex[cross_validation_idex-1]]+question_length[cut_idex[cross_validation_idex]:data_length]

        if experiment_setting.guide_label and (not os.path.isfile(experiment_setting.state_vector_path+"state_vector_tensor.pth") or os.path.isfile(experiment_setting.state_vector_path+"length_list.json")):
            CausalLM.state_vector(reference_texts,model,tokenizer,experiment_setting.state_vector_path,question_length=question_length)

        if experiment_setting.few_shot_num>0:
            for i in range(experiment_setting.few_shot_num):
                few_shot_prompt+=reference_texts[i]
                few_shot_prompt+="\n\n"

    print(experiment_setting.state_vector_path+"state_vector_tensor.pth")
    if experiment_setting.guide_label:
        state_vector_tensor=torch.load(experiment_setting.state_vector_path+"state_vector_tensor.pth",map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),weights_only=True)
        token_id_tensor=torch.load(experiment_setting.state_vector_path+"token_id_tensor.pth",map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),weights_only=True)
        if experiment_setting.reference_num:
            state_vector_tensor=state_vector_tensor[0:experiment_setting.reference_num]
            token_id_tensor=token_id_tensor[0:experiment_setting.reference_num]
        with open(experiment_setting.state_vector_path+"length_list.json",'r',encoding='utf-8') as f:
            length_list=json.load(f)
        print(state_vector_tensor.shape)


    if experiment_setting.prompt_mode in ["origin","RAG","chat"]:
        few_shot_prompt=Prompt.prompt_log(experiment_setting.prompt_mode,experiment_setting.dataset_name)

    if experiment_setting.prompt_mode in ["RAG"]:
        few_shot_prompt=few_shot_prompt+reference_texts
    return test_question,test_answer,state_vector_tensor,token_id_tensor,length_list,few_shot_prompt

