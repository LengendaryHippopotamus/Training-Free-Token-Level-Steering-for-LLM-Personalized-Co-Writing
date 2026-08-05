import os
import json
import torch


def state_vector(text_list,model,tokenizer,save_path,question_length=None):
    print("Causal LM state vector")
    state_vector_tensor=0
    token_id_tensor=0
    length_list=[0]
    device=model.device
    for idex,text in enumerate(text_list):
        if idex<5 or (idex%10==0 and idex<100) or idex%500==0:
            print(idex)
        inputs=tokenizer(text,return_tensors="pt").to(device)
        with torch.no_grad():
            outputs=model(**inputs,output_hidden_states=True)
        last_hidden_layer=outputs.hidden_states[-1]
        lm_head_inputs=last_hidden_layer[0]

        normalized_lm=lm_head_inputs
        next_token_id=torch.cat((inputs["input_ids"][0][1:],torch.tensor([151643]).long().to(device)))

        if question_length is not None:
            normalized_lm=normalized_lm[question_length[idex]:]
            next_token_id=next_token_id[question_length[idex]:]
        if idex==0:
            state_vector_tensor=normalized_lm
            token_id_tensor=next_token_id
            length_list.append(next_token_id.shape[0])
        else:
            state_vector_tensor=torch.cat((state_vector_tensor, normalized_lm), dim=0)
            token_id_tensor=torch.cat((token_id_tensor, next_token_id), dim=0)
            length_list.append(next_token_id.shape[0]+length_list[-1])

    os.makedirs(save_path, exist_ok=True)
    torch.save(state_vector_tensor,save_path+"state_vector_tensor.pth")
    torch.save(token_id_tensor,save_path+"token_id_tensor.pth")

    with open(save_path+"length_list.json", 'w', encoding='utf-8') as f:
        json.dump(length_list, f, indent=4)

    print("state vector num",token_id_tensor.shape[0])
    return

if __name__=="__main__":
    from transformers import AutoTokenizer,AutoModelForCausalLM
    # 加载模型和分词器
    model_name="Qwen2.5-0.5B"
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    # model=AutoModelForCausalLM.from_pretrained(model_name,
    #                                            device_map="auto",
    #                                            torch_dtype="auto")

    # 示例文本
    # text="人工智能是一门研究如何使计算机模拟人类智能行为和思维过程的科学与技术。"
    text="人工智能是新兴学科"

    # 文本处理
    # inputs=tokenizer(text,return_tensors="pt").to(model.device)
    inputs=tokenizer(text,return_tensors="pt")
    # print(inputs)
    print(inputs["input_ids"][0])
    print(type(inputs["input_ids"][0]))

