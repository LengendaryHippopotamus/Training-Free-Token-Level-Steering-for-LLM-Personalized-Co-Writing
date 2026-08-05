import torch
from transformers import LogitsProcessor

class CADLogitsProcessor(LogitsProcessor):
    def __init__(self,model,uncond_input_ids,alpha=0.5):
        """
        初始化 Context-Aware Decoding 处理器
        :param model: 你的 Qwen/Llama 模型对象 (同一个即可，不需要双份显存)
        :param uncond_input_ids: 只有用户当前输入，不带任何增广/RAG prompt的序列 (Shape: [1, seq_len])
        :param alpha: CAD 罚项系数，默认为 0.5。越大，越偏向于 RAG context。
        """
        self.model=model
        self.uncond_input_ids=uncond_input_ids
        self.alpha=alpha

        # 用于保存"无RAG序列"的 KV-Cache
        self.past_key_values=None
        self.is_first_step=True

    def __call__(self,input_ids: torch.LongTensor,scores: torch.FloatTensor) -> torch.Tensor:
        """
        :param input_ids: 主循环(带RAG序列)当前的所有 token
        :param scores: 主循环生成的最新 logits 分布，相当于 Logits_{with_context}
        """
        # 注意：这里我们使用 torch.no_grad() 是因为 Inference 阶段
        with torch.no_grad():
            if self.is_first_step:
                # 第一步：对"无 RAG 序列"进行全量 Prefill (预填充阶段)
                outputs=self.model(
                    input_ids=self.uncond_input_ids,
                    use_cache=True
                )
                uncond_logits=outputs.logits[:,-1,:]  # 获取最后一个 token 的 logits
                self.past_key_values=outputs.past_key_values
                self.is_first_step=False
            else:
                # 第二步及之后：强制同步（这是最巧妙的一步！）
                # 此时主循环已经在上一步根据 CAD logits 挑选出了一个 token 并且 append 到了 input_ids
                # 我们抽取出这个"已经被选定的上一个 token"，强行喂给"无 RAG 序列"进行 Decode
                last_chosen_token=input_ids[:,-1:]

                outputs=self.model(
                    input_ids=last_chosen_token,
                    past_key_values=self.past_key_values,  # 传入上一步的 cache
                    use_cache=True
                )
                uncond_logits=outputs.logits[:,-1,:]
                self.past_key_values=outputs.past_key_values

        # === 核心 CAD 公式计算 ===
        # Logits_CAD = (1 + α) * Logits_cond - α * Logits_uncond

        scores-=torch.logsumexp(scores,dim=1,keepdim=True)
        uncond_logits-=torch.logsumexp(uncond_logits,dim=1,keepdim=True)

        cad_scores=(1+self.alpha)*scores-self.alpha*uncond_logits

        return cad_scores



# # 1. 准备大 Context (包含 RAG/补充数据集检索结果 + User Input)
# # 形如：[RAG Context] + "I am writing a piece of..."
# inputs = tokenizer(prompt_with_rag, return_tensors="pt").to(model.device)
#
# # 2. 准备小 Context (原汁原味的 User Input)
# # 形如："I am writing a piece of..."
# uncond_inputs = tokenizer(user_input_only, return_tensors="pt").to(model.device)
#
# # 3. 初始化你的 CAD Processor
# cad_processor = CADLogitsProcessor(
#     model=model,
#     uncond_input_ids=uncond_inputs.input_ids,
#     alpha=0.5 # CAD 的超参，通常设为 0.5 能出不错的效果
# )
#
# # 4. 组装并生成
# generate_kwargs = {
#     **inputs, # 注意：外层 generate 喂的是包含 RAG context 的长序列
#     'max_new_tokens': min(generate_setting.generate_token_length, generate_setting.remaining_length),
#     'num_return_sequences': 1,
#     'do_sample': False, # 根据你的正文 "uniformly applied greedy decoding"，这里建议用 False 保障公平基线
#     'pad_token_id': tokenizer.eos_token_id,
#     'logits_processor': [cad_processor] # 你的 CAD processor 挂载在这里
# }
#
# outputs = model.generate(**generate_kwargs)
#
# # outputs 即为你 CAD 修正后生成的最终序列