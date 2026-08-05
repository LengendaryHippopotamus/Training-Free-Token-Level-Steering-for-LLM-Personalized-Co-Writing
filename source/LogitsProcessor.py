import math
import numpy
import torch
import transformers
from sympy.assumptions.predicates.order import NonNegativePredicate
from transformers import LogitsProcessor

from source import Adapter

class OriginLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        tokenizer_vocab_size,
        eot_token_id,
        current_length,
        paranum
    ):
        super().__init__()
        self.tokenizer=tokenizer
        self.tokenizer_vocab_size=tokenizer_vocab_size
        self.current_length=current_length
        self.paranum=paranum
        # 预创建end_tensor模板
        self.end_tensor_template=torch.full((tokenizer_vocab_size,),-float('inf'),device='cuda')
        self.end_tensor_template[eot_token_id]=0

    def __call__(self,input_ids: torch.LongTensor,modified_scores: torch.Tensor) -> torch.Tensor:
        # texts=self.tokenizer.batch_decode(input_ids,skip_special_tokens=False)
        # print(texts)
        return modified_scores


    def get_top_tokens(self,E,k,mode="print"):
        # 获取top-k概率值及其索引 (token_id)
        topk_values,topk_indices=torch.topk(E,k,dim=1)

        if mode=="generate":
            return topk_values,topk_indices

        elif mode=="print":
            # 转换为CPU numpy数组以便处理
            topk_values_np=topk_values.cpu().numpy()
            topk_indices_np=topk_indices.cpu().numpy()

            batch_results=[]
            # 遍历batch中的每个样本
            for i in range(E.size(0)):
                sample_results=[]
                token_ids=topk_indices_np[i].tolist()

                # 获取token文本 (批量解码提高效率)
                token_texts=self.tokenizer.convert_ids_to_tokens(token_ids)

                # 为当前样本组装结果
                for j in range(5):
                    prob=float(topk_values_np[i,j])
                    text=token_texts[j]
                    sample_results.append((prob,text))
                batch_results.append(sample_results)
            return batch_results


class CoTdecodeLogitsProcessor(OriginLogitsProcessor):
    def __init__(
        self,
        tokenizer,
        tokenizer_vocab_size,
        eot_token_id,
        current_length,
        paranum
    ):
        super().__init__(
            tokenizer,
            tokenizer_vocab_size,
            eot_token_id,
            current_length,
            paranum
        )
        self.index=0
        self.log=torch.zeros(paranum,device='cuda')
        self.flag=torch.ones(paranum,dtype=torch.bool,device='cuda')

    def __call__(self,input_ids: torch.LongTensor,modified_scores: torch.Tensor) -> torch.Tensor:
        if self.index==0:
            topk_values,topk_indices=torch.topk(modified_scores,self.paranum,dim=1)

            new_scores=torch.full_like(modified_scores,-float('inf'))
            batch_indices=torch.arange(self.paranum,device=modified_scores.device)
            token_indices=topk_indices[0]
            new_scores[batch_indices,token_indices[batch_indices]]=0
            modified_scores=new_scores

        else:
            modified_scores-=torch.logsumexp(modified_scores,dim=1,keepdim=True)
            modified_scores=super().__call__(input_ids,modified_scores)
            topk_values,_=torch.topk(modified_scores,2,dim=1)

            diffs=torch.exp(topk_values[:,0])-torch.exp(topk_values[:,1])
            # 向量化条件更新
            update_mask=modified_scores[:,151643]!=0
            self.log[update_mask]+=diffs[update_mask]

            ended_mask=~update_mask
            newly_ended_mask=ended_mask&self.flag
            if newly_ended_mask.any():
                self.log[newly_ended_mask]/=self.index
                self.flag[newly_ended_mask]=False
        self.index+=1
        return modified_scores

class CustomLogitsProcessor(OriginLogitsProcessor):
    def __init__(
        self,
        logits_setting,
        adapter_setting,
        interactive_mode
    ):
        super().__init__(
            logits_setting.tokenizer,
            logits_setting.tokenizer_vocab_size,
            logits_setting.eot_token_id,
            logits_setting.current_length,
            logits_setting.paranum
        )
        initial_hs1=1.0
        initial_hs2=logits_setting.data_proportion
        self.logits_setting=logits_setting

        self.last_hidden_state=None
        self.cur_beam_indices=None

        if logits_setting.history_label or logits_setting.observe_label:
            self.hs0=-torch.full((self.paranum,),math.log(self.tokenizer_vocab_size)+math.log(initial_hs1+initial_hs2)).cuda()
            self.hs1=torch.full((self.paranum,self.tokenizer_vocab_size),math.log(initial_hs1)).cuda()
            self.hs2=torch.full((self.paranum,self.tokenizer_vocab_size),math.log(initial_hs2)).cuda()

        if logits_setting.observe_label:
            self.hs0log=self.hs0.reshape((1,self.paranum))
            self.hs1log=torch.full((1,self.paranum),math.log(initial_hs1)).cuda()
            self.hs2log=torch.full((1,self.paranum),math.log(initial_hs2)).cuda()
            self.kllog=None
            self.etpy1=None
            self.etpy2=None

        if logits_setting.plot in ["guide","origin"]:
            self.plot_hs=[]
            self.plot_df=[]

        self.hsC=None

        self.adapter_setting=adapter_setting
        if logits_setting.state_norm_mode=="batch":
            state_vector_tensor=adapter_setting.state_vector_tensor
            self.state_mean=state_vector_tensor.mean(dim=0,keepdim=True)
            self.state_std=state_vector_tensor.std(dim=0,keepdim=True,unbiased=False)
            self.adapter_setting.state_vector_tensor=(state_vector_tensor-self.state_mean)/self.state_std


        self.frequency=torch.ones((self.tokenizer_vocab_size),dtype=torch.float32,device=self.adapter_setting.token_id_tensor.device)

        tmp_arr=self.adapter_setting.token_id_tensor.cpu().numpy()
        values,counts=numpy.unique(tmp_arr,return_counts=True)
        for val,cnt in zip(values,counts):
            self.frequency[val]=(math.pow(cnt,self.adapter_setting.frequency_bias))

        self.count=0
        self.interactive_mode=interactive_mode
        self.observe_num=5

    def __call__(self,input_ids: torch.LongTensor,scores: torch.FloatTensor) -> torch.Tensor:
        scores-=torch.logsumexp(scores,dim=1,keepdim=True)
        self.count+=1
        # self.cur_beam_indices=transformers.generation.utils.global_beam_indices
        if self.count>1 and (self.logits_setting.history_label or self.logits_setting.observe_label):
            if self.cur_beam_indices is None:
                C_tensor=torch.arange(self.paranum, dtype=torch.long).unsqueeze(1).expand(self.paranum,self.count)
            else:
                C_tensor=torch.stack([torch.stack(list(c_i)).squeeze(-1) for c_i in self.cur_beam_indices])
                if len(C_tensor.shape)==1:
                    C_tensor=C_tensor.reshape((self.paranum,1))

            idx1=C_tensor[:,-1].long()  # (batch,)
            idx2=input_ids[idx1,-1].long()  # (batch,)
            values1=self.hs1[idx1,idx2]  # (batch,)
            values2=self.hs2[idx1,idx2]  # (batch,)
            values1+=self.hs0
            values2+=self.hs0

            if self.logits_setting.observe_label:
                self.hs0log=self.hs0log[:,idx1]
                self.hs1log=self.hs1log[:,idx1]
                self.hs2log=self.hs2log[:,idx1]
                self.kllog=self.kllog[:,idx1]

                self.hs0log=torch.cat([self.hs0log,self.hs0.reshape((1,self.paranum))],dim=0)
                self.hs1log=torch.cat([self.hs1log,values1.reshape((1,self.paranum))],dim=0)
                self.hs2log=torch.cat([self.hs2log,values2.reshape((1,self.paranum))],dim=0)

            self.hs1=values1.unsqueeze(1).expand(-1,self.tokenizer_vocab_size)
            self.hs2=values2.unsqueeze(1).expand(-1,self.tokenizer_vocab_size)

        logits_dtype=scores.dtype


        E=self.ditstribution_calc(
            B=(self.last_hidden_state[:,-1,:]-self.state_mean)/self.state_std,
            logits_dtype=logits_dtype
        )

        if self.logits_setting.plot in ["guide","origin"]:
            self.plot_hs.append((self.last_hidden_state[:,-1,:]-self.state_mean)/self.state_std)
            if self.logits_setting.plot=="origin":
                return scores


        if self.logits_setting.history_label or self.logits_setting.observe_label:
            self.hs1=scores
            self.hs2=torch.log((1-self.logits_setting.inject_ratio)*E+self.logits_setting.inject_ratio*torch.exp(scores))


        # modified_scores=torch.logsumexp(torch.stack([scores,torch.log(E)+self.logits_setting.bias],dim=-1),dim=-1)
        # modified_scores-=torch.logsumexp(modified_scores,dim=1,keepdim=True)
        modified_scores=Adapter.function_3(scores,E,self.logits_setting.bias,self.adapter_setting.adapter_function[2])

        if self.interactive_mode=="test":
            top_token_list=[]
            for item in [torch.exp(scores),E,torch.exp(modified_scores)]:
                top_token_list.append(self.get_top_tokens(item,self.observe_num))
            self.print_top_token(top_token_list,self.observe_num)

        if self.logits_setting.observe_label:

            etpy1=-torch.sum(torch.exp(scores)*scores,dim=1)
            log_E=torch.log(E)
            log_E=torch.nan_to_num(log_E,nan=0.0,posinf=0.0,neginf=0.0)
            etpy2=-torch.sum(E*log_E,dim=1)
            kllog=torch.sum(torch.exp(self.hs1)*(self.hs1-self.hs2),dim=1).reshape((1,-1))

            if self.kllog==None:
                self.kllog=kllog
                self.etpy1=etpy1
                self.etpy2=etpy2
            else:
                self.kllog=torch.cat([self.kllog,kllog],dim=0)
                self.etpy1=torch.cat([self.etpy1,etpy1],dim=0)
                self.etpy2=torch.cat([self.etpy2,etpy2],dim=0)
        if self.logits_setting.history_label or self.logits_setting.observe_label:
            logsumexp_1=torch.logsumexp(self.hs1,dim=1)
            logsumexp_2=torch.logsumexp(self.hs2,dim=1)
            self.hs0=-torch.logsumexp(torch.stack([logsumexp_1,logsumexp_2],dim=-1),dim=-1)
            modified_scores+=self.hs0.unsqueeze(1).expand(-1,self.tokenizer_vocab_size)


        modified_scores=super().__call__(input_ids,modified_scores)

        return modified_scores


    def ditstribution_calc(
        self,
        B: torch.Tensor,
        logits_dtype
    ) -> torch.Tensor:
        A=self.adapter_setting.state_vector_tensor
        D=self.adapter_setting.token_id_tensor
        device=A.device
        assert B.device==device and D.device==device,"All inputs must be on the same device"

        C_float=Adapter.function_1(A,B,self.adapter_setting.adapter_function[0])
        C_float=Adapter.function_2(C_float,self.adapter_setting.dataset_topk,self.adapter_setting.adapter_function[1],self.adapter_setting.shape_idex)

        if self.count>1:
            modified_C=self.hsC*self.adapter_setting.shift_rate+C_float*(1-self.adapter_setting.shift_rate)
            C_float=modified_C

        if self.logits_setting.plot in ["guide","origin"]:
            self.plot_df.append(C_float)

        self.hsC=torch.cat((C_float[:,-1].unsqueeze(1),C_float[:,:-1]),dim=1)

        b=B.size(0)
        E=torch.zeros((b,self.tokenizer_vocab_size),dtype=logits_dtype,device=device)

        index=(D).unsqueeze(0).expand(b,-1).long()  # 构造索引矩阵 (b, n)
        E.scatter_add_(1,index,C_float)  # 高效向量化操作

        E*=self.frequency
        sum_E=E.sum(dim=1,keepdim=True)

        E=E/sum_E
        return E

    def mask_tensor(self):
        if self.adapter_setting.tmp_state_vector_tensor==None:
            self.adapter_setting.tmp_state_vector_tensor=self.adapter_setting.state_vector_tensor
            self.adapter_setting.tmp_token_id_tensor=self.adapter_setting.token_id_tensor

        self.adapter_setting.state_vector_tensor=torch.cat((
            self.adapter_setting.tmp_state_vector_tensor[0:self.adapter_setting.length_list[self.adapter_setting.mask]],
            self.adapter_setting.tmp_state_vector_tensor[self.adapter_setting.length_list[self.adapter_setting.mask+1]:]),dim=0)
        self.adapter_setting.token_id_tensor=torch.cat((
            self.adapter_setting.tmp_token_id_tensor[0:self.adapter_setting.length_list[self.adapter_setting.mask]],
            self.adapter_setting.tmp_token_id_tensor[self.adapter_setting.length_list[self.adapter_setting.mask+1]:]),dim=0)
        # self.adapter_setting.mask

        if self.logits_setting.state_norm_mode=="batch":
            self.state_mean=self.adapter_setting.state_vector_tensor.mean(dim=0,keepdim=True)
            self.state_std=self.adapter_setting.state_vector_tensor.std(dim=0,keepdim=True,unbiased=False)
            self.adapter_setting.state_vector_tensor=(self.adapter_setting.state_vector_tensor-self.state_mean)/self.state_std

    def print_top_token(self,r,num):
        for r_i in r:
            assert len(r_i)==self.paranum, "para_num Error"
        for i in range(self.paranum):
            for k in range(len(r)):
                for j in range(num):
                    print(f"{r[k][i][j][0]*100:.2f}",end="\t")
                    print(r[k][i][j][1],end="\t")
                    if len(r[k][i][j][1])<4:
                        print("\t\t",end="")
                    elif len(r[k][i][j][1])<8:
                        print("\t",end="")
                print("")

