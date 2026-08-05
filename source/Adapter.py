import math
import torch

def function_1(
    A: torch.Tensor,
    B: torch.Tensor,
    idex
):
    if idex==1:
        C=torch.matmul(B,A.T)
        return C.float()
    elif idex in [2,3,4]:
        A=A.float()
        B=B.float()
        # Step 2: 计算绝对差值张量C (广播机制自动处理维度扩展)
        C=torch.abs(B[:,None,:]-A[None,:,:])  # 结果形状: (b, n, p)
        if idex==2:
            Cp=C.mean(dim=-1)  # 沿最后一个维度(p)求平均
        elif idex==3:
            Cp=C.max(dim=-1).values  # 沿最后一个维度(p)取最大值
        elif idex==4:
            Cp=torch.norm(C,p=2,dim=-1)
        return torch.reciprocal(Cp)

        # C_mean=C.mean(dim=-1)  # 沿最后一个维度(p)求平均
        # C_max=C.max(dim=-1).values  # 沿最后一个维度(p)取最大值
        # C2=torch.norm(C,p=2,dim=-1)
        # if idex==2:
        #     return torch.reciprocal(C_mean)
        # elif idex==3:
        #     return torch.reciprocal(C_max)
        # elif idex==4:
        #     return


def function_2(
    C_float,
    dataset_topk,
    idex,
    shape_idex=None
):
    batch_size,nums=C_float.shape
    if idex in [1,2]:
        k=math.ceil(nums*dataset_topk)
        mask=torch.zeros_like(C_float,dtype=torch.bool)
        _,indices=torch.topk(C_float,k=k,dim=1)
        cols=indices.reshape(-1)
        rows=torch.arange(batch_size).repeat_interleave(k)
        mask[rows,cols]=True

        if idex==1:
            C_float=(C_float-C_float.mean(dim=1,keepdim=True))/C_float.std(dim=1,keepdim=True,unbiased=False)
            C_float=torch.sigmoid(C_float)
            C_float=C_float*mask
            return C_float
        else:
            C_masked=C_float*mask.float()  # 将mask转换为float类型便于计算
            eps=1e-8
            sum_masked=C_masked.sum(dim=1,keepdim=True)
            count_masked=mask.float().sum(dim=1,keepdim=True)
            mean_masked=sum_masked/(count_masked+eps)
            squared_diff=(C_float-mean_masked)**2
            squared_diff_masked=squared_diff*mask.float()
            sum_squared_diff=squared_diff_masked.sum(dim=1,keepdim=True)
            std_masked=torch.sqrt(sum_squared_diff/(count_masked+eps))
            C_float=(C_float-mean_masked)/(std_masked+eps)

            C_float=torch.sigmoid(C_float)
            C_float=C_float*mask.float()
            return C_float

    if idex in [3,4,5,6,7]:
        P=create_topk_tensor(nums,dataset_topk,mode=idex-2,shape_idex=shape_idex)
    elif idex==8:
        P=create_exponential_tensor(nums,dataset_topk,shape_idex)
    return align_tensor_to_reference(C_float, P)


def align_tensor_to_reference(C_float,P):
    # 将C_float的每一行按值的大小顺序对齐到参考向量P
    batch_size,nums=C_float.shape
    # 1. 对C的每一行进行排序，得到排序后的索引（从大到小）
    # sorted_indices的形状: (batch_size, nums)
    sorted_indices=torch.argsort(C_float,dim=1,descending=True)
    # 2. 对参考向量P进行排序（从大到小）
    P_sorted,_=torch.sort(P,descending=True)
    # 3. 将排序后的P扩展到与C相同的batch大小
    # P_expanded的形状: (batch_size, nums)
    P_expanded=P_sorted.unsqueeze(0).expand(batch_size,-1)
    # 4. 创建一个与C形状相同的空张量，用于存放对齐后的结果
    aligned_C=torch.zeros_like(C_float)
    # 5. 使用scatter_函数将P_expanded的值按照排序索引放入aligned_C
    # 使用torch.arange(batch_size)创建行索引
    row_indices=torch.arange(batch_size).unsqueeze(1).expand(-1,nums)
    # scatter操作：将P_expanded的值放到aligned_C的对应位置
    # dim=1表示在列维度上进行scatter
    aligned_C.scatter_(1,sorted_indices,P_expanded)
    return aligned_C


def create_topk_tensor(l,topk,mode=1,shape_idex=None):
    device="cuda"
    P=torch.zeros(l,dtype=torch.float32,device=device)
    k=int(l*topk)+1
    indices=torch.arange(k,dtype=torch.float32,device=device)
    if mode==1: # 线性衰减: 从1到0
        values=1-indices/(k-1 if k>1 else 1)
    elif mode==2: # cosine-1: cos在[0, pi/2]区间  映射: indices从0到k-1 -> 角度从0到pi/2
        x=indices/(k-1 if k>1 else 1)  # 归一化到[0, 1]
        theta=x*(math.pi/2)  # 映射到[0, pi/2]
        values=torch.cos(theta)
    elif mode==3: # cosine-2: cos在[pi/2, pi]区间  映射: indices从0到k-1 -> 角度从pi/2到pi
        x=indices/(k-1 if k>1 else 1)  # 归一化到[0, 1]
        theta=math.pi/2+x*(math.pi/2)  # 映射到[pi/2, pi]
        values=torch.cos(theta)+1
    elif mode==4: # cosine-3: cos在[0, pi]区间  映射: indices从0到k-1 -> 角度从0到pi
        x=indices/(k-1 if k>1 else 1)  # 归一化到[0, 1]
        theta=x*math.pi  # 映射到[0, pi]
        values=(torch.cos(theta)+1)/2  # 从1到0
    elif mode==5:
        values=(1-indices/(k-1))**shape_idex

    values[0]=1.0
    values[-1]=0.0
    P[:k]=values
    return P


def create_exponential_tensor(l,topk,shape_idex):
    indices=torch.arange(0,l,dtype=torch.float32,device="cuda")
    a=shape_idex/(l*topk)
    P=torch.exp(-a*indices)
    return P


def function_3(log_p1,p2,bias,idex):
    if idex==1:
        modified_scores=torch.logsumexp(torch.stack([log_p1,torch.log(p2)+bias],dim=-1),dim=-1)
    elif idex==2:
        modified_scores=torch.minimum(log_p1,torch.log(p2)/bias)
    modified_scores-=torch.logsumexp(modified_scores,dim=1,keepdim=True)
    return modified_scores
