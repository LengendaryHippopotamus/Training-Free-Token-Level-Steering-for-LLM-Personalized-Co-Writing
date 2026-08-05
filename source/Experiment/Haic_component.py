# 用于记录一些辅助函数
import math
import numpy
import torch

def A_calc(A_batch,average_label=True):
    """
    批量计算面积，输入形状为(m, 41, 31)
    返回形状为(m,)的面积数组
    """
    # 获取每个样本的第一行，形状为(m, 31)
    first_rows=A_batch[:,0,:]
    # 计算每个样本的列最小值，形状为(m, 31)
    col_mins=numpy.min(A_batch,axis=1)
    # 计算比值，注意处理分母为0的情况
    # 只计算第1-30列（原代码中索引1-30，对应比值索引0-29）
    # 创建掩码避免除零错误
    denominator=first_rows[:,1:]  # 形状为(m, 30)
    numerator=col_mins[:,1:]  # 形状为(m, 30)
    # 使用np.where处理分母为0的情况
    with numpy.errstate(divide='ignore',invalid='ignore'):
        ratio=numpy.where(denominator!=0,1-numerator/denominator,1)
    if average_label:
        # 沿列方向求和，得到每个样本的面积
        areas=numpy.sum(ratio,axis=1)
        return areas
    else:
        return ratio[:,-1]

def B_calc(A_batch):
    # m=A_batch.shape[1]
    n=A_batch.shape[2]
    return 1-A_batch[:,n-1,n-1]/A_batch[:,0,n-1]

def C_calc(A_batch):
    return numpy.argmin(A_batch,axis=1)

def score_at_k(x,m,n):
    # 给定m个数x_1,x_2,...,x_m，从中随机挑选n个数（不重复，n<=m），计算挑选到的最大的数的期望
    # 当x取0或1时退化为pass_at_k
    # 对数组进行排序（从小到大）
    x_sorted=numpy.sort(x)
    # 计算期望
    expectation=0.0
    # 遍历k从n到m
    for k in range(n,m+1):
        # k在排序数组中的索引是k-1（因为索引从0开始）
        x_k=x_sorted[k-1]
        # 组合数C(k-1, n-1)
        comb=math.comb(k-1,n-1)
        expectation+=x_k*comb
    total_combinations=math.comb(m,n)
    expectation/=total_combinations
    return expectation


def score_at_k_table(x,y,m,n):
    # 给定m个数x_1,x_2,...,x_m，从中随机挑选n个数（不重复，n<=m），计算挑选到的最大的数的期望
    # 用于Levenshtein等需要计算另一组数的情况
    y_sorted=y[numpy.argsort(x)]
    # expectation=numpy.zeros((
    #     self.generate_string_length+1,
    #     self.reference_length+1
    # ),dtype=float)
    expectation=numpy.zeros_like(y[0], dtype=float)
    # 遍历k从n到m
    for k in range(n,m+1):
        comb=math.comb(k-1,n-1)
        expectation+=y_sorted[k-1]*comb
    total_combinations=math.comb(m,n)
    expectation/=total_combinations
    return expectation


def score_at_k_vectorized(X,b,m):
    """
    计算从每行的m个数中随机挑选j个数的最大值的期望
    参数:X: numpy数组, 形状 (b, m)
    返回:Y: numpy数组, 形状 (b, m), Y[i, j] = score_at_k(X[i], m, j+1)
    """
    # 1. 对每行排序 (按最后一个轴排序)
    X_sorted=numpy.sort(X,axis=1)
    # 2. 预计算组合数矩阵 C (m×m上三角矩阵)
    #    C[k, j] = C(k, j) 当 k >= j，否则 0
    C=numpy.zeros((m,m))
    for k in range(m):
        # 使用列表推导式填充每行，避免内层循环
        C[k,:k+1]=[math.comb(k,j) for j in range(k+1)]
    # 3. 计算分子：矩阵乘法一步完成所有(b,j)组合的计算
    #    Y_temp[i, j] = sum_{k=j}^{m-1} X_sorted[i, k] * C(k, j)
    Y_temp=X_sorted@C  # 形状 (b, m)
    # 4. 预计算分母：C(m, j+1) 对于 j=0..m-1
    denom=numpy.array([math.comb(m,j+1) for j in range(m)],dtype=float)
    # 5. 广播除法得到最终结果
    Y=Y_temp/denom
    return Y


def get_z_from_cummax(x,y):
    # 将y转换为torch张量以便使用gather函数
    y_tensor=torch.from_numpy(y)
    # 使用cummax获取累积最大值及其索引
    # cummax返回(values, indices)
    # indices[i,j] 表示在x[i,:j+1]中最大值的索引
    _,indices=torch.cummax(x,dim=1)
    # 使用gather函数根据索引从y中取值
    # 我们需要在第二维上收集数据
    # 注意：gather要求indices的形状与输出形状相同
    z_tensor=torch.gather(y_tensor,1,indices)
    # 转换回numpy数组
    z=z_tensor.numpy()
    return z


def batch_truncate_and_decode(
    text: str,
    truncation_lengths,
    tokenizer
):
    """
    批量截取token并还原为文本
    Args:
        text: 原始文本
        truncation_lengths: 需要截取的长度列表
    """
    # 1. 将文本tokenize
    tokens=tokenizer.encode(text,add_special_tokens=False)

    # 3. 批量获取截取后的token序列
    # 使用numpy加速操作
    all_truncated_tokens=[]

    for length in truncation_lengths:
        truncated=tokens[:length]
        all_truncated_tokens.append(truncated)

    # 4. 批量解码
    # 使用errors='replace'可以让解码继续，但会标记无效字节
    decoded_texts=[]
    for truncated_tokens in all_truncated_tokens:
        decoded=tokenizer.decode(
            truncated_tokens,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True
        )
        decoded_texts.append(decoded)
    # 5. 创建长度到解码文本的映射
    length_to_text=dict(zip(truncation_lengths,decoded_texts))

    # 6. 检查半个字符问题并输出结果
    length_log=[]
    for length in truncation_lengths:
        # actual_length=min(length,max_length)
        text_result=length_to_text[length]

        # 检查半个字符问题
        # 方法1: 检查Unicode替换字符 � (U+FFFD)
        last_token_str=text_result[-1]
        if '�' in text_result or last_token_str.startswith('##') or '�' in last_token_str:
            print(f"提示: 截取前{length}个token时可能出现了半个字符 (检测到Unicode替换字符)")

        # 方法2: 检查解码后的字节再编码是否会产生错误
        try:
            # 将文本编码为字节再解码，检查是否有解码错误
            encoded_bytes=text_result.encode('utf-8')
            # 尝试严格解码
            encoded_bytes.decode('utf-8',errors='strict')
        except UnicodeDecodeError:
            print(f"警告: 截取前{length}个token时检测到不完整的UTF-8序列")
        length_log.append(len(text_result))
    return length_log