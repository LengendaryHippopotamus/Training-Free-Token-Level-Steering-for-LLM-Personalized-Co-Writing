# 用于从记录文件中提取信息

def extract_values(file_path,keyword_defaults):
    """
    从文件中提取指定关键字的值，如果未找到则使用默认值

    参数:
    file_path: 文件路径
    keyword_defaults: 字典，键为关键字名称，值为默认值

    返回:
    字典，包含找到的值或默认值
    """
    # 初始化结果字典，所有值初始为默认值
    results=keyword_defaults.copy()

    with open(file_path,'r') as file:
        for line in file:
            # 跳过空行
            if not line.strip():
                continue

            # 分割行内容
            parts=line.split()
            if len(parts)<2:
                continue
            key=parts[0]
            value_str=parts[1]

            # 检查是否是我们要找的关键字
            if key in results:
                # 获取默认值以确定类型
                default_value=keyword_defaults[key]
                # 根据默认值的类型转换字符串值
                # try:
                if isinstance(default_value,bool):
                    if value_str.lower() in ('true','yes','1'):
                        results[key]=True
                    elif value_str.lower() in ('false','no','0'):
                        results[key]=False
                    else:
                        results[key]=bool(int(value_str))
                elif isinstance(default_value,int):
                    results[key]=int(value_str)
                elif isinstance(default_value,float):
                    results[key]=float(value_str)
                else:
                    # 默认为字符串类型
                    results[key]=value_str
                # except (ValueError,TypeError):
                #     # 转换失败时保持默认值
                #     pass

            # 如果所有值都已找到，提前退出循环
            if all(results[key]!=keyword_defaults[key] for key in keyword_defaults):
                break

    return results
    # return [results[key] for key in keyword_defaults]


if __name__=="__main__":


    file_path=r"1.txt"

    para={
        "dataset_name":"hmr-haic",
        "question_num":75,
        "sampling_num":10516,
        "guide_label":False,
        "guide_method":"nucleus",
        "initial_length":20,
        "step_length":10,
        "generate_token_length":50,
        "repeat_num":1,
        "batch_size":1,
        "bias":0.0,
        "history_rate":1.0,
        "shift_rate":0.0,
        "frequency_bias":0.0
    }
    exp_para=extract_values(file_path,para)
    print(exp_para)