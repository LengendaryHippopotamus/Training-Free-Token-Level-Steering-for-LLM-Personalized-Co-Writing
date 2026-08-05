from transformers import AutoModelForCausalLM,AutoTokenizer

def load_model(model_path,load_model=True):
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    if load_model:
        model=AutoModelForCausalLM.from_pretrained(model_path,device_map="cuda",torch_dtype="auto")
        print(model_path,model.config._name_or_path,model.device)
    else:
        model=None

    tokenizer.pad_token=tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    vocab_size,eot_token=get_config(model_path)
    return model,tokenizer,vocab_size,eot_token

def get_config(model_path):
    model_dir=model_path.rstrip('/').split('/')[-1]
    # 预定义配置映射
    configs={
        'Qwen2.5-0.5B':(151936,151643),
        'Qwen2.5-1.5B':(151936,151643),
        'Qwen2.5-3B':(151936,151643),
        'Qwen2.5-7B':(152064,151643),
        'Qwen2.5-Math-7B':(152064,151643),
        'Qwen3-0.6B-Base':(151936,151643),
        'Qwen3-1.7B-Base':(151936,151643),
        'Qwen3-4B-Base':(151936,151643),
        'Qwen3-8B-Base':(151936,151643),
        'Llama-3.2-3B':(128256,128001)
    }
    # 查找匹配的配置
    for key,config in configs.items():
        if key in model_dir:
            return config
    # 默认配置
    print("vocab_size error")
    return (151936,151643)

