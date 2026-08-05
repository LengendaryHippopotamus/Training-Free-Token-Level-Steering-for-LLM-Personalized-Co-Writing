def create_safe_string(full_text,split_idx=50,tokenizer=None):
    """
    将文本在第 split_idx 个 token 附近安全切分，生成 prompt 和 ground_truth。
    """
    # 1. 获取所有 Token
    all_tokens=tokenizer.encode(full_text,add_special_tokens=False)
    assert len(all_tokens)>=split_idx
    current_idx=split_idx

    # 开始尝试寻找最佳切分点
    while current_idx>0:
        # --- 步骤 2: 尝试 Decode ---
        candidate_tokens=all_tokens[:current_idx]
        # errors='replace' 会把切坏的字节变成 ，方便我们检测
        candidate_text=tokenizer.decode(candidate_tokens,errors="replace")

        # --- 步骤 3: 检查末尾乱码 (Byte Level Check) ---
        if candidate_text.endswith('\ufffd'):
            # print(f"[Warn] Index {current_idx}: 切分导致末尾出现替换字符()，正在回退...")
            current_idx-=1
            continue

        # --- 步骤 4: 双重确认 Re-encode (Consistency Check) ---
        # 目的：确保 decode 出来的文本，重新 encode 后，Token 序列和原来一模一样。
        # 这样能避免“切分点正好把一个本来应该合并的词切开了”或者“Tokenizer 做了奇怪的 Normalization”

        re_encoded_ids=tokenizer.encode(candidate_text,add_special_tokens=False)

        if re_encoded_ids!=candidate_tokens:
            # 这种情况比较少见，但在 BBPE 中可能发生。
            # 例如：原 tokens 是 [A, B]，组合成文本是 "AB"。
            # 但如果 "A" 本身是一个包含空格后缀的 token，切分后导致空格处理逻辑变化，
            # 重新 encode 可能得到不同的 ID 序列。
            # print(f"[Warn] Index {current_idx}: Re-encode 后 Token 不一致。")
            # print(f"       原始: {candidate_tokens[-3:]} ...")
            # print(f"       新编: {re_encoded_ids[-3:]} ...")
            # print(f"       正在回退以寻找更稳定的切分点...")
            current_idx-=1
            continue

        # --- 通过所有检查，锁定切分点 ---
        break

    return current_idx

