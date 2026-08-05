import numpy
import torch

import shutil
import os



from source.Experiment import Haic_line,Haic_component,Result_extract2

def ext(file_path_list,
        platform,mode="abstract",
        haic_node_simplify=True,
        generate_string_length=40,
        reference_length=30,
        gsl=None
        ):

    metirc_log="auto"
    from transformers import AutoTokenizer
    tokenizer_path={"h100":"Qwen2.5-0.5B/"}[platform]
    Qwen_tokenizer=AutoTokenizer.from_pretrained(tokenizer_path)

    for i,file_path_i in enumerate(file_path_list):
        file_path_log={
            "h100":r"output/"
        }
        file_path=file_path_log[platform]+file_path_i
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

        try:
            exp_para=Result_extract2.extract_values(file_path+r"/1.txt",para)

            if exp_para["guide_method"]=="Cloud_deepseek_v3.2":
                model_tokenizer=AutoTokenizer.from_pretrained("DeepSeek-V3.2/")
            elif exp_para["guide_method"]=="Cloud_Qwen3-32B":
                model_tokenizer=AutoTokenizer.from_pretrained("Qwen3-32B/")
            else:
                model_tokenizer=Qwen_tokenizer

            if not exp_para["guide_label"] and exp_para["guide_method"]!="CoTdecode":
                exp_para["guide_method"]="origin"

            # print(exp_para)
            # print(gsl=="auto",exp_para["dataset_name"]!="CDN")
            reflash=False
            if gsl=="auto":
                if exp_para["dataset_name"]!="CDN":
                    generate_string_length=40
                    reference_length=40
                else:
                    generate_string_length=80
                    reference_length=80
                    # reflash=True
                # print(generate_string_length,reference_length)

            token_level=True
            original_prompt=False
            if file_path_i<"20251217":
                haic_node_simplify=False
                token_level=False
            if file_path_i<"20251219":
                original_prompt=True

            data_path_log={"h100":"data/"}

            haic_r=Haic_line.haic_eval(
                file_path,
                dataset_name=exp_para["dataset_name"],
                generate_string_length=generate_string_length,
                reference_length=reference_length,
                sampling_num=exp_para["sampling_num"],
                guide_method=exp_para["guide_method"],
                # metirc_log="auto",
                # metirc_log=metirc_log,
                # metirc_log=["Jaccard"],
                metirc_log=["Qwen3embd_1"],
                # metirc_log=["Levenshtein","pinyin","Jaccard","AvgWordVec"],
                repeat=exp_para["repeat_num"],
                reflash=reflash,
                mode="abstract",
                length_label=False,
                haic_node_simplify=haic_node_simplify,
                token_level=token_level,
                original_prompt_label=original_prompt,
                tokenizer1=model_tokenizer,
                tokenizer2=Qwen_tokenizer,
                data_path=data_path_log[platform]
            )
            if metirc_log!=[]:
                haic_r.embedding_weights=torch.load("qwen_embeddings.pt")

            print(file_path_i)
            # print(mode=="abstract")
            if mode=="abstract":
                haic_r.haic_result_line()
                score=numpy.column_stack((
                    # Haic_component.A_calc(haic_r.score["Levenshtein"])*100/reference_length,
                    # Haic_component.A_calc(haic_r.score["Levenshtein"],average_label=False)*100,
                    Haic_component.B_calc(haic_r.score["Levenshtein"])*100,
                    # Haic_component.A_calc(haic_r.score["pinyin"])*100/reference_length,
                    # Haic_component.A_calc(haic_r.score["pinyin"],average_label=False)*100,
                    Haic_component.B_calc(haic_r.score["pinyin"])*100,
                    haic_r.score["Jaccard"]*100,
                    haic_r.score["AvgWordVec"]*100,
                    # haic_r.score["Qwen3embd"]*100
                ))
                m,n=score.shape
                for a in range(m):
                    for b in range(n):
                        # print(score[a,b],end="\t")
                        print(f'{score[a,b]:.{2}f}',end="\t")
                    print("")
                # print(haic_r.score["Qwen3embd"])
            elif mode=="detail":
                score_log=haic_r.detail_eval_trubo()
                if platform=="local":
                    import Haic_plot2
                    matrices=Haic_plot2.convert_list_to_matrices(score_log)
                    Haic_plot2.plot_macro_heatmaps(matrices,metric_names=["Levenshtein","pinyin","Jaccard","AvgWordVec"])
        except BaseException as e:
            print(file_path_i)
            print(e)


