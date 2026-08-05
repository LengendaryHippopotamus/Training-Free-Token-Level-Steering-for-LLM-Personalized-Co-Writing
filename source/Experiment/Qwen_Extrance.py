import sys
import numpy
from source.Experiment import Haic_qwen_line,Haic_component,Result_extract2

file_path_list=[]
date="20260129/"
for i in range(int(sys.argv[1]),int(sys.argv[2])+1):
    file_path_list.append(date+str(i)+"/")

for i,file_path_i in enumerate(file_path_list):
    try:
        file_path=r"output/"+file_path_i
        para={
            "sampling_num":10516,
            "guide_label":False,
            "guide_method":"nucleus",
            "repeat_num":1
        }
        exp_para=Result_extract2.extract_values(file_path+r"/1.txt",para)
        if not exp_para["guide_label"] and exp_para["guide_method"]=="Nucleus":
            exp_para["guide_method"]="origin"

        haic_r=Haic_qwen_line.haic_qwen_eval(
            file_path,
            sampling_num=exp_para["sampling_num"],
            guide_method=exp_para["guide_method"],
            repeat=exp_para["repeat_num"],
            url="http://localhost:"+sys.argv[3]+"/v1",
            max_worker=2048
        )
        haic_r.Qwen_embedding_line()
        # haic_r.load_socre()
        # print(file_path_i)
        # score=numpy.column_stack((
        #     Haic_component.A_calc(haic_r.score["Levenshtein"]),
        #     Haic_component.A_calc(haic_r.score["pinyin"]),
        #     haic_r.score["Jaccard"],
        #     haic_r.score["AvgWordVec"],
        #     haic_r.score["Qwen3embd"]
        # ))
        # print(score)
    except BaseException as e:
        print(repr(file_path_i),e)

