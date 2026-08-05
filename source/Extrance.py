
import sys
import os
import time
import shutil
from source import main, Function_argv

class Setting_Controller():
    def __init__(self):
        super().__init__()
        self.setting_combination=Function_argv.setting_combination_class()
        self.experiment_setting=Function_argv.experiment_setting_class()
        self.generate_setting=Function_argv.generate_setting_class()
        self.logits_setting=Function_argv.logits_setting_class()
        self.adapter_setting=Function_argv.adapter_setting_class()

        self.path=dict(
            basic_output_path="output/",
            basic_model_path="llm/",
            basic_data_path="data/"
        )

    def combination(self):
        self.setting_combination.experiment_setting=self.experiment_setting
        self.setting_combination.generate_setting=self.generate_setting
        self.setting_combination.logits_setting=self.logits_setting
        self.setting_combination.adapter_setting=self.adapter_setting

    def line_for_server_experiment(self):
        main.mainline(self.setting_combination)

    def basic_setting(self):
        self.generate_setting.num_beams=1
        self.generate_setting.num_return_sequences=1
        self.generate_setting.temperature=0
        self.generate_setting.do_sample=False
        self.generate_setting.top_p=0.95
        self.generate_setting.generate_token_length=500
        self.logits_setting.data_proportion=1
        self.experiment_setting.step_length=10
        self.adapter_setting.dataset_topk=0.001

    def setting(self):
        self.experiment_setting.dataset_name=sys.argv[1]
        self.experiment_setting.reference_dataset_name=sys.argv[2]

        if len(sys.argv)>5:
            setting_string=sys.argv[5]
            self.excauter(setting_string)

        if self.experiment_setting.dataset_name in self.experiment_setting.haic_dataset:
            self.experiment_setting.dataset_path=self.path["basic_data_path"]+self.experiment_setting.dataset_name+"/"
        if self.experiment_setting.reference_dataset_name in self.experiment_setting.dataset_name:
            self.experiment_setting.state_vector_path=self.path["basic_data_path"]+sys.argv[2]+"/"+sys.argv[3]+"/"
        self.experiment_setting.model_path=self.path["basic_model_path"]+sys.argv[3]+"/"
        self.experiment_setting.output_path=self.path["basic_output_path"]+sys.argv[4]

    def excauter(self,setting_string):
        setting_s=setting_string.split(":")
        print(setting_s)
        for set_s in setting_s:
            s=set_s.split("=")
            attr_method={
                "log":"experiment_setting.log_label",
                "ow":"experiment_setting.overwrite",
                "mode":"experiment_setting.interactive_mode",
                "guide":"experiment_setting.guide_label",
                "refnum":"experiment_setting.reference_num",
                "method":"experiment_setting.guide_method",
                "fshot":"experiment_setting.few_shot_num",
                "prompt":"experiment_setting.prompt_mode",
                "num":"experiment_setting.experiment_num",
                "idex":"experiment_setting.experiment_idex",
                "estc":"experiment_setting.ensure_word_complete",
                "sample":"generate_setting.do_sample",
                "temp":"generate_setting.temperature",
                "rept":"generate_setting.repeat_num",
                "batch":"experiment_setting.batch_size",
                "gtl":"generate_setting.generate_token_length",
                "bias":"logits_setting.bias",
                "topk":"adapter_setting.dataset_topk",
                "hsCs":"adapter_setting.shift_rate",
                "fqcb":"adapter_setting.frequency_bias",
                "adptf":"adapter_setting.adapter_function",
                "shpx":"adapter_setting.shape_idex",
                "plot":"logits_setting.plot"
            }
            if s[0] in ["fshot","rept","batch","num","gtl","refnum"]:
                self.set_property_by_string(self,attr_method[s[0]],int(s[1]))
            elif s[0] in ["log","ow","guide","sample","estc"]:
                self.set_property_by_string(self,attr_method[s[0]],bool(int(s[1])))
            elif s[0] in ["mode","method","prompt","plot"]:
                self.set_property_by_string(self,attr_method[s[0]],s[1])
            elif s[0] in ["temp","bias","topk","hsCs","fqcb","shpx"]:
                self.set_property_by_string(self,attr_method[s[0]],float(s[1]))
            elif s[0] in ["adptf","idex"]:
                tmp=s[1][1:-1].split(",")
                self.set_property_by_string(self,attr_method[s[0]],[int(it) for it in tmp])
            elif s[0]=="beam":
                self.generate_setting.num_beams=int(s[1])
                self.generate_setting.num_return_sequences=int(s[1])
            elif s[0]=="server":
                if s[1] in ["107","108"]:
                    self.path=dict(
                        basic_output_path="output/",
                        basic_model_path="llm/",
                        basic_data_path="data/"
                    )

    def set_property_by_string(self,obj,prop_name,value):
        """通用的通过字符串设置对象属性的函数"""
        if '.' in prop_name:
            parts=prop_name.split('.')
            current=obj
            for part in parts[:-1]:
                current=getattr(current,part)
            setattr(current,parts[-1],value)
        else:
            setattr(obj,prop_name,value)
        return True


if __name__=="__main__":
    sc=Setting_Controller()
    sc.basic_setting()
    sc.setting()

    if os.path.exists(sc.experiment_setting.output_path):
        if not sc.experiment_setting.overwrite:
            raise FileExistsError(f"文件夹已存在: {sc.experiment_setting.output_path}")
        else:
            shutil.rmtree(sc.experiment_setting.output_path)
    os.makedirs(sc.experiment_setting.output_path, exist_ok=True)

    with open(sc.experiment_setting.output_path+"/1.txt","a") as logf:
        original_stdout=sys.stdout
        if sc.experiment_setting.log_label:
            sys.stdout=logf
            sc.setting_combination.logf=logf
        else:
            sc.setting_combination.logf=original_stdout

        if sc.experiment_setting.reference_dataset_name[0:6]=="IFEval":
            sc.experiment_setting.mask_mode="self"
        sc.combination()

        if sc.experiment_setting.cross_validation_num>1:
            for i in range(sc.experiment_setting.cross_validation_num):
                sc.setting_combination.experiment_setting.cross_validation_idex=i
                sc.setting_combination.experiment_setting.state_vector_path="data/"+sys.argv[2]+"/"+str(sc.experiment_setting.cross_validation_num)+"/"+str(i)+"/"+sys.argv[3]+"/"
                sc.line_for_server_experiment()
        else:
            sc.line_for_server_experiment()

        end_time=time.time()
        print(f"{end_time-sc.setting_combination.start_time}seconds")

        sys.stdout=original_stdout

