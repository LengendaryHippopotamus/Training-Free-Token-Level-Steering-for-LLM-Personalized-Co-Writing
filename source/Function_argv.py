import time

class experiment_setting_class():
    def __init__(self):
        super().__init__()
        self.dataset_name=None
        # self.interactive_mode="interactive"#人机交互式伴写
        self.interactive_mode="experiment"#实验
        # self.interactive_mode="test"#用于观察和测试模型情况
        self.log_label=True
        self.overwrite=False
        self.experiment_num=1000000 #测试集中选取测试样例的最大值
        self.experiment_idex=None
        self.cross_validation_num=1
        self.cross_validation_idex=1
        self.batch_size=1
        self.actual_batch_size=1 #当重复次数不足时 actual_batch_size<batch_size
        self.guide_label=True
        self.guide_method="Nucleus" #用于切换各种baseline方法 CoTdecode origin server
        self.mask_mode=None
        self.reference_dataset_name=None
        self.reference_num=None
        self.prompt_mode=None
        self.few_shot_num=0
        # self.record_path="data/record"
        self.model_path=r"Qwen\Qwen2.5-0.5B"
        self.dataset_path=r"data\\"
        self.state_vector_path=r"data\\"
        self.parse_data_path=r"data\parse_data.json" #所有整理好的参考数据集
        self.record_path=r"data\record\\" #河南人民医院病历数据
        self.output_path=r"output\\"
        self.initial_length=20 #仅用于haic
        self.step_length=20 #仅用于haic
        self.ensure_word_complete=False
        self.cuda_is_available=True
        self.CoTdecodelog=[]
        self.plot_hs_log=[]
        self.plot_df_log=[]
        self.haic_dataset=["CDN"]


class generate_setting_class():
    def __init__(self):
        super().__init__()
        self.generate_token_length=40
        self.remaining_length=None
        self.num_beams=20
        self.num_return_sequences=5
        self.do_sample=True
        self.temperature=0.7
        self.top_p=0.95
        self.repeat_num=1
        self.observe_label=False #用于记录模型logits情况，调试时启用

class logits_setting_class():
    def __init__(self):
        super().__init__()
        self.data_proportion=1.0 #先验数据比
        self.state_norm_mode="batch"
        self.inject_ratio=0.4#分布注入，最初用于稳定累积数据概率，后用于调节数据矫正比例，现已废弃，改用bias
        self.bias=0
        self.history_label=False
        self.history_rate=1
        self.observe_label=False
        self.paranum=0
        self.tokenizer=None
        self.tokenizer_vocab_size=None
        self.current_length=0
        self.plot=None

class adapter_setting_class():
    def __init__(self):
        super().__init__()
        self.state_vector_tensor=None
        self.token_id_tensor=None
        # self.tokenizer_vocab_size=None #改到logits_setting中
        self.adapter_function=None
        self.length_list=None
        self.mask=None # None 无mask   self mask自己   auto mask不需要变
        self.tmp_state_vector_tensor=None # 用于在有mask的情况下记录原始tensor
        self.tmp_token_id_tensor=None
        # self.tokenizer=None
        self.dataset_topk=0.9
        self.shift_rate=0
        self.frequency_bias=0
        self.adapter_function=[1,1,1]
        self.shape_idex=1.75

class setting_combination_class():
    def __init__(self):
        super().__init__()
        self.experiment_setting=experiment_setting_class()
        self.generate_setting=generate_setting_class()
        self.logits_setting=logits_setting_class()
        self.adapter_setting=adapter_setting_class()
        self.logf=None
        self.start_time=time.time()
