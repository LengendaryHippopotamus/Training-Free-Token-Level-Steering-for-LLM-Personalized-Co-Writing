import torch

def nucleus_generate(
    model,
    tokenizer,
    inputs,
    custom_adapter_processor,
    generate_setting
):
    # model.config.output_hidden_states=True
    model.config.output_hidden_states=bool(custom_adapter_processor)  # 按需开启隐藏状态

    # def forward_hook(module,inputs,output):
    #     if output.hidden_states is not None:
    #         custom_adapter_processor.last_hidden_state=output.hidden_states[-1]
    # forward_handle=model.register_forward_hook(forward_hook)

    if custom_adapter_processor is not None:
        def forward_hook(module,input,output):
            if output.hidden_states is not None:
                custom_adapter_processor.last_hidden_state=output.hidden_states[-1]

        forward_handle=model.register_forward_hook(forward_hook)
    else:
        forward_handle=None


    try:
        # 根据是否使用处理器来配置参数
        generate_kwargs={
            **inputs,
            'max_new_tokens':min(generate_setting.generate_token_length,generate_setting.remaining_length),
            'num_beams':generate_setting.num_beams,
            'num_return_sequences':generate_setting.num_return_sequences,
            'do_sample':generate_setting.do_sample,
            'pad_token_id':tokenizer.eos_token_id,
            'output_scores':True,
            'return_dict_in_generate':True,
            'logits_processor':[custom_adapter_processor] if custom_adapter_processor else None
        }

        # 仅在采样时添加温度参数
        if generate_setting.do_sample:
            generate_kwargs.update({
                'temperature':generate_setting.temperature,
                'top_p':generate_setting.top_p
            })

        outputs=model.generate(**generate_kwargs)

        # print(outputs.sequences.shape)

    finally:
        if forward_handle:
            forward_handle.remove()

    if custom_adapter_processor and generate_setting.observe_label==True:

        torch.set_printoptions(precision=4,sci_mode=False)

        hs0log=custom_adapter_processor.hs0log
        hs1log=custom_adapter_processor.hs1log
        hs2log=custom_adapter_processor.hs2log
        etpy1=custom_adapter_processor.etpy1
        etpy2=custom_adapter_processor.etpy2

        probability_proportion=torch.sigmoid(custom_adapter_processor.hs1log-custom_adapter_processor.hs2log)
        KL_distribution=torch.clamp(custom_adapter_processor.kllog, min=0, max=1000)

        if custom_adapter_processor.paranum==1:
            hs0log=hs0log.reshape((-1,))
            hs1log=hs1log.reshape((-1,))
            hs2log=hs2log.reshape((-1,))
            etpy1=etpy1.reshape((-1,))
            etpy2=etpy2.reshape((-1,))

            probability_proportion=probability_proportion.reshape((-1,))
            KL_distribution=KL_distribution.reshape((-1,))

        print(hs0log)
        print(hs1log)
        print(hs2log)
        print(etpy1)
        print(etpy2)
        print(probability_proportion)
        print(KL_distribution)

    return outputs.sequences