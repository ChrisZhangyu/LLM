import logging
import re
import sys
import time
sys.path.append('../../')

import torch
import reflexion_prompt as prompt
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

def get_model_by_name(model_name, device):


    llama_pattern = re.compile(r'.*llama.*', re.IGNORECASE)
    start = time.time()
    if  llama_pattern.match(model_name):
        # model = LlamaForCausalLM.from_pretrained(model_name, config=model_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, add_bos_token=False)
        # codeLlama
        model = AutoGPTQForCausalLM.from_quantized(model_name,
                                                     use_cuda_fp16=False,
                                                     device_map="auto",
                                                     use_triton=False,

                                                     trust_remote_code=False,
                                                     use_safetensors=True,
                                                     inject_fused_attention=False,
                                                     revision="gptq-8bit-128g-actorder_True")
        # llama_QLora_70b
        # model = AutoGPTQForCausalLM.from_quantized(model_name,
        #                                            use_cuda_fp16=False,
        #                                            revision="gptq-4bit-128g-actorder_True",
        #                                            inject_fused_attention=False,
        #                                            use_safetensors=True,
        #                                            trust_remote_code=False,
        #                                            device="cuda:0",
        #                                            use_triton=False,
        #                                            quantize_config=None)
    else:
        model_config = AutoConfig.from_pretrained(model_name, device=device, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, device=device, config=model_config,
                                                  trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     config=model_config,
                                                     trust_remote_code=True,
                                                     )
    model.to(device)
    model.eval()
    end = time.time()
    print(f"加载模型耗时:{end - start}")

    if device == torch.device('cuda') and hasattr(model, 'parallelize'):
        model.parallelize()

    return model, tokenizer
