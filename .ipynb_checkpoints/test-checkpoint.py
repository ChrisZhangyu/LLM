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
    # model_config = AutoConfig.from_pretrained(model_name, device=device, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(model_name, device=device, config=model_config, trust_remote_code=True, use_fast=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    llama_pattern = re.compile(r'llama', re.IGNORECASE)
    model_basename = "model"
    if  llama_pattern.match(model_name):
        model = LlamaForCausalLM.from_pretrained(model_name, config=model_config)
    else:
        # model = AutoModelForCausalLM.from_pretrained(model_name,
        #                                              config=model_config,
        #                                              trust_remote_code=True,
        #                                              )
        
        model = AutoGPTQForCausalLM.from_quantized(model_name,
                                                    revision="gptq-4bit-128g-actorder_True",
                                                    inject_fused_attention=False,
                                                    use_safetensors=True,
                                                    trust_remote_code=False,
                                                    device="cuda:0",
                                                    use_triton=False,
                                                    quantize_config=None)
    model.to(device)
    model.eval()
    
    # reflexion prompt
    
    code = '''
        def can_get_24(nums):
    if len(nums) == 1:
        return abs(nums[0] - 24) < 1e-6  # 检查是否非常接近 24

    # 枚举所有可能的两个数的组合
    for i in range(len(nums)):
        for j in range(len(nums)):
            if i != j:
                new_nums = [nums[k] for k in range(len(nums)) if k != i and k != j]

                # 对于每一种操作，递归调用函数并检查是否能得到 24
                if can_get_24(new_nums + [nums[i] + nums[j]]):
                    return True
                if can_get_24(new_nums + [nums[i] - nums[j]]):
                    return True
                if can_get_24(new_nums + [nums[i] * nums[j]]):
                    return True
                if nums[j] != 0 and can_get_24(new_nums + [nums[i] / nums[j]]):
                    return True

    return False
    '''
    
    prompt = f"you are a code expert. Please give a rate for this code: \n [code] \n  {code}"
    prompt_template=f'''### Human: {prompt}
                        ### Assistant:
                    '''
    input_ids = tokenizer.encode(prompt_template, return_tensors="pt").cuda()
    logging.getLogger().setLevel(logging.INFO)
    logging.info("输入指令prompt")
    start = time.time()
    # 去掉了, max_new_tokens=512
    rate = model.generate(inputs=input_ids, max_new_tokens=256)
    end = time.time()
    logging.info(f"耗时{end - start}")
    print("**********")
    print(tokenizer.decode(rate[0]))
    if device == torch.device('cuda') and hasattr(model, 'parallelize'):
        model.parallelize()

    return model, tokenizer

get_model_by_name('/root/autodl-fs/model/llama_2_70b_qlora_gptq', 'cuda')