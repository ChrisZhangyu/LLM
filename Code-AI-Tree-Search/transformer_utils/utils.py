import logging
import re
import sys
import time
sys.path.append('../../')

import torch
import reflexion_prompt as prompt
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM

def get_model_by_name(model_name, device):
    model_config = AutoConfig.from_pretrained(model_name, device=device, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, device=device, config=model_config, trust_remote_code=True)
    llama_pattern = re.compile(r'llama', re.IGNORECASE)

    if  llama_pattern.match(model_name):
        model = LlamaForCausalLM.from_pretrained(model_name, config=model_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     config=model_config,
                                                     trust_remote_code=True,
                                                     )
    model.to(device)
    model.eval()

    # reflexion prompt
    # input_ids = tokenizer.encode(prompt.PY_REFLEXION_CHAT_INSTRUCTION_V2, return_tensors="pt").to(device)
    # logging.getLogger().setLevel(logging.INFO)
    # logging.info("输入指令prompt")
    # start = time.time()
    # model.generate(input_ids, max_new_tokens=2048)
    # end = time.time()
    # logging.info(f"耗时{end - start}")
    if device == torch.device('cuda') and hasattr(model, 'parallelize'):
        model.parallelize()

    return model, tokenizer
