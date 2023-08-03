import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def get_model_by_name(model_name, device):
    model_config = AutoConfig.from_pretrained(model_name, device=device, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, device=device, config=model_config, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 config=model_config,
                                                 device=device,
                                                 trust_remote_code=True,
                                                 )

    if device == torch.device('cuda') and hasattr(model, 'parallelize'):
        model.parallelize()

    return model, tokenizer
