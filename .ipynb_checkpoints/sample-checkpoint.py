from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_name_or_path = "/root/autodl-fs/model/llama_2_70b_qlora_gptq"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        revision="gptq-4bit-128g-actorder_True",
       inject_fused_attention=False,
        use_safetensors=True,
        trust_remote_code=False,
        device="cuda:0",
        quantize_config=None)
"""
To download from a specific branch, use the revision parameter, as in this example:

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        revision="gptq-4bit-32g-actorder_True",
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=False,
        device="cuda:0",
        quantize_config=None)
"""

prompt = "Tell me about AI"
prompt_template=f'''### Human: {prompt}
### Assistant:
'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
print(tokenizer.decode(output[0]))

# Inference can also be done using transformers' pipeline

# Prevent printing spurious transformers error when using pipeline with AutoGPTQ
# logging.set_verbosity(logging.CRITICAL)

# print("*** Pipeline:")
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=512,
#     temperature=0.7,
#     top_p=0.95,
#     repetition_penalty=1.15
# )

# print(pipe(prompt_template)[0]['generated_text'])
