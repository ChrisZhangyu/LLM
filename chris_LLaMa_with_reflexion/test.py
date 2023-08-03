import json
import os.path

from transformers import AutoModel, AutoTokenizer, AutoConfig

path = "../model/codegeex2-6b"


def get_model():
    model_config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(path, config=model_config, trust_remote_code=True)

    codeGeex = AutoModel.from_pretrained(path, config=model_config, trust_remote_code=True, device='cuda')
    codeGeex.eval()
    prompt = "# language: Python\n# write a bubble sort function\n"

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(codeGeex.device)
    output = codeGeex.generate(inputs, max_length=512, top_k=1)
    print(tokenizer.decode(output[0]))


def read_result(result_file):
    code = []
    results = [os.path.join(result_file, f) for f in os.listdir(result_file)]
    for result in results:
        if not result.endswith(".log"):
            with open(result, "r+", encoding="utf-8") as f:
                code.append(json.load(f))

    print(code[0]["codes"][0])
    print(code[0].keys())
    print(code[0]["rewards"])




if __name__ == '__main__':
    read_result("../Code-AI-Tree-Search/generate/results")
