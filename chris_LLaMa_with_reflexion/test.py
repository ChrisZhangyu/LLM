import json
import logging
import os.path

import numpy as np

import reflexion_prompt as rp
from transformers import AutoModel, AutoTokenizer, AutoConfig, LlamaForCausalLM
import multiprocessing


path = "../model/open_llama_3b_v2"


def get_model():
    model_config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(path, config=model_config, trust_remote_code=True)

    llama = LlamaForCausalLM.from_pretrained(path, config=model_config).to("cuda")
    # llama = AutoModel.from_pretrained(path, config=model_config, trust_remote_code=True).to("cuda")
    llama.eval()
    prompt = rp.PY_SELF_REFLECTION_CHAT_INSTRUCTION_V2
    # prompt = "what is the biggest animal in the world"
    inputs = tokenizer.encode(prompt,  return_tensors="pt").to(llama.device)
    output = llama.generate(inputs,  top_k=1)
    output_str = tokenizer.decode(output[0])
    with open("input.txt", "w+") as f:
        lines = prompt.split(".")
        lines = [line + "\n" for line in lines]
        f.writelines(lines)
    with open("output.txt", "w+") as f:
        lines = output_str.split(".")
        lines = [line + "\n" for line in lines]
        f.writelines(lines)
    print(output_str)


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


def add(a, b):

    while True:
        x = 1
    return a + b


def reflexion_test(a, b):
    logging.getLogger().setLevel(logging.INFO)
    manager = multiprocessing.Manager()
    error_queue = multiprocessing.Queue()

    result = manager.list()

    task = multiprocessing.Process(target=add, args=(a, b))
    task.start()
    logging.info("程序开始执行，等待5秒钟")
    task.join(5)

    if task.is_alive():
        logging.info("程序超时，杀死子进程")
        task.kill()
    if not result:
        logging.info("程序执行完毕，但结果为空")
        print(result)
    logging.info("查看子进程异常信息")
    if not error_queue:
        error = error_queue.get(block=False)
        print(error)
        logging.info("程序出现异常，返回异常")
    logging.info("主进程退出")


def get_human_eval():
    data_path = "../reflexion/programming_runs/benchmarks/humaneval-py_hardest50.jsonl"
    with open(data_path, "r") as f:
        for line in f:
            file = json.loads(line)
            print(file["prompt"])
            break


def get_reward():
    r1 = np.float64(1.213123123)
    r2 = "None error"
    return r1, r2


def delete_log():
    path = "../Code-AI-Tree-Search/generate/results"
    files = os.listdir(path)
    log_files = [i for i in files if i.endswith(".log")]
    for file in log_files:
        os.remove(os.path.join(path, file))


if __name__ == '__main__':
    # try:
    #     estimate, reflexion_error = get_reward()
    # except Exception as e:
    #     raise e
    delete_log()
