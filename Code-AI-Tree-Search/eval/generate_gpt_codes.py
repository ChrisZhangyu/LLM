"""
Useful functions from original code generation script from APPS.
"""

import io
import json
import random

import os

from reindent import run as run_reindent

# okay with parallelization
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr,
        ret,
        config={
            "dry-run": False,
            "help": False,
            "to": 10,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 10,
            "all-tabs": False
        }
    )

    return ret.getvalue()


def generate_apps_prompt(args, test_case_path, prompt_path, solutions_path, tokenizer, starter_path=None):
    _input = "\nQUESTION:\n"
    # 这个表示ANSWER字符串在这里加还是在default_pi里加，因为如果要使用llama+反馈的方法需要在default_pi里加ANSWER
    add_answer = True
    # guanaco的 prompt的模板，注意与llama2完全不同，如果使用llama2的模板就会导致生成的字符全为unk
    import re
    llama_pattern = re.compile(r'.*llama.*', re.IGNORECASE)
    if llama_pattern.match(type(tokenizer).__name__):
        instruct = f'''You are a code expert.You will be given the problem description below QUESTION.You need to generate code to solve the QUESTION.Your code will be tested so don't test yourself, but make sure your code reads the input and prints the output correctly.Tests are performed through standard input and output, so your code must have something to read the input and print the result.You will have several chances to generate code for the same problem.If your previous code have errors, you should revise your code.Sometimes some code snippets are given, you need to complete it or you can rewrite the code to ensure that it can solve the problem without errors.The following are the format requirements of the code\n-----FORMAT-----:\n```python\nthe code to read input\nthe code to solve problem\nthe code to print result\n```'''

        _input = instruct + _input
        add_answer = False

    with open(prompt_path, "r", encoding="utf-8") as f:
        data = f.readlines()
        data = [item for item in data if item != '\n']
        data = "".join(data)
    _input += data
    if starter_path != None:
        with open(starter_path, "r") as f:
            data = f.readlines()
            data = "".join(data)
            data = "\n" + data  # + "\n"
        _input += data
    else:
        # _input += "\n\n"
        pass

    with open(test_case_path, "r") as f:
        data = json.load(f)
    if not data.get("fn_name"):
        _input += "\nUse Standard Input format"  # \n"
    else:
        _input += "\nUse Call-Based format"  # \n"

    # _input += "\n-----PREVIOUS CODE-----:\n"
    # _input += "\n-----EXCEPTION FROM YOUR PREVIOUS CODE-----:\n"
    # llama_gptq_70b
    # prompt_template = f'''### Human: {_input}\n### Assistant:'''
    # code_llama
    prompt_template = f'''[INST] {_input}\n[/INST]'''
    _input += "\nANSWER:\n"

    _input = prompt_template
    if args.peeking > 0.0:
        # Need to do some peeking.

        # Read one example solution
        with open(solutions_path, 'r') as f:
            sols = json.load(f)

        # Choose the shortest solution for the model to use.
        # This is so we can conserve tokens (1024 max)
        # sample_sol = min(sols, key=len)

        # # Add args.peeking% of that solution to the prompt
        # sample_sol_token_ids = tokenizer.encode(sample_sol, verbose=False)
        # num_to_keep = int(len(sample_sol_token_ids) * args.peeking)
        # sample_sol_token_ids = sample_sol_token_ids[:num_to_keep]
        # _input += tokenizer.decode(sample_sol_token_ids)

        # Alternatively take a random solution
        sample_sol = random.choice(sols)
        rand_sol = reindent_code(sample_sol)
        rand_sol = tokenizer.encode(rand_sol, verbose=False)
        tokens_taken = int(args.peek_frac * len(rand_sol))
        rand_sol = rand_sol[:tokens_taken]
        _input += tokenizer.decode(rand_sol)
    else:
        sample_sol = None

    return _input, sample_sol


def llama_prompt(func):
    def wrapper(s):
        # 如果使用llama相关模型则包装原来的函数，
        temp_s = func(s)
        import re
        pattern = r"python(.*?)`"
        code = re.findall(pattern, temp_s, re.DOTALL)
        complete_code = "\n".join(code)
        return complete_code
    return wrapper


@llama_prompt
def get_output_str_from_state_for_apps(s):
    """
    Get the code from the transformer output
    """
    # ANSWER后面可能保存的是代码
    if "ANSWER:" in s:
        s = s.split("ANSWER:\n")[1]

    return s.replace("<|endoftext|>", "")
