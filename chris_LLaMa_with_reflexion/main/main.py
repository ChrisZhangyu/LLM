import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

from generator.generator import Generator
from generator.generator_factory import get_generator
from prompt.prompt_factory import prompt_factory

model_path = '../model'
tokenizer_path = '../model/tokenizer.model'


def train():
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    model = LlamaForCausalLM.from_pretrained(model_path)
    model.to('cuda')
    prompt = 'Q: What is the largest animal?\nA:'
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generation_output = model.generate(
        input_ids=input_ids, max_new_tokens=32
    )
    print(tokenizer.decode(generation_output[0]))


def reflexion_main():

    gen = get_generator("LLaMa")


if __name__ == '__main__':
    pass
