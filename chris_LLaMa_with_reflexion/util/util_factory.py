"""
    生成器负责产生代码和文本反馈
"""
from generator.generator import Generator
from prompt.prompt_factory import prompt_factory
import transformers


def get_generator(model_type: str):
    return Generator(prompt_factory("Python"), )


def get_executor():
    pass


def get_environment():
    pass
