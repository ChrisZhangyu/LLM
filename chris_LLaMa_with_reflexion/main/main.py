from MCTS.mcts import mcts_procedure
from util.util_factory import *

model_path = '../model'
tokenizer_path = '../model/tokenizer.model'


def reflexion_mcts_main(args):
    gen = get_generator("LLaMa")
    exe = get_executor()
    env = get_environment()

    # mcts_procedure(env, exe, gen)


if __name__ == '__main__':
    pass
