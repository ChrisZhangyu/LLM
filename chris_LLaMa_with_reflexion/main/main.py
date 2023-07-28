import time

from MCTS.mcts import mcts_procedure
from util.util_factory import *

model_path = '../model'
tokenizer_path = '../model/tokenizer.model'


def reflexion_mcts_main(args, env, agent, start_time):
    term_cond = lambda: agent.default_policy.sample_times > args.max_sample_times or time.time() - start_time > args.time_limit \
                        or (args.early_stop and max(env.cached_rewards.values()) == 1.)

    try:
        done = False
        s = env.state

        if len(s) >= args.horizon:
            return None, None
        else:
            # run mcts. a bit hacky, but this will run mcts simulation. we do not need to take any action
            agent.act(env, done, term_cond=term_cond)
    except Exception as e:
        if args.debug:
            raise e
        else:
            print("Unexpected exception in generating solution")
            return None, None

        # these may not be assigned, set default values

    states = env.get_complete_programs()

    return states, {'sample_times': agent.default_policy.sample_times, }


if __name__ == '__main__':
    pass
