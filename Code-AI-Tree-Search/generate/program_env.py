import copy
import json
import os
import warnings
from abc import abstractmethod, ABC
from collections import OrderedDict
from types import SimpleNamespace

from eval.compute_reward import compute_reward
from eval.generate_gpt_codes import get_output_str_from_state_for_apps


class ProgramEnv(ABC):
    """
    Code generation environment.

    State: a list of tokens.
    Action: a token (an integer).
    Reward: pass rate of the program (on the training set in training, and on the test set in testing).
    """

    def __init__(self, terminal_token, horizon):
        """
        Args:
            terminal_token: The token for the terminal action
            horizon: the maximum length including the prompt
        """
        self.terminal_token = terminal_token
        self.horizon = horizon

        # state -> reward
        # we may need to retrieve the states (programs) in the order they were saved, so use OrderedDict
        self.cached_reward = OrderedDict()
        self.cached_error = OrderedDict()

    """
        s是父节点的state, a是当前节点的action。代码生成里，action就是要生成的字符。state保存了已经生成的所有字符
    """

    def transition(self, s, a, is_model_dynamic=True):
        if isinstance(a, list):
            next_state = s + a
        else:
            next_state = s + [a]

        # 如果a是终止符，或者超过最大长度就停止
        if a == self.terminal_token or len(next_state) == self.horizon:
            # either the program finishes, or the state reaches the maximum length
            done = True
        else:
            done = False

        if done:
            # 生成下一个状态并计算奖励
            reward = self.get_reward(next_state)
        else:
            reward = 0  # no intermediate reward，没有即时奖励

        return next_state, reward, done

    def step(self, action):
        self.state, reward, done = self.transition(self.state, action)

        return self.state, reward, done, {}

    @abstractmethod
    def get_reward(self, s, mode='train'):
        """
        This needs to be defined for each dataset
        """
        pass

    def convert_state_to_program(self, s):
        """
        The state may be different from the program. This converts it back to executable program.
        """
        return s

    def equality_operator(self, s1, s2):
        return s1 == s2

    def get_complete_programs(self):
        """
        Return the list of complete programs reached so far.
        This can be found from the list of cached rewards.
        """
        return list(map(lambda x: list(x), self.cached_reward.keys()))


class APPSProgramEnv(ProgramEnv):
    """
    Code generation environment for APPS dataset.
    """

    def __init__(self, prob_path, tokenizer, model_name, horizon, public_test_cases=None):
        self.prob_path = prob_path
        self.tokenizer = tokenizer
        self.model = model_name
        self.public_test_cases = public_test_cases
        # code from generate_gpt_codes that generate paths for all essential files
        public_test_case_path = os.path.join(prob_path, "public_input_output.json")
        test_case_path = os.path.join(prob_path, "input_output.json")
        prompt_path = os.path.join(prob_path, "question.txt")
        starter_path = os.path.join(prob_path, "starter_code.py")
        solutions_path = os.path.join(prob_path, "solutions.json")

        if not os.path.exists(starter_path):
            starter_path = None
        if not os.path.exists(test_case_path):
            raise Exception("input_output.json missing so can't do testing. Invalid ProgramEnv.")
        if not os.path.exists(prompt_path):
            raise Exception("question.json missing. Invalid ProgramEnv.")
        if public_test_cases == 'desc' and not os.path.exists(public_test_case_path):
            raise Exception('using public test cases in problem description, but public test cases missing.')

        from eval.generate_gpt_codes import generate_apps_prompt
        # generate prompt to encode question text and an "ANSWER" prompt to the state
        # no need to load the full arglist here, it only needs to check the value of peeking (using default value 0.0 here)
        gpt_args = SimpleNamespace(peeking=0.0)
        state, _ = generate_apps_prompt(gpt_args, test_case_path, prompt_path, solutions_path, tokenizer, starter_path)


        self.init_prompt = copy.copy(state)

        self.state = self.tokenizer.encode(state)
        terminal_token = self.tokenizer.encode('<|endoftext|>')

        super(APPSProgramEnv, self).__init__(terminal_token=terminal_token, horizon=horizon)

    def convert_state_to_program(self, s):
        # 解码出字符
        s = self.tokenizer.decode(s)
        return get_output_str_from_state_for_apps(s)

    def get_canonical_state(self):
        raise NotImplementedError()

    # 根据当前的state计算奖励值
    def get_reward(self, s, mode='train'):
        """
        Returns:
            The reward of program in s.
        """
        if s is None:
            return 0
        # 如果state的序列已经计算过奖励值了直接返回，cache会保存已经计算过的state序列的奖励值
        if tuple(s) in self.cached_reward.keys() and mode == 'train':
            # cache rewards for training
            return self.cached_reward[tuple(s)], self.cached_error[tuple(s)]
        # 将状态转换成程序，状态现在是一堆数值，调用tokenizer的decode方法
        output_str = self.convert_state_to_program(s)
        # 计算奖励，prob_path是外部给的，
        #  这里的reward就是通过率，通过执行代码的测试用例计算出来
        reward, test_info = compute_reward(self.prob_path, output_str, mode=mode, public_test_cases=self.public_test_cases)
        if not isinstance(test_info, str):
            info = test_info[0]
            reflexion_error = test_info[1]
        else:
            reflexion_error = test_info

        # node.info['reflexion_feedback'] = reflexion_error

        if mode == 'train':
            self.cached_reward[tuple(s)] = reward
            self.cached_error[tuple(s)] = reflexion_error

        return reward, reflexion_error
