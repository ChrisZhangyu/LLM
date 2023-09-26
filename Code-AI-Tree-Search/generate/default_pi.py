import logging
import time
import warnings
from abc import abstractmethod
import random
from typing import Optional, Union
import re
import torch
import numpy as np

from dyna_gym.agents.mcts import DecisionNode
from transformer_utils.cache import GPTTopKCache, GPTSeqCache


class DefaultPolicyHeuristic:
    def __init__(self, k, horizon, env):
        self.k = k
        self.horizon = horizon
        self.env = env
        self.sample_times = 0
        self.time_stamps = []  # time stamp when a new sample is generated

    @abstractmethod
    def get_predict_sequence(self, state, node=None, horizon=None):
        pass

    @abstractmethod
    def get_top_k_predict(self, state):
        pass

    def clean_up(self, new_state):
        # implement this if need to do anything after each token is generated
        pass


class APPSHeuristic(DefaultPolicyHeuristic):
    def __init__(self,
                 tokenizer,
                 model,
                 k,
                 num_beams,
                 test_all_beams,
                 horizon,
                 device,
                 env,
                 value_model=None,
                 new_token_num=None,
                 use_seq_cache=False,  # disable all caching by default
                 use_prompt_cache=False,
                 top_k_cache_steps=0,
                 ts_mode='best',
                 debug=False):
        super(APPSHeuristic, self).__init__(k=k, horizon=horizon, env=env)

        self.tokenizer = tokenizer
        self.k = k
        self.num_beams = num_beams
        self.test_all_beams = test_all_beams
        # 长度
        self.horizon = horizon
        self.new_token_num = new_token_num
        self.device = device
        self.env = env

        self.use_seq_cache = use_seq_cache
        self.use_prompt_cache = use_prompt_cache  # todo
        self.top_k_cache_steps = top_k_cache_steps
        self.ts_mode = ts_mode

        self.debug = debug

        self.model = model
        self.value_model = value_model

        self.use_value = (self.value_model is not None)

        if self.ts_mode == 'sample' and self.use_seq_cache:
            warnings.warn("Cannot use sequence caching in sample mode, disabling it.")
            self.use_seq_cache = False

        if self.use_value and self.new_token_num is None:
            warnings.warn("Using a value function but not setting a shorter planning horizon (args.new_token_num)."
                          "Why using a value function?")

        self.model.to(device)
        if self.use_value:
            self.value_model.to(device)

        if device == torch.device('cuda'):
            if hasattr(self.model, 'parallelize'):
                self.model.parallelize()
            if self.value_model is not None and hasattr(self.model, 'parallelize'):
                self.value_model.parallelize()

        self.top_k_cache = GPTTopKCache(k, cache_steps=top_k_cache_steps, tokenizer=tokenizer)
        self.seq_cache = GPTSeqCache()
        self.prompt_key_values = None

        self.terminal_token = self.env.terminal_token
        self.feedback = "None error"
        self.previous_code = ""

    def get_short_horizon_sequence(self, state):
        """
        Returns:
            predicted sequence, but only with up to self.new_token_num new tokens.
            This uses self.get_predict_sequence.
        """
        # 根据当前结点一路预测到底，但是有限制，新的代码token数量小于new_token_num
        # add length of prompt and existing program
        horizon = len(state) + self.new_token_num
        # don't exceed the length of Transformer input
        horizon = min(horizon, self.horizon)

        return self.get_predict_sequence(state, horizon=horizon)

    def get_predict_sequence(self, state, node=None, horizon=None):
        """
        Args:
            state: 当前结点所处环境的状态
            horizon: return a new sequence with this extra length
            node_or_previous_code: 可能是之前生成的代码，也可能是MCTS的当前结点，取决于两种方案
        Returns:
            Get the most likely sequence starting from state.
        """
        with torch.no_grad():

            encoded_ids = state  # as a list
            if self.use_seq_cache:
                # seq_cache会存储之前生成过的token序列，节省计算时间
                output_ids = self.seq_cache.get(encoded_ids)
                if output_ids is not None:
                    return output_ids
            if horizon is None:
                horizon = self.horizon

            # 如果有exception_str的字段，说明需要exception
            exception_str = "-----EXCEPTION FROM YOUR PREVIOUS CODE-----:"
            input_prompt = self.tokenizer.decode(encoded_ids)
            # 找到exception字段
            if exception_str in input_prompt:
                input_prompt_list = input_prompt.split(exception_str)
                match = re.search(r'line (\d+)', self.feedback)
                if match is not None:
                    line_number = int(match.group(1)) - 18
                    self.feedback = re.sub(r'line \d+', f'line {line_number}', self.feedback)
                input_prompt = input_prompt_list[0] + f"\n{exception_str}\n{self.feedback}\n" + input_prompt_list[1]

            # 如果有previous_str，说明需要之前的code
            previous_str = "-----PREVIOUS CODE-----:"
            # 找到previous字段
            if previous_str in input_prompt:
                input_prompt_list = input_prompt.split(previous_str)
                # 如果node不是None，说明需要用node的父节点生成的代码当做previous字段，否则用模型上一次生成的代码做previous字段
                if node is not None and node.parent.parent is not None:
                    code = self.tokenizer.decode(node.parent.parent.state)
                else:
                    # if isinstance(self.previous_code, str):
                    # code = "There is no code please generate new code" if len(self.previous_code) == 0 else self.previous_code
                    code = self.previous_code
                    # else:
                    #     code = self.tokenizer.decode(self.previous_code)
                input_prompt = input_prompt_list[0] + f"\n{previous_str}\n{code}\n" + input_prompt_list[1]
            # if self.feedback != "None error":
            #     reflexion = self.get_reflexion_prompt()
            encoded_ids = self.tokenizer.encode(input_prompt)
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)

            start_time = time.time()
            sample_mode = (self.ts_mode == 'sample')
            # torch.cuda.empty_cache()
            model_output = self.model.generate(
                inputs=input_ids,
                top_k=self.k,
                num_beams=(1 if sample_mode else self.num_beams),  # if sampling enabled, beam should always be 1
                num_return_sequences=self.num_beams,
                do_sample=sample_mode,
                early_stopping=True,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=True,
                max_new_tokens=512,
                use_cache=True  # huggingface default cache is always enabled
            )
            # print(self.tokenizer.decode((model_output[0][0])))
            if self.top_k_cache_steps > 0:
                if hasattr(model_output, 'beam_indices'):
                    # beam search output

                    self.top_k_cache.add(input_ids, model_output.sequences, model_output.scores,
                                         beam_indices=model_output.beam_indices)
                else:
                    self.top_k_cache.add(input_ids, model_output.sequences, model_output.scores)

            if self.debug: print('generate sequence time: ' + str(time.time() - start_time))

            output_ids_list = model_output.sequences.tolist()

            if len(output_ids_list) > 1 and self.test_all_beams:
                # if got multiple output_ids using beam search, and going to test all beams (which takes more time)
                # pick the one that has the highest reward
                # 如果根据束搜索得到了多个结果，那么输出奖励值最高的
                cand_rewards = [self.env.get_reward(output_ids) for output_ids in output_ids_list]
                output_ids = output_ids_list[np.argmax(cand_rewards)]
            else:
                output_ids = output_ids_list[0]

            if self.use_seq_cache:
                self.seq_cache.add(encoded_ids, output_ids)

            self.sample_times += 1
            self.time_stamps.append(time.time())

            if self.debug:
                print('==== generated program ====')
                print(self.env.convert_state_to_program(output_ids))
                print('===========================')
            temp_code = self.env.convert_state_to_program(output_ids)
            self.previous_code = temp_code if temp_code else "There is no previous code"
            return output_ids

    def get_value(self, state):
        with torch.no_grad():
            encoded_ids = state
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)
            # 利用value model估计价值
            est_value = self.value_model(input_ids).logits.item()

            if self.debug:
                print(f"estimated value is {est_value}")

            return est_value

    def get_top_k_predict(self, state):
        """
        Returns:
            A list of k most likely tokens generate in state (descending in their scores)
            The probability of each action
        """
        # 这里终于调用了语言模型，利用语言模型预测k个最可能的action(也就是token，代码生成中每步的动作就是产生token)
        with torch.no_grad():
            #
            if self.top_k_cache_steps > 0:
                # 缓存了前k个token
                top_k_info = self.top_k_cache.get(state)
                # 缓存没有会返回None，如果不为None说明缓存击中
                if top_k_info is not None:
                    print('top-k cache hit')
                    return top_k_info

            encoded_ids = state
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)
            start_time = time.time()
            # 段级蒙特卡洛
            # node_length = 6

            # 随机长度
            # node_length = random.randint(3, 10)
            #
            node_length = 1

            # 将state作为输入传入模型，生成k个最可能的字符, generate方法有待研究

            model_output = self.model.generate(
                inputs=input_ids,
                top_k=self.k,
                num_beams=self.num_beams,
                early_stopping=True,
                max_new_tokens=node_length,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
            )
            # debug模式下会打印模型的运行时间
            logging.info('generate top-k time: ' + str(time.time() - start_time))
            # model_output的数据类型有待考证。根据代码推测，模型的输出包括token本身及token的score
            if node_length > 1:
                # model_output.scores的维度为(10, 3, 32000)中间那个3代表了num_beams。源代码每次都选择num_beams的第一个，
                # 而后从词典中(即32000的维度上)选出概率最高的3个
                # 这里的temp_list是长度为10的list，每个list的项是一个(2, 3, 3)的tensor。第一个(1, 3, 3)是3个beam中每个beam的top3最大的token的scores值，
                # 第二个(1, 3, 3)是3个beam中每个beam的top3最大的token的id值，
                # 最后要组成2个(10, 3)的list，根据源代码，每次都选择3个beam分支的第一个
                temp_list = [torch.topk(item, k=self.k, sorted=True, dim=1) for item in model_output.scores]
                top_k_scores = [item[0][0] for item in temp_list]
                top_k_tokens = [item[1][0] for item in temp_list]
                # 对score求平均并归一化
                top_k_scores = torch.softmax(torch.stack(top_k_scores, dim=0).sum(dim=0) / node_length, dim=0)
                top_k_tokens = torch.stack(top_k_tokens, dim=0).T
                return top_k_tokens.tolist(), top_k_scores
            else:
                top_k_scores, top_k_tokens = torch.topk(model_output.scores[0][0], k=self.k, sorted=True)
                # 对score归一化
                top_k_scores = torch.softmax(top_k_scores, dim=-1)
                return top_k_tokens.tolist(), top_k_scores.tolist()

    def clean_up(self, new_state):
        if self.use_seq_cache:
            # clear hashed sequences that are not consistent with new_state
            self.seq_cache.clear(new_state)

        if self.top_k_cache_steps > 0:
            self.top_k_cache.clear(new_state)

    def get_reflexion_prompt(self
                             ,
                             ):

        reflexion_instruct = f'''### Human:\n[PREVIOUS CODE]:\n{self.previous_code} \n[EXCEPTION FROM YOUR PREVIOUS CODE]:\n{self.feedback}\n\nThe above is the code you generated last time, as well as the errors it generated and the pass rate on the test cases.Please provide suggestions for modification, dont provide code.\n'''
        _input = reflexion_instruct + "\n### Assistant:"
        input_ids = self.tokenizer.encode(_input)

        input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)

        sample_mode = (self.ts_mode == 'sample')

        model_output = self.model.generate(
            inputs=input_ids,
            top_k=self.k,
            num_beams=(1 if sample_mode else self.num_beams),  # if sampling enabled, beam should always be 1
            num_return_sequences=self.num_beams,
            do_sample=sample_mode,
            early_stopping=True,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_scores=True,
            max_new_tokens=512,
            use_cache=True  # huggingface default cache is always enabled
        )
        reflexion = self.tokenizer.decode(model_output[0][0])

        return reflexion