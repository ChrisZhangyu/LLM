import logging
import time
import warnings
from abc import abstractmethod

import torch
import numpy as np

from transformer_utils.cache import GPTTopKCache, GPTSeqCache


class DefaultPolicyHeuristic:
    def __init__(self, k, horizon, env):
        self.k = k
        self.horizon = horizon
        self.env = env
        self.sample_times = 0
        self.time_stamps = []  # time stamp when a new sample is generated

    @abstractmethod
    def get_predict_sequence(self, state, horizon=None):
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

    def get_predict_sequence(self, state, horizon=None, node=None):
        """
        Args:
            horizon: return a new sequence with this extra length
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

            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)
            # reflexion加入的地方
            feedback = node.info.get("reflexion_error") if node.info.get("reflexion_error") else "None error"
            feedback_prompt = f"This is the exception from your previous code：\n{feedback}"
            feedback_prompt_ids = self.tokenizer.encode(feedback_prompt, return_tensors="pt").to(self.device)
            input_ids = torch.cat((feedback_prompt_ids, input_ids), 1)
            # input_ids = self.get_input_with_reflexion_prompt(feedback, state)
            start_time = time.time()

            sample_mode = (self.ts_mode == 'sample')

            model_output = self.model.generate(
                input_ids,
                top_k=self.k,
                num_beams=(1 if sample_mode else self.num_beams),  # if sampling enabled, beam should always be 1
                num_return_sequences=self.num_beams,
                do_sample=sample_mode,
                early_stopping=True,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=True,
                max_new_tokens=horizon,
                use_cache=True  # huggingface default cache is always enabled
            )

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
                    if self.debug: print('top-k cache hit')
                    return top_k_info

            encoded_ids = state
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)
            start_time = time.time()
            # 将state作为输入传入模型，生成k个最可能的字符, generate方法有待研究
            model_output = self.model.generate(
                input_ids,
                top_k=self.k,
                num_beams=self.num_beams,
                early_stopping=True,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
            )
            # debug模式下会打印模型的运行时间
            logging.info('generate top-k time: ' + str(time.time() - start_time))
            # model_output的数据类型有待考证。根据代码推测，模型的输出包括token本身及token的score
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

    def get_input_with_reflexion_prompt(self,
                                        state,
                                        feedback="",
                                        ):
        prev_func_impl = self.seq_cache.get(reflexion=True)
        import reflexion_prompt as prompt
        feedback = self.tokenizer.decode(feedback[0])

        self_reflection_input = prompt.PY_SELF_REFLECTION_CHAT_INSTRUCTION_V2 + f'\n\n[function impl]:\n{prev_func_impl}\n\n[unit test results]:\n{feedback}\n\n[self-reflection]:'
        with open("self_reflection_input.txt", "w") as f:
            f.write(self_reflection_input)
        self_reflection_input = self.tokenizer.encode(self_reflection_input, return_tensors="pt").to(self.model.device)
        self_reflection = self.model.generate(self_reflection_input, max_new_tokens=1024, use_cache=True)
        #   解码成字符串
        self_reflection = self.tokenizer.decode(self_reflection[0])

        final_input = f"[previous impl]:\n{prev_func_impl}\n\n[unit test results from previous " \
                      f"impl]:\n{feedback}\n\n[reflection on previous impl]:\n{self_reflection}\n\n[partial code you have generated]{state}[improve" \
                      f"d impl]"

        with open("self_reflection_output.txt", "w") as f:
            f.write(self_reflection)

        final_input = self.tokenizer.encode(final_input, return_tensors="pt").to(self.model.device)

        return final_input
