import time

import numpy as np
import torch

from default_policy.AbstractDefaultPolicy import AbstractDefaultPolicy
from generate.default_pi import APPSHeuristic
from transformer_utils.cache import GPTTopKCache, GPTSeqCache


class TransformerDefaultPolicy(AbstractDefaultPolicy):

    def __init__(self,
                 tokenizer,
                 model,
                 value_model,
                 device,
                 num_beams,
                 k,
                 length_limitation,
                 env,
                 reflexion,
                 sample_times,
                 sample_mode,
                 top_k_cache_steps = 0,
                 use_seq_cache = True,
                 ):
        super().__init__(k, length_limitation, env)
        self.sample_mode = sample_mode
        self.sample_times = sample_times

        self.use_seq_cache = use_seq_cache
        self.num_beams = num_beams
        self.device = device
        self.model = model
        self.value_model = value_model

        self.top_k_cache_steps = top_k_cache_steps
        self.top_k_cache = GPTTopKCache(k, cache_steps=top_k_cache_steps, tokenizer=tokenizer)
        self.seq_cache = GPTSeqCache()

    # 这里可以加入reflexion，来指导下一个action的生成
    def get_top_k_predict(self, state):
        self.model.to(self.device)
        if self.top_k_cache_steps > 0:
            top_k = self.top_k_cache.get(state)
            if top_k is not None:
                print("top_k cache hit")
                return top_k
        start_time = time.time()
        with torch.no_grad:
            input_ids = torch.LongTensor(state).unsqueeze(0).to(self.device)
            # 这里要添加reflexion,然后可以将reflexion添加为value函数或者作为reward

            output = self.model.generate(
                input_ids,
                top_k=self.k,
                num_beams=self.num_beams,
                early_stopping=True,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
            )
            print('generate top-k time: ' + str(time.time() - start_time))
            # model_output的数据类型有待考证。根据代码推测，模型的输出包括token本身及token的score
            top_k_scores, top_k_tokens = torch.topk(output.scores[0][0], k=self.k, sorted=True)
            # 对score归一化
            top_k_scores = torch.softmax(top_k_scores, dim=-1)

            return top_k_tokens.tolist(), top_k_scores.tolist()

    def get_predict_sequence(self, state):
        input_ids = torch.LongTensor(state).unsqueeze(0).to(self.device)
        if self.use_seq_cache:
            # seq_cache会存储之前生成过的token序列，节省计算时间
            output_ids = self.seq_cache.get(input_ids)
            if output_ids is not None:
                return output_ids
        with torch.no_grad:
            start_time = time.time()
            model_output = self.model.generate(
                input_ids,
                top_k=self.k,
                num_beams=(1 if self.sample_mode == "sample" else self.num_beams),
                # if sampling enabled, beam should always be 1
                num_return_sequences=self.num_beams,
                do_sample=self.sample_mode,
                early_stopping=True,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=True,
                max_length=self.length_limitation,
                use_cache=True  # huggingface default cache is always enabled
            )
            print('generate sequence time: ' + str(time.time() - start_time))

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
                self.seq_cache.add(state, output_ids)

            print('==== generated program ====')
            print(self.env.convert_state_to_program(output_ids))
            print('===========================')

            return output_ids

    def get_value(self, state):
        with torch.no_grad():
            input_ids = torch.LongTensor(state).unsqueeze(0).to(self.device)
            # 利用value model估计价值
            est_value = self.value_model(input_ids).logits.item()

            print(f"estimated value is {est_value}")
            return est_value

    def get_short_sequence(self, state):
        pass
