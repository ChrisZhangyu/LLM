from default_policy.AbstractDefaultPolicy import AbstractDefaultPolicy


class TransformerDefaultPolicy(AbstractDefaultPolicy):
    def __init__(self,
                 tokenizer,
                 mode,
                 device,
                 k,
                 length_limitation,
                 env):
        super().__init__(k, length_limitation, env)


    def get_top_k_predict(self):
        pass

    def get_predict_sequence(self):
        pass
