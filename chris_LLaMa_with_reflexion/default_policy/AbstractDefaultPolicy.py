from abc import abstractmethod


class AbstractDefaultPolicy:
    def __init__(self, k, length_limitation, env):
        self.k = k
        self.length_limitation = length_limitation
        self.env = env

    @abstractmethod
    def get_predict_sequence(self, state):
        pass

    @abstractmethod
    def get_top_k_predict(self, state):
        pass

    @abstractmethod
    def get_value(self, state):
        pass
