from abc import abstractmethod
from cmath import sqrt, log

from MCTS.TreeNode import Edge


class AbstractAgent:
    def __init__(self,
                 default_policy,
                 rollouts=100,
                 gamma=0.9,
                 ucb_constant=6.36396103068,
                 ucb_base=50.):
        self.ucb_base = ucb_base
        self.rollouts = rollouts
        self.default_policy = default_policy
        self.gamma = gamma
        self.ucb_constant = ucb_constant

    @abstractmethod
    def act(self, env, done):
        pass

    @abstractmethod
    def tree_policy(self, q, node, done) -> Edge:
        pass

    def ucb(self, q, node) -> Edge:
        return q + self.ucb_constant * sqrt(log(node.parent.visits)) / (1 + len(node.sampled_returns))

    def p_ucb(self, q, node) -> Edge:
        return q + self.ucb_constant * node.prob * sqrt(log(node.parent.visits)) / (1 + len(node.sampled_returns))

    def var_p_ucb(self, q, node) -> Edge:
        ucb_parameter = log((node.parent.visits + self.ucb_base + 1) / self.ucb_base) + self.ucb_constant
        return q + ucb_parameter * node.prob * sqrt(log(node.parent.visits)) / (1 + len(node.sampled_returns))


