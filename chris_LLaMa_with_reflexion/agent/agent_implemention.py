from typing import Optional

from MCTS.TreeNode import Edge
from MCTS.mcts import mcts_procedure
from agent.AbstractAgent import AbstractAgent


class Agent(AbstractAgent):
    def __init__(self, action_space: Optional[list, None],
                 default_policy,
                 reuse_tree,
                 tree_policy_type = "var_p_ucb"):

        super().__init__(default_policy)
        if not self.default_policy and action_space is None:
            raise Exception(f"There is no action can be taken because default_policy and is{self.default_policy} ")

        self.tree_policy_type = tree_policy_type
        self.reuse_tree = reuse_tree
        self.root = None

    def tree_policy(self, q, node, done) -> Edge:

        if self.tree_policy_type == "ucb":
            return self.ucb(q, node)
        elif self.tree_policy_type == "p_ucb":
            return self.p_ucb(q, node)
        elif self.tree_policy_type == "var_p_ucb":
            return self.var_p_ucb(q, node)
        else:
            raise Exception(f"Unknown tree policy {self.tree_policy_type}")

    def act(self, env, done, term_cond=None):
        root = self.root if self.reuse_tree else None
        opt_act, self.root = mcts_procedure(self, env,
                                            root=root,
                                            rollout_weight=1,
                                            terminate_condition=term_cond)
        return opt_act
