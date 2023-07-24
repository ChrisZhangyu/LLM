"""
MCTS Algorithm

Required features of the environment:
env.state
env.action_space
env.transition(s ,a , is_model_dynamic)
env.equality_operator(s1, s2)
"""
import random
import itertools
import warnings

from tqdm import tqdm

import dyna_gym.utils.utils as utils
from gym import spaces
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt

from eval.utils import hierarchy_pos


# 用于计算Q(s, a), sampled_returns存储了所有
def chance_node_value(node, mode = "best"):
    """
    Value of a chance node
    """
    if len(node.sampled_returns) == 0:
        return 0

    if mode == "best":
        # max return (reasonable because the model is deterministic?)
        return max(node.sampled_returns)
    elif mode == "sample":
        # Use average return
        return sum(node.sampled_returns) / len(node.sampled_returns)
    else:
        raise Exception(f"Unknown tree search mode {mode}")


def combinations(space):
    if isinstance(space, spaces.Discrete):
        return range(space.n)
    elif isinstance(space, spaces.Tuple):
        return itertools.product(*[combinations(s) for s in space.spaces])
    else:
        raise NotImplementedError


def mcts_tree_policy(ag, children):
    return random.choice(children)


def mcts_procedure(ag, tree_policy, env, done, root = None, rollout_weight = 1., term_cond = None, ts_mode = "best"):
    """
    Compute the entire MCTS procedure wrt to the selected tree policy.
    Funciton tree_policy is a function taking an agent + a list of ChanceNodes as argument
    and returning the one chosen by the tree policy.
    """
    decision_node_num = 0
    if root is not None:
        # if using existing tree, making sure the root is updated correctly
        assert root.state == env.state
    else:
        # create an empty tree
        # 树节点类
        root = DecisionNode(None, env.state, ag.action_space.copy(), done, dp=ag.dp, id=decision_node_num)
        decision_node_num += 1

    # make sure the rollout number is at least one, and is at most ag.rollouts
    # rollout_weight是什么意思不知道
    if rollout_weight > 1:
        warnings.warn("How come rollout_weight > 1? Setting to 1.")
    # 对rollouts截断，weight表示缩放因子,值小于1
    rollouts = np.clip(int(ag.rollouts * rollout_weight), 1, ag.rollouts)

    print("Performing rollouts.")
    for _ in tqdm(range(rollouts)):
        # uct_multiple_exp不会有term_cond变量，uct_exp才会有
        if term_cond is not None and term_cond():
            break
        rewards = []  # Rewards collected along the tree for the current rollout
        node = root  # Current node
        terminal = done

        # Selection
        select = True
        # 整体流程
        # 先判断是dNode还是cNode,如果是dNode,就不停的向下选择值最大的结点。直到当前结点时cNode。
        # 如果走到了cNode，那么就与环境交互，把父节点的状态加上当前结点的action作为新状态，并计算新状态的奖励值，进入下一个判断
        #       遍历当前结点的子节点，如果子节点的状态和新状态相同，那么进入到子节点中，如果都不相同，则退出选择阶段，说明当前的状态确实是新状态
        while select:
            # decisionNode表示已有树节点，选择阶段直接选择策略函数返回值最大的那个
            if type(node) == DecisionNode:  # DecisionNode
                if node.is_terminal:
                    # 当前是终止符的话直接退出选择阶段，进入124行
                    select = False  # Selected a terminal DecisionNode
                else:
                    # uct_tree_policy，p_uct_tree_policy，var_p_uct_tree_policy三种策略中的一个
                    # MCTS的两种策略，tree_policy和default_policy,
                    # tp用来选择已经完全扩展的节点，dp用来选择没有完全扩展的节点，即还有子节点没探索过
                    node = tree_policy(ag, node.children)  # Shun: move down the tree, node is now a ChanceNode

            else:  # ChanceNode
                # 进入这个分支表明当前结点时chanceNode，cNode的特点是只有action和action的概率。state_表示采取在父节点的state下采取当前action到达当前的state_p
                # cNode表示可能的下一个节点
                # 这里的reward是通过率，state_p是新状态，由父节点state经过action产生。action在树里表示边，实际存储在cNode里
                # state_p 就是 node.parent.state + node.action
                state_p, reward, terminal = env.transition(node.parent.state, node.action, ag.is_model_dynamic)
                rewards.append(reward)

                new_state = True
                # 如果当前节点没有子节点，就是一个完全没有探索过的节点，那么直接就到124行结束选择阶段
                # 根据整体的代码逻辑来看,cNode最多只有一个dNode结点，因为state_p = parent.state + node.action
                # 如果cNode不变,那么parent.state不会变，而且每个cNode结点只存储了一个action,因此action也不会变
                for i in range(len(node.children)):
                    # Note that for deterministic transitions, node.children contains at most one child
                    # 这里判断子节点的状态和state_p是否一样，一样的话说明state_p的子节点已经产生了，进入到子节点中，并且将new_state设置为false表示仍然在选择阶段
                    # 如果不一样就遍历下一个子节点，如果所有的子节点都没有出现state_p状态
                    # 说明state_p的状态还没有创建节点，进入到扩展阶段，也就是说当前的节点还没有完全扩展完。
                    if env.equality_operator(node.children[i].state, state_p):
                        # Shun: state_p already in the tree, point node to the corresponding Decision Node
                        node = node.children[i]
                        new_state = False
                        break
                # 如果不是新状态还要继续停留在选择阶段，到达这一步有两种情况，
                # 1.没有子节点，说明完全没有扩展
                # 2.采取action产生的状态子节点中没有，说明没有扩展完
                if new_state:
                    select = False  # Selected a ChanceNode

        # Expansion
        # If node is a decision node, then it must be a terminal node, do nothing here
        # 根据前面的逻辑，走到这一步时的node指向的肯定是cNode结点，如果是dNode，说明已经走到了终止符号，即前面的代码走到了第96行，然后跳转到当前行
        if type(node) == ChanceNode:
            # 为当前节点创建子节点，state_p为新状态，子节点类型都是cNode
            node.children.append(
                DecisionNode(node, state_p, ag.action_space.copy(), terminal, dp=ag.dp, id=decision_node_num))
            decision_node_num += 1
            node = node.children[-1]

        # Evaluation
        # now `rewards` collected all rewards in the ChanceNodes above this node
        assert (type(node) == DecisionNode)
        state = node.state
        if ag.dp is None:
            t = 0
            estimate = 0
            while (not terminal) and (t < ag.horizon):
                # 从动作空间中采样
                action = env.action_space.sample()
                # 产生下一个状态
                state, reward, terminal = env.transition(state, action, ag.is_model_dynamic)
                # 如果没有价值模型，那么用传统的方法估计价值，gamma是衰减因子，越往后衰减越严重
                estimate += reward * (ag.gamma ** t)
                t += 1
        else:
            if not node.is_terminal:
                # 这里可以选择根据计算价值还是计算奖励，奖励是短期的，价值是长期的
                if ag.dp.use_value:
                    # 对代码生成限制长度，这样可以不用将完整的程序生成
                    state = ag.dp.get_short_horizon_sequence(state)
                    # this state may not be a complete program, call the pre-trained value model to get an estimate
                    # 利用value model估计价值，传统的价值计算是通过从当前节点一直模拟到终点，然后计算结果，算出这一条路径的价值，本文是直接用transformer估计
                    estimate = ag.dp.get_value(state)
                else:
                    # follow the default policy to get a terminal state
                    state = ag.dp.get_predict_sequence(state)
                    # reward就是通过率
                    estimate = env.get_reward(state)

                    # save this information for demo
                    node.info['complete_program'] = state
            else:
                # the rewards are defined on terminating actions, the terminal states have no rewards
                estimate = 0

        # Backpropagation
        # 反向传播，根据当前扩展出的节点的价值和访问次数，更新父节点的访问次数和价值
        node.visits += 1
        node = node.parent
        assert (type(node) == ChanceNode)
        while node:
            if len(rewards) != 0:
                # 价值估计
                estimate = rewards.pop() + ag.gamma * estimate
            node.sampled_returns.append(estimate)
            node.parent.visits += 1
            node = node.parent.parent

        # should finish backpropagating all the rewards back
        assert len(rewards) == 0

    return max(root.children, key=lambda n: chance_node_value(n, mode=ts_mode)).action, root


class DecisionNode:
    """
    Decision node class, labelled by a state

    Args:
        dp: default policy, used to prioritize and filter possible actions
    """

    def __init__(self, parent, state, possible_actions = [], is_terminal = False, dp = None, id = None):
        self.id = id
        self.parent = parent
        self.state = state
        self.is_terminal = is_terminal
        if self.parent is None:  # Root node
            self.depth = 0
        else:  # Non root node
            self.depth = parent.depth + 1
        # 如果默认策略是空，那么就根据传进来的actions进行随机选择，否则按照生成策略去生成可能的actions。
        # possible_actions来源于各个agent的action_space变量，在创建agent时由创建者指定
        if dp is None:
            self.possible_actions = possible_actions
            random.shuffle(self.possible_actions)

            # if no default policy is provided, assume selection probability is uniform
            self.action_scores = [1.0 / len(self.possible_actions)] * len(self.possible_actions)
        # 根据默认策略进行action的生成，这里就是论文中说的transformer的beam_search算法
        else:
            # get possible actions from dp
            # default policy suggests what children to consider
            # 由Transformer产生备选字符及其概率
            top_k_predict, top_k_scores = dp.get_top_k_predict(self.state)

            self.possible_actions = top_k_predict
            self.action_scores = top_k_scores

        # populate its children
        # 根据备选的k个action构建子节点
        self.children = [ChanceNode(self, (act, score)) for act, score in
                         zip(self.possible_actions, self.action_scores)]

        self.explored_children = 0
        # this decision node should be visited at least once, otherwise p-uct makes no sense for this node
        self.visits = 1
        # used to save any information of the state
        # we use this for saving complete programs generated from it
        self.info = {}

    def is_fully_expanded(self):
        return all([child.expanded() for child in self.children])


class ChanceNode:
    """
    Chance node class, labelled by a state-action pair
    The state is accessed via the parent attribute
    """

    def __init__(self, parent, action_and_score):
        self.parent = parent
        self.action = action_and_score[0]
        # 下一个节点的概率，由语言模型生成，这里传统的mcts可能会采取不同的采样算法来缩小搜索空间
        self.prob = action_and_score[1]  # the probability that this action should be token, provided by default policy
        self.depth = parent.depth

        self.children = []
        self.sampled_returns = []

    def expanded(self):
        return len(self.children) > 0


class MCTS(object):
    """
    MCTS agent
    """

    def __init__(self, action_space, rollouts = 100, horizon = 100, gamma = 0.9, is_model_dynamic = True):
        if type(action_space) == spaces.discrete.Discrete:
            self.action_space = list(combinations(action_space))
        else:
            self.action_space = action_space
        self.n_actions = len(self.action_space)
        self.rollouts = rollouts
        self.horizon = horizon
        self.gamma = gamma
        self.is_model_dynamic = is_model_dynamic

    def reset(self, p = None):
        """
        Reset the attributes.
        Expect to receive them in the same order as init.
        p : list of parameters
        """
        if p == None:
            self.__init__(self.action_space)
        else:
            utils.assert_types(p, [spaces.discrete.Discrete, int, int, float, bool])
            self.__init__(p[0], p[1], p[2], p[3], p[4])

    def display(self):
        """
        Display infos about the attributes.
        """
        print('Displaying MCTS agent:')
        print('Action space       :', self.action_space)
        print('Number of actions  :', self.n_actions)
        print('Rollouts           :', self.rollouts)
        print('Horizon            :', self.horizon)
        print('Gamma              :', self.gamma)
        print('Is model dynamic   :', self.is_model_dynamic)

    def act(self, env, done):
        opt_act, _, = mcts_procedure(self, mcts_tree_policy, env, done)
        return opt_act


def update_root(ag, act, state_p):
    root_updated = False
    for chance_node in ag.root.children:
        if act == chance_node.action:
            for decision_node in chance_node.children:
                if decision_node.state == state_p:
                    ag.root = decision_node
                    root_updated = True
                    break

    if not root_updated:
        raise Exception("root update fails, can't find the next state, action pair in tree.")


def pre_order_traverse(
        decision_node: DecisionNode,
        decision_node_fn = lambda n, d: None,
        chance_node_fn = lambda n, d: None,
        depth = 0):
    """
    Postorder traversal of the tree rooted at state
    Apply fn once visited
    """
    decision_node_fn(decision_node, depth)

    for chance_node in decision_node.children:
        chance_node_fn(chance_node, depth)
        for next_decision_node in chance_node.children:
            pre_order_traverse(next_decision_node, decision_node_fn, chance_node_fn, depth + 1)


def get_all_decision_nodes(root: DecisionNode):
    """
    Get all decision nodes in the tree
    """
    decision_nodes = []
    pre_order_traverse(root, decision_node_fn=lambda n, d: decision_nodes.append(n))
    return decision_nodes


def plot_tree(root: DecisionNode, env, filename):
    """
    Plot the tree rooted at root
    """
    tokenizer = env.tokenizer

    def printer(node: ChanceNode, depth):
        # print the average return of the *parent* of this state
        # (this is easier to implement than printing all its children nodes)
        print("\t" * depth, repr(tokenizer.decode(node.action)), 'p', node.prob, 'q', chance_node_value(node),
              'len(returns)', len(node.sampled_returns))

    pre_order_traverse(root, chance_node_fn=printer)

    # plot the tree
    G = nx.DiGraph()
    G.add_node(root.id, label='<PD>')

    def add_node(node: ChanceNode, depth):
        if len(node.children) > 0:
            child_id = node.children[0].id
            parent_id = node.parent.id

            G.add_node(child_id)
            G.add_edge(parent_id, child_id, label=repr(tokenizer.decode(node.action)))

    pre_order_traverse(root, chance_node_fn=add_node)

    plt.figure(figsize=(15, 15))

    pos = hierarchy_pos(G, root=root.id)
    nx.draw(G, pos, with_labels=True)

    edge_labels = nx.get_edge_attributes(G, 'label')
    # plot labels on the edges horizontally
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, rotate=False)

    plt.savefig(filename + '.pdf', format="pdf")
    plt.close()


def convert_to_json(root: DecisionNode, env, selected_act):
    """
    Save the information of children of root into a list.
    Does not distinguish layers. So works when the tree only expands one level.
    """
    ret = []

    def get_info(node: ChanceNode, depth):
        if node.action == env.terminal_token:
            # terminal state has no complete_program attribute, since the program is already complete
            complete_program = env.convert_state_to_program(node.children[0].state)
        else:
            complete_program = env.convert_state_to_program(node.children[0].info['complete_program'])

        info = {'token': env.tokenizer.decode(node.action),
                'state': env.convert_state_to_program(node.children[0].state),
                'selected': node.action == selected_act,
                'score': chance_node_value(node),
                'complete_program': complete_program}
        ret.append(info)

    pre_order_traverse(root, chance_node_fn=get_info)
    return ret
