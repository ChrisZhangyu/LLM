import numpy as np
from tqdm import tqdm

from TreeNode import *


def mcts_procedure(env, exe, gen, agent, rollout_weight, root, terminate_condition):
    """
        env: 环境变量，
        exe: 执行器
        gen: 生成器
    """
    if root is not None:
        assert root.state == env.state
    else:
        root = Node(None, env.state, agent.action_space.copy())
    # 确保rollout在[1, agent.rollouts]之间,rollout表示每次mcts执行的次数
    rollouts = np.clip(int(agent.rollouts * rollout_weight), 1, agent.rollouts)
    cur_node = root
    for i in tqdm(range(rollouts)):
        # 选择阶段，知道找到一个叶子结点
        leaf_node,  action = selection(cur_node, env, agent)
        # 扩展阶段，根据transformer选择
        temp_node = expansion(leaf_node, agent, env, action)
        evaluation(temp_node)
        backpropagation()


def backpropagation():
    pass


def evaluation(node, agent, env):
    if not node.is_terminal:
        if agent.default_policy.use_value:
            # 模拟至结束
            temp_full_state = agent.default_policy.get_short_sequence(node.state)
            value = agent.default_policy.get_value(temp_full_state)
        else:
            temp_full_state = agent.default_policy.get_predict_sequence(node.state)
            value = agent.default_policy.get_reward(temp_full_state)
            node.info['complete_program'] = temp_full_state
    return value

def expansion(node: Node, agent, env, action):
    new_state, reward, terminal = env.transition(node.state, action)
    top_k_action, top_k_probability = agent.default_policy.get_top_k_predict(new_state)
    new_node = Node(parent=node, state=new_state, possible_actions=top_k_action, action_probability=top_k_probability)
    node.children.append(new_node)

    return new_node


def selection(node, env, agent):
    rewards = []
    action = None
    while node.children is None:
        if node.is_terminal:
            break
        else:
            # 采取action,到达下一个node
            node, action = agent.tree_policy(node)
    return node, action
