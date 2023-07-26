import numpy as np
from tqdm import tqdm

from TreeNode import *


def possible_action_value(edge: Edge, select_mode = "best"):
    if len(edge.simulation_rewards) == 0:
        return 0
    if select_mode == "best":
        return max(edge.simulation_rewards)
    elif select_mode == "average":
        return sum(edge.simulation_rewards) / len(edge.simulation_rewards)
    else:
        raise Exception(f"Unknown tree search model{select_mode}")


def mcts_procedure(agent, env, rollout_weight, root, terminate_condition, select_mode = "best"):
    """
        env: 与模型交互的环境，
        agent: 动作主体,控制
    """
    if root is not None:
        assert root.state == env.state
    else:
        root = Node(None, env.state, agent.action_space.copy())
    # 确保rollout在[1, agent.rollouts]之间,rollout表示每次mcts执行的次数
    rollouts = np.clip(int(agent.rollouts * rollout_weight), 1, agent.rollouts)
    cur_node = root
    for i in tqdm(range(rollouts)):
        rewards_for_backpropagation = []
        # 选择阶段，找到一个叶子结点或一个结点选择了一个没有扩展的action
        cur_node, edge = selection(cur_node, env)
        # 扩展阶段，根据transformer选择
        temp_node = expansion(cur_node, agent, env, edge, rewards_for_backpropagation)
        # 评估阶段，即模拟游戏直到结束，这里是直接生成整段代码
        value = evaluation(temp_node, agent, env)
        # 根据评估得到的value,更新所有上层结点的访问次数和value
        backpropagation(temp_node, value, rewards_for_backpropagation, agent)

    return max(root.children, key=lambda _edge: possible_action_value(_edge, select_mode=select_mode)).action, root


def backpropagation(node: Node, value, rewards_for_backpropagation, agent):
    edge = node.parent
    while node:
        # visits更新在结点上
        node.visits += 1
        if len(rewards_for_backpropagation) != 0:
            # 根据贝尔曼方程qπ(s,a)=Eπ(Gt|St=s,At=a)=Eπ(Rt+1+  γRt+2 + γ^2Rt+3 + ...|St=s,At=a)
            # = vπ(s)=Eπ(Rt+1 + γvπ(St+1) |St=s)
            # Q(s, a) = R_t + 衰减 * Q_t+1
            # rewards_for_backpropagation保存了之前每个状态的reward, value在评估阶段给出
            value = rewards_for_backpropagation.pop() + agent.gamma * value
        # Q(s, a)更新在边上
        edge.simulation_rewards.append(value)
        node = edge.parent
    assert len(rewards_for_backpropagation) == 0


def evaluation(node, agent, env):
    value = None
    if not node.is_terminal:
        # 直接使用Transformer估计价值
        if agent.default_policy.use_value:
            # 模拟至结束
            temp_full_state = agent.default_policy.get_short_sequence(node.state)
            value = agent.default_policy.get_value(temp_full_state)
        # 使用Transformer生成完整代码后，用完整程序的即时奖励当做价值
        else:
            temp_full_state = agent.default_policy.get_predict_sequence(node.state)
            value = env.get_reward(temp_full_state)
            node.info['complete_program'] = temp_full_state
    return value


def expansion(node: Node, agent, env, edge: Edge, rewards_for_backpropagation: list):
    new_state, reward, terminal = env.transition(node.state, edge.action)
    # 经过这条边的历史奖励都要记录，方便计算Q(s, a)
    edge.simulation_rewards.append(reward)
    rewards_for_backpropagation.append(reward)

    # 利用Transformer预测最有可能的k个action
    top_k_action, top_k_probability = agent.default_policy.get_top_k_predict(new_state)

    # 建立结点及结点所连接的边
    new_node = Node(parent=edge, state=new_state)
    children = [Edge(parent=new_node, action=action, action_probability=probability)
                for action, probability in zip(top_k_action, top_k_probability)]
    new_node.children = children

    edge.children.append(new_node)

    return new_node


def selection(node, agent):
    cur = node
    edge = None
    while node.children is None:
        if node.is_terminal:
            break
        else:
            # 采取action,到达下一个node
            edge = agent.tree_policy(node)
        # 如果选择的edge(实际上代表了action)还没有结点，就停止选择，进入扩展阶段
        if edge.children is None:
            break
    return cur, edge
