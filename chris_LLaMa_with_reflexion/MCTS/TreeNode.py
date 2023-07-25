class Node:
    def __init__(self,
                 parent,
                 state,
                 possible_actions=[],
                 is_terminal=False,
                 id=None,
                 children=None,
                 action_probability=None,
                 ):
        self.id = id
        self.parent = parent
        self.state = state
        self.is_terminal = is_terminal
        if self.parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

        self.children = children
        self.possible_action = possible_actions
        self.action_probability = action_probability
        self.simulation_rewards = []

        self.explored_children = 0
        # this decision node should be visited at least once, otherwise p-uct makes no sense for this node
        self.visits = 1
        # used to save any information of the state
        # we use this for saving complete programs generated from it
        self.info = {}
