import math
import random

# DATA STRUCTURE DEF
class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.cum_reward = 0

    def add_child(self, child_state, action):
        child = MCTSNode(child_state, parent=self, action=action)
        self.children.append(child)
        return child
        
    def compute_q(self):
        if self.visits == 0:
            return 0
        else:
            return self.cum_reward / self.visits

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_actions())
    

class MCTS:
    def __init__(self, init_state, exploration_const=math.sqrt(2), num_iter=1000, discount_factor=0.95, depth=100):
        self.root = MCTSNode(init_state)
        self.exploration_const = exploration_const
        self.num_iter = num_iter
        self.discount_factor = discount_factor
        self.depth = depth
    
    
    # SELECTION
    def selection(self):
        current_node = self.root

        while current_node.is_fully_expanded() and not current_node.state.is_terminal_state():
            best_score = -float('inf')
            best_child = None

            for child in current_node.children:
                curr_score = self.ucb_score(child)
                if curr_score > best_score:
                    best_score = curr_score
                    best_child = child
            
            current_node = best_child

        return current_node

    # EXPANSION
    def expansion(self, node):
        if node.state.is_terminal_state():
            return node
            
        legal_actions = node.state.get_legal_actions()
        tried_actions = [child.action for child in node.children]
        untried_actions = [action for action in legal_actions if action not in tried_actions]

        if untried_actions:
            action = random.choice(untried_actions)
            next_state = node.state.take_action(action)
            return node.add_child(next_state, action)
        else:
            return None

        return node

    # ROLLOUT
    def rollout(self, node):
        current_state = node.state
        total_rollout_reward = 0
        depth = 0
        
        while not current_state.is_terminal_state() and depth < self.depth:
            legal_actions = current_state.get_legal_actions()
            if not legal_actions:
                break
                
            action = random.choice(legal_actions)
            current_state = current_state.take_action(action)
            total_rollout_reward += current_state.reward * (self.discount_factor ** depth)
            depth += 1
            
        return total_rollout_reward

    # BACKPROPAGATION
    def backprop(self, node, rollout_reward):
        future_return = rollout_reward
        while node is not None:
            node.visits += 1
            total_return = node.state.reward + self.discount_factor * future_return
            node.cum_reward += total_return
            
            future_return = total_return
            node = node.parent
        
    # UCB SCORE
    def ucb_score(self, node):
        return node.compute_q() + self.exploration_const * math.sqrt(math.log(self.root.visits) / node.visits)

    

