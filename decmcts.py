import math
import random
import copy
import numpy as np

class DecMCTSNode:
    def __init__(self, action, parent=None, state=None):
        self.action = action
        self.parent = parent
        self.children = []
        self.state = state  # The environment state after taking the sequence of actions
        
        # D-UCT variables
        self.last_update_t = 0
        self.visits = 0
        self.discounted_visits = 0.0
        self.discounted_reward_sum = 0.0
        
        # Track sequence of actions leading to this node
        self.action_sequence = []
        if parent is not None:
            self.action_sequence = parent.action_sequence + [action]
            
        self.untried_actions = None
        if state is not None and not state.is_terminal_state():
            self.untried_actions = state.get_legal_actions()
        else:
            self.untried_actions = []

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self):
        return self.state is None or self.state.is_terminal_state()

    def add_child(self, action, next_state):
        child = DecMCTSNode(action, parent=self, state=next_state)
        self.children.append(child)
        if action in self.untried_actions:
            self.untried_actions.remove(action)
        return child
        
    def update(self, reward, t, gamma):
        if self.last_update_t > 0:
            scale = gamma ** (t - self.last_update_t)
            self.discounted_visits *= scale
            self.discounted_reward_sum *= scale
            
        self.discounted_visits += 1.0
        self.discounted_reward_sum += reward
        self.visits += 1
        self.last_update_t = t
        
    def get_q(self):
        if self.discounted_visits == 0:
            return 0.0
        return self.discounted_reward_sum / self.discounted_visits
        
    def get_expected_visits_at(self, t, gamma):
        if self.last_update_t == 0:
            return 0.0
        scale = gamma ** (t - self.last_update_t)
        return self.discounted_visits * scale


def d_ucb_score(parent, child, t, gamma, Cp):
    child_v = child.get_expected_visits_at(t, gamma)
    if child_v == 0:
        return float('inf')
    
    parent_v = parent.get_expected_visits_at(t, gamma)
    if parent_v == 0:
        return float('inf')
        
    q = child.get_q()
    return q + 2 * Cp * math.sqrt(max(0.0, math.log(parent_v)) / child_v)


class DecMCTS:
    def __init__(self, robot_id, robot_ids, global_objective_fn, 
                 null_sequences=None,
                 gamma=0.9, Cp=1.0, depth=10, 
                 num_samples_dist_opt=50,
                 beta_initial=1.0, alpha=0.01):
        self.robot_id = robot_id
        self.robot_ids = robot_ids
        self.global_objective_fn = global_objective_fn
        self.null_sequences = null_sequences or {r: [] for r in robot_ids}
        
        self.gamma = gamma
        self.Cp = Cp
        self.depth = depth
        
        self.num_samples_dist_opt = num_samples_dist_opt
        self.initial_beta = beta_initial
        self.beta = beta_initial
        self.alpha = alpha
        
        self.t = 0
        self.root = None # Needs arbitrary start state initialization
        
        # Dictionary to store distributions Q of other robots
        # q_dists[r] = {action_seq_tuple: prob}
        self.q_dists = {r: {} for r in robot_ids if r != robot_id}
        
        # Local distribution
        self.local_X = [] # List of top action sequence tuples
        self.local_dist = {} # dict mapping action_seq_tuple -> prob
        
        # Track best complete rollout sequences for sample space extraction
        self.best_rollouts = []

    def sample_seq(self, dist_dict):
        if not dist_dict:
            return []
        seqs = list(dist_dict.keys())
        probs = list(dist_dict.values())
        prob_sum = sum(probs)
        if prob_sum == 0 or not np.isfinite(prob_sum):
            probs = [1.0/len(probs)] * len(probs)
        else:
            probs = [p/prob_sum for p in probs]
        idx = np.random.choice(len(seqs), p=probs)
        return list(seqs[idx])

    def local_utility(self, joint_sequences):
        g_val = self.global_objective_fn(joint_sequences)
        
        # Determine global objective if this robot took null actions
        null_joint = copy.deepcopy(joint_sequences)
        null_joint[self.robot_id] = self.null_sequences[self.robot_id]
        g_null = self.global_objective_fn(null_joint)
        
        return g_val - g_null

    def grow_tree(self, num_iterations=1):
        for _ in range(num_iterations):
            self.t += 1
            
            # Selection
            node = self.root
            while not node.is_terminal() and node.is_fully_expanded():
                node = max(node.children, key=lambda c: d_ucb_score(node, c, self.t, self.gamma, self.Cp))
                
            # Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                action = random.choice(node.untried_actions)
                next_state = node.state.take_action(action)
                node = node.add_child(action, next_state)
                
            # Rollout
            curr_state = node.state
            rollout_actions = []
            depth = 0
            while not curr_state.is_terminal_state() and len(node.action_sequence) + depth < self.depth:
                legal = curr_state.get_legal_actions()
                if not legal:
                    break
                a = random.choice(legal)
                rollout_actions.append(a)
                curr_state = curr_state.take_action(a)
                depth += 1
                
            full_seq = node.action_sequence + rollout_actions
            
            # Construct joint action sequence
            joint = {self.robot_id: full_seq}
            for r in self.robot_ids:
                if r != self.robot_id:
                    joint[r] = self.sample_seq(self.q_dists[r])
                    
            # Evaluate
            reward = self.local_utility(joint)
            
            # Save to best rollouts for sample space derivation
            self.add_to_best_rollouts(full_seq, reward)
            
            # Backpropagation
            curr = node
            while curr is not None:
                curr.update(reward, self.t, self.gamma)
                curr = curr.parent

    def add_to_best_rollouts(self, seq, score, max_len=10):
        tup = tuple(seq)
        for i, (s, sc) in enumerate(self.best_rollouts):
            if s == tup:
                if score > sc:
                    self.best_rollouts[i] = (s, score)
                    self.best_rollouts.sort(key=lambda x: x[1], reverse=True)
                return
                
        self.best_rollouts.append((tup, score))
        self.best_rollouts.sort(key=lambda x: x[1], reverse=True)
        if len(self.best_rollouts) > max_len:
            self.best_rollouts.pop()

    def update_sample_space(self, num_sequences=5):
        self.local_X = [tup for tup, score in self.best_rollouts[:num_sequences]]
        
        # Reset distribution to uniform over the new sample space
        if len(self.local_X) > 0:
            prob = 1.0 / len(self.local_X)
            self.local_dist = {s: prob for s in self.local_X}
        else:
            self.local_dist = {}
        
        self.beta = self.initial_beta

    def update_distribution(self):
        if not self.local_X:
            return
            
        # Estimate expected reward over all joint sampling
        expected_f_r = 0.0
        for _ in range(self.num_samples_dist_opt):
            joint = {self.robot_id: self.sample_seq(self.local_dist)}
            for r in self.robot_ids:
                if r != self.robot_id:
                    joint[r] = self.sample_seq(self.q_dists[r])
            expected_f_r += self.local_utility(joint)
        expected_f_r /= max(1, self.num_samples_dist_opt)
        
        new_dist = {}
        # Estimate expected reward playing specific local seq
        for xr_tup in self.local_X:
            xr = list(xr_tup)
            
            expected_f_r_given_xr = 0.0
            for _ in range(self.num_samples_dist_opt):
                joint = {self.robot_id: xr}
                for r in self.robot_ids:
                    if r != self.robot_id:
                        joint[r] = self.sample_seq(self.q_dists[r])
                expected_f_r_given_xr += self.local_utility(joint)
                
            expected_f_r_given_xr /= max(1, self.num_samples_dist_opt)
            
            q_val = self.local_dist.get(xr_tup, 1.0 / len(self.local_X))
            
            # Compute Entropy
            entropy = 0.0
            for v in self.local_dist.values():
                if v > 0:
                    entropy -= v * math.log(v)
                    
            ln_q = math.log(q_val) if q_val > 0 else -100.0
                
            # Gradient descent step
            update_delta = self.alpha * q_val * (
                (expected_f_r - expected_f_r_given_xr) / self.beta + entropy + ln_q
            )
            
            new_val = q_val - update_delta
            new_dist[xr_tup] = max(1e-9, new_val) # Prevent exactly 0
            
        # Normalize
        total = sum(new_dist.values())
        if total > 0:
            for k in new_dist:
                new_dist[k] /= total
        else:
            for k in new_dist:
                new_dist[k] = 1.0 / len(new_dist)
                
        self.local_dist = new_dist
        
        # Cool beta
        self.beta *= 0.99

    def get_best_action_sequence(self):
        if self.local_dist:
            best_seq = max(self.local_dist.items(), key=lambda x: x[1])[0]
            return list(best_seq)
        elif self.best_rollouts:
            return list(self.best_rollouts[0][0])
        return []


