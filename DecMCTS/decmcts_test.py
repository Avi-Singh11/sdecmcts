from decmcts import DecMCTS, DecMCTSNode
import copy

# A simple 1D state where players must coordinate.
# Player 1 and Player 2 both start at pos 2, max steps 3.
# To maximize global objective, they should move away from each other.

class Simple1DState:
    def __init__(self, pos, steps=0, max_steps=3):
        self.pos = pos
        self.steps = steps
        self.max_steps = max_steps
        
    def is_terminal_state(self):
        return self.steps >= self.max_steps
        
    def get_legal_actions(self):
        return ["left", "right"]
        
    def take_action(self, action):
        p = self.pos - 1 if action == "left" else self.pos + 1
        return Simple1DState(p, self.steps + 1, self.max_steps)

def simple_global_objective(joint_sequences):
    # Reward is dist in between both agents
    end_pos = {1: 2, 2: 2} # Both start at 2
    for r, seq in joint_sequences.items():
        for act in seq:
            if act == "left": end_pos[r] -= 1
            elif act == "right": end_pos[r] += 1
    return abs(end_pos[1] - end_pos[2])

agent1 = DecMCTS(robot_id=1, robot_ids=[1, 2], global_objective_fn=simple_global_objective, depth=3)
agent2 = DecMCTS(robot_id=2, robot_ids=[1, 2], global_objective_fn=simple_global_objective, depth=3)

agent1.root = DecMCTSNode(action=None, state=Simple1DState(pos=2, max_steps=3))
agent2.root = DecMCTSNode(action=None, state=Simple1DState(pos=2, max_steps=3))

epochs = 20
print("Starting DecMCTS Test")
for ep in range(epochs):
    # Plan
    agent1.grow_tree(num_iterations=20)
    agent2.grow_tree(num_iterations=20)
    
    agent1.update_sample_space()
    agent2.update_sample_space()
    
    agent1.update_distribution()
    agent2.update_distribution()
    
    # Communicate
    # TODO: Implement restrictive communciation; add handling to @decmcts.py
    agent1.q_dists[2] = copy.deepcopy(agent2.local_dist)
    agent2.q_dists[1] = copy.deepcopy(agent1.local_dist)
    
print("Agent 1 Distribution:", agent1.local_dist)
print("Agent 1 Choice:", agent1.get_best_action_sequence())

print("Agent 2 Distribution:", agent2.local_dist)
print("Agent 2 Choice:", agent2.get_best_action_sequence())

print("Final expected distance (Objective):", simple_global_objective({
    1: agent1.get_best_action_sequence(),
    2: agent2.get_best_action_sequence()
}))
