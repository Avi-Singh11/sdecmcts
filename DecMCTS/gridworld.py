import random
from decmcts import MCTS

class GridWorldState:
    def __init__(self, agent_pos, goal_pos, grid_size, reward=0, is_terminal=None):
        self.agent_pos = agent_pos
        self.goal_pos = goal_pos
        self.grid_size = grid_size
        self.reward = reward
        self.is_terminal = (agent_pos == goal_pos) if is_terminal is None else is_terminal

    def get_legal_actions(self):
        actions = ["stay"]
        if self.agent_pos[0] > 0:
            actions.append("up")
        if self.agent_pos[0] < self.grid_size[0] - 1:
            actions.append("down")
        if self.agent_pos[1] > 0:
            actions.append("left")
        if self.agent_pos[1] < self.grid_size[1] - 1:
            actions.append("right")
        return actions

    # TODO: EMBED UNCERTAINTIES INTO ACTIONS: STAY WITH PROB 0.8, MOVE WITH PROB 0.1, ETC.
    def take_action(self, action):
        if action == "up":
            new_agent_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
        elif action == "down":
            new_agent_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
        elif action == "left":
            new_agent_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
        elif action == "right":
            new_agent_pos = (self.agent_pos[0], self.agent_pos[1] + 1)
        elif action == "stay":
            new_agent_pos = self.agent_pos
        
        is_term = (new_agent_pos == self.goal_pos)
        step_reward = 100 if is_term else -1
        
        return GridWorldState(
            agent_pos=new_agent_pos, 
            goal_pos=self.goal_pos, 
            grid_size=self.grid_size,
            reward=step_reward,
            is_terminal=is_term
        )

    def is_terminal_state(self):
        return self.is_terminal
    
    
        