import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
from gridworld import GridWorldState
from mcts import MCTS

def get_best_action(mcts):
    best_action = None
    best_visits = -1
    for child in mcts.root.children:
        if child.visits > best_visits:
            best_visits = child.visits
            best_action = child.action
    return best_action

def get_action_stats(mcts):
    stats = {}
    for child in mcts.root.children:
        stats[child.action] = {
            'visits': child.visits,
            'q': child.compute_q()
        }
    return stats

def main():
    grid_size = (5, 5)
    goal_pos = (4, 4)
    start_pos = (0, 0)
    
    state = GridWorldState(agent_pos=start_pos, goal_pos=goal_pos, grid_size=grid_size)
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    path = [start_pos]
    max_steps = 1000
    step = 0
    
    print("Starting simulation... The matplotlib window should pop up!")
    
    while not state.is_terminal_state() and step < max_steps:
        # Run MCTS Iterations
        # Giving it 500 iterations for decent convergence on a 5x5 grid
        mcts = MCTS(init_state=state, num_iter=500, depth=50) 
        for _ in range(mcts.num_iter):
            leaf = mcts.selection()
            child = mcts.expansion(leaf)
            if child is None:
                child = leaf
            reward = mcts.rollout(child)
            mcts.backprop(child, reward)
            
        action = get_best_action(mcts)
        stats = get_action_stats(mcts)
        
        # Execute the chosen action in the environment
        state = state.take_action(action)
        path.append(state.agent_pos)
        step += 1
        
        # --- Visualization Updates ---
        ax.clear()
        
        # Setup Grid
        ax.set_xlim(-0.5, grid_size[1] - 0.5)
        ax.set_ylim(grid_size[0] - 0.5, -0.5) # Invert Y so (0,0) is top-left
        ax.set_xticks(np.arange(-0.5, grid_size[1], 1))
        ax.set_yticks(np.arange(-0.5, grid_size[0], 1))
        ax.grid(True, color='black', linewidth=1)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        # Color Goal
        goal_rect = plt.Rectangle((goal_pos[1] - 0.5, goal_pos[0] - 0.5), 1, 1, facecolor='lightgreen', edgecolor='none')
        ax.add_patch(goal_rect)
        ax.text(goal_pos[1], goal_pos[0], 'Goal', ha='center', va='center', weight='bold')

        # Draw Start
        ax.text(start_pos[1], start_pos[0], 'Start', ha='center', va='center', color='gray')
        
        # Draw Trajectory
        if len(path) > 1:
            y_coords, x_coords = zip(*path)
            ax.plot(x_coords, y_coords, color='blue', alpha=0.4, linewidth=3, marker='o', markersize=4)
            
        # Draw Current Agent Position
        agent_pos = state.agent_pos
        agent_circle = plt.Circle((agent_pos[1], agent_pos[0]), 0.3, color='red')
        ax.add_patch(agent_circle)
        
        # Format Statistics string
        stats_text = f"Step: {step}\nIntended Action: '{action}'\nCurrent Pos: {agent_pos}\n\n"
        stats_text += "MCTS Evaluation:\n"
        for act, d in stats.items():
            stats_text += f" - {act}: V={d['visits']}, Q={d['q']:.1f}\n"
            
        # Note if the agent 'slipped'
        if len(path) > 1:
            prev_pos = path[-2]
            if action != "stay" and prev_pos == agent_pos:
                stats_text += "\n[!] Agent Slipped!"
        
        ax.text(1.05, 0.5, stats_text, transform=ax.transAxes, verticalalignment='center',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))
        
        ax.set_title("MCTS GridWorld Agent")
        plt.tight_layout()
        plt.pause(0.5)
        
    print(f"Simulation ended at step {step}. Terminal state reached: {state.is_terminal_state()}")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
