import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game2048 import Game2048State
from decmcts import MCTS

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

def print_board(state, step, total_score, action, stats):
    print("\n" + "="*40)
    print(f"Step: {step} | Score: {total_score} | Move: '{action.upper() if action else 'START'}'")
    print("-" * 25)
    for row in state.board:
        print(" ".join([f"{val:4}" if val > 0 else "   ." for val in row]))
    print("-" * 25)
    
    if stats:
        print("MCTS Analysis:")
        for act, d in stats.items():
            print(f" - {act:5}: V={d['visits']:<4}, Q={d['q']:.1f}")
    print("="*40)

def main():
    state = Game2048State()
    
    max_steps = 2000
    step = 0
    total_score = 0
    
    print("Starting MCTS 2048 solver")
    print_board(state, step, total_score, None, None)
    
    while not state.is_terminal_state() and step < max_steps:
        mcts = MCTS(init_state=state, num_iter=400, depth=30, discount_factor=0.9)
        for _ in range(mcts.num_iter):
            leaf = mcts.selection()
            child = mcts.expansion(leaf)
            if child is None:
                child = leaf
            reward = mcts.rollout(child)
            mcts.backprop(child, reward)
            
        action = get_best_action(mcts)
        stats = get_action_stats(mcts)
        
        # Take Action
        new_state = state.take_action(action)
        step += 1
        total_score += new_state.reward
        
        print_board(new_state, step, total_score, action, stats)
        state = new_state
        
    print(f"\nGAME OVER at step {step}. Total Score: {total_score}")

if __name__ == "__main__":
    main()
