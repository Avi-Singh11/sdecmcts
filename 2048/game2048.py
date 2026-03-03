import random

class Game2048State:
    def __init__(self, board=None, reward=0, is_terminal=False):
        if board is None:
            self.board = [[0]*4 for _ in range(4)]
            self.spawn_tile()
            self.spawn_tile()
        else:
            self.board = [row[:] for row in board]
            
        self.reward = reward
        self.is_terminal = is_terminal

    def get_empty_cells(self):
        empty = []
        for r in range(4):
            for c in range(4):
                if self.board[r][c] == 0:
                    empty.append((r, c))
        return empty

    def spawn_tile(self):
        empty = self.get_empty_cells()
        if empty:
            r, c = random.choice(empty)
            self.board[r][c] = 2 if random.random() < 0.9 else 4

    def compress(self, board):
        changed = False
        new_board = [[0]*4 for _ in range(4)]
        for r in range(4):
            pos = 0
            for c in range(4):
                if board[r][c] != 0:
                    new_board[r][pos] = board[r][c]
                    if c != pos:
                        changed = True
                    pos += 1
        return new_board, changed

    def merge(self, board):
        changed = False
        reward = 0
        for r in range(4):
            for c in range(3):
                if board[r][c] == board[r][c + 1] and board[r][c] != 0:
                    board[r][c] *= 2
                    board[r][c + 1] = 0
                    reward += board[r][c]
                    changed = True
        return board, changed, reward

    def reverse(self, board):
        new_board = []
        for r in range(4):
            new_board.append(list(reversed(board[r])))
        return new_board

    def transpose(self, board):
        new_board = [[board[c][r] for c in range(4)] for r in range(4)]
        return new_board

    def try_move(self, action):
        """Simulates a move without spawning a new tile. Returns (new_board, changed, reward)"""
        temp_board = [row[:] for row in self.board]
        changed = False
        reward = 0
        
        if action == 'up':
            temp_board = self.transpose(temp_board)
            temp_board, changed1 = self.compress(temp_board)
            temp_board, changed2, reward = self.merge(temp_board)
            temp_board, _ = self.compress(temp_board)
            temp_board = self.transpose(temp_board)
            changed = changed1 or changed2
            
        elif action == 'down':
            temp_board = self.transpose(temp_board)
            temp_board = self.reverse(temp_board)
            temp_board, changed1 = self.compress(temp_board)
            temp_board, changed2, reward = self.merge(temp_board)
            temp_board, _ = self.compress(temp_board)
            temp_board = self.reverse(temp_board)
            temp_board = self.transpose(temp_board)
            changed = changed1 or changed2
            
        elif action == 'left':
            temp_board, changed1 = self.compress(temp_board)
            temp_board, changed2, reward = self.merge(temp_board)
            temp_board, _ = self.compress(temp_board)
            changed = changed1 or changed2
            
        elif action == 'right':
            temp_board = self.reverse(temp_board)
            temp_board, changed1 = self.compress(temp_board)
            temp_board, changed2, reward = self.merge(temp_board)
            temp_board, _ = self.compress(temp_board)
            temp_board = self.reverse(temp_board)
            changed = changed1 or changed2
            
        return temp_board, changed, reward

    def get_legal_actions(self):
        actions = []
        for action in ['up', 'down', 'left', 'right']:
            _, changed, _ = self.try_move(action)
            if changed:
                actions.append(action)
        return actions

    def take_action(self, action):
        new_board, changed, step_reward = self.try_move(action)
        
        # If the action actually did something, spawn a new tile (stochastic environment step)
        if changed:
            new_state = Game2048State(board=new_board, reward=step_reward)
            new_state.spawn_tile()
            new_state.is_terminal = new_state.check_terminal()
            return new_state
        return Game2048State(board=self.board, reward=0, is_terminal=self.is_terminal)

    def check_terminal(self):
        if len(self.get_empty_cells()) > 0:
            return False
        
        # Check horizontal merges
        for r in range(4):
            for c in range(3):
                if self.board[r][c] == self.board[r][c+1]:
                    return False
        # Check vertical merges
        for r in range(3):
            for c in range(4):
                if self.board[r][c] == self.board[r+1][c]:
                    return False
        return True

    def is_terminal_state(self):
        return self.is_terminal
