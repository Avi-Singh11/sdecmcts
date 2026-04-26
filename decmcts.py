"""
Dec-MCTS implementation:
Builds on the single-agent MCTS base in Scripts/mcts.py.

Your state object must implement:
    state.get_legal_actions()  -> list of actions
    state.take_action(action)  -> new state
    state.is_terminal_state()  -> bool
    state.reward               -> float  (only needed if using single-agent MCTS)
"""

import math
import random
import copy

from mcts import MCTSNode, MCTS  # single-agent base


# PART 1 — DEC-MCTS NODE
class DecMCTSNode:
    """
    Node in robot r's local search tree T^r.
    Each edge = one action by robot r.
    A path root→node = action sequence x^r.

    Extends the single-agent node with:
      - action_sequence: cached path from root (needed by rollout evaluator)
      - untried_actions: list maintained for expansion
      - D-UCT discounted statistics (Section 4.3.1, Equations 4-7)

    D-UCT recurrence (applied once per MCTS iteration t):
        disc_visits[t] = gamma * disc_visits[t-1]  +  1{this node visited}
        disc_reward[t] = gamma * disc_reward[t-1]  +  F_t * 1{this node visited}
    """

    def __init__(self, state, parent=None, action=None):
        self.state    = state
        self.parent   = parent
        self.action   = action
        self.children = []

        # Standard (undiscounted) statistics — used for final action extraction
        self.visits     = 0
        self.cum_reward = 0.0

        # D-UCT discounted statistics
        self.disc_visits = 0.0
        self.disc_reward = 0.0

        # Cache the full action sequence root→this node
        self.action_sequence = (
            parent.action_sequence + [action] if parent is not None else []
        )


        # Untried actions: copy so we never alias state internals
        self.untried_actions = (
            list(state.get_legal_actions())
            if state is not None and not state.is_terminal_state()
            else []
        )

    # Tree structure

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self):
        return self.state is None or self.state.is_terminal_state()

    def add_child(self, action, next_state):
        child = DecMCTSNode(next_state, parent=self, action=action)
        self.children.append(child)
        if action in self.untried_actions:
            self.untried_actions.remove(action)
        return child

    # Q-value accessors

    def disc_q(self):
        """Discounted empirical average F̄ (Equation 6)."""
        return self.disc_reward / self.disc_visits if self.disc_visits > 0 else 0.0

    def q(self):
        """Undiscounted Q — used for final greedy extraction (more stable)."""
        return self.cum_reward / self.visits if self.visits > 0 else 0.0

    # D-UCT upper confidence bound

    def d_ucb(self, parent, t, gamma, Cp):
        """
        D-UCT score (Equation 3 / 7):

            U = F̄_j(gamma) + 2*Cp * sqrt( log(disc_visits_parent) / disc_visits_j )

        Returns inf for unvisited nodes or when parent has too few visits
        (log(x) < 0 for x < 1).
        """
        if self.disc_visits == 0:
            return float("inf")
        if parent.disc_visits <= 1.0:
            return float("inf")
        exploration = 2.0 * Cp * math.sqrt(math.log(parent.disc_visits) / self.disc_visits)
        return self.disc_q() + exploration

    def update_discounted(self, reward, visited, gamma):
        """
        Incremental D-UCT update — call once per MCTS iteration.

        `visited` = True if this node was on the selection/expansion path.
        Applies gamma decay to existing statistics, then folds in new sample.
        """
        self.disc_visits = gamma * self.disc_visits + (1.0 if visited else 0.0)
        self.disc_reward = gamma * self.disc_reward + (reward if visited else 0.0)


# PART 2 — DEC-MCTS PLANNER
class DecMCTS:
    """
    Dec-MCTS planner for a single robot r.

    The outer loop cycles three phases per iteration n:
        1. GROW TREE    — D-UCT MCTS using other robots' sampled sequences
        2. UPDATE DIST  — gradient descent on product distribution q^r_n
        3. COMMUNICATE  — broadcast/receive (X̂^r_n, q^r_n)  [handled externally]

    Parameters
    ----------
    robot_id         : hashable identifier for this robot
    robot_ids        : list of all robot identifiers (including this one)
    init_state       : initial state object
    local_utility_fn : f^r(joint_sequences) -> float
                       joint_sequences = {robot_id: [action, ...], ...}
                       Should implement Equation 1:
                           g(x^r u x^(r)) - g(x^r_∅ u x^(r))
    gamma            : D-UCT discount factor ∈ (0.5, 1)
    Cp               : exploration constant  (paper: Cp > 1/sqrt(8) ≈ 0.354)
    rollout_depth    : max depth for random rollout simulation
    tau              : MCTS iterations per outer loop iteration
    num_seq          : size of sample space X̂^r_n  (paper uses 10)
    num_samples      : MC samples for expectation estimates in dist update
    beta_init        : initial temperature β for distribution optimization
    beta_decay       : multiplicative cooling per outer iteration
    alpha            : gradient step size (paper: alpha = 0.01)
    """

    def __init__(
        self,
        robot_id,
        robot_ids,
        init_state,
        local_utility_fn,
        gamma         = 0.9,
        Cp            = 1.0 / math.sqrt(2),
        rollout_depth = 20,
        tau           = 10,
        num_seq       = 10,
        num_samples   = 30,
        beta_init     = 1.0,
        beta_decay    = 0.99,
        alpha         = 0.01,
    ):
        self.robot_id         = robot_id
        self.robot_ids        = robot_ids
        self.local_utility_fn = local_utility_fn
        self.gamma            = gamma
        self.Cp               = Cp
        self.rollout_depth    = rollout_depth
        self.tau              = tau
        self.num_seq          = num_seq
        self.num_samples      = num_samples
        self.beta             = beta_init
        self.beta_decay       = beta_decay
        self.alpha            = alpha

        # Search tree T^r
        self.root = DecMCTSNode(init_state)

        # Global iteration counter t
        self.t = 0

        # Other robots' received distributions: {robot_id: {seq_tuple: prob}}
        # Empty until first message received — robot uses default (empty) policy
        self.received_dists = {r: {} for r in robot_ids if r != robot_id}

        # Robot r's own distribution q^r_n and sample space X̂^r_n
        self.X_hat = []   # List[tuple] -> sample space
        self.q     = {}   # Dict[tuple, float] -> distribution over X_hat

    # PUBLIC API
    def iterate(self, n_outer=1):
        """
        Run n_outer iterations of the outer loop.

        Each outer iteration:
          (a) updates sample space X̂^r_n
          (b) runs tau MCTS rollouts  (Algorithm 2)
          (c) updates distribution q^r_n  (Algorithm 3)
          (d) cools beta
        """
        for _ in range(n_outer):
            self._update_sample_space()          
            for _ in range(self.tau):            
                self._grow_tree_once()
            self._update_distribution()          
            self.beta *= self.beta_decay         

    def receive(self, robot_id, X_hat, q):
        """
        Receive distribution from another robot.
        Accepts either a list of probs aligned with X_hat, or a dict.
        """
        self.received_dists[robot_id] = dict(zip(
            [tuple(x) for x in X_hat],
            q if isinstance(q, list) else list(q.values())
        ))

    def receive_dist_dict(self, robot_id, dist_dict):
        """Convenience: receive a {seq_tuple: prob} dict directly."""
        self.received_dists[robot_id] = copy.copy(dist_dict)

    def get_distribution(self):
        """
        Return (X̂^r_n, q^r_n) for broadcasting.
        """
        return list(self.X_hat), copy.copy(self.q)

    def best_action_sequence(self):
        """
        Return argmax_{x^r ∈ X̂^r_n} q^r_n(x^r).
        Falls back to greedy Q-value tree descent if distribution is empty.
        """
        if self.q:
            return list(max(self.q, key=self.q.get))
        return self._greedy_tree_sequence()

    def best_action(self):
        """Return just the first action of the best sequence."""
        seq = self.best_action_sequence()
        return seq[0] if seq else None

    # ALGORITHM 2: GROW TREE (one iteration)
    def _grow_tree_once(self):
        """
        One iteration of growing the tree.
        Uses D-UCT for selection instead of standard UCB.
        Evaluation uses local_utility_fn over the joint sequence.
        """
        self.t += 1

        # Selection: D-UCT descent
        node = self.root
        path = [node]
        while not node.is_terminal() and node.is_fully_expanded() and node.children:
            node = max(
                node.children,
                key=lambda c: c.d_ucb(node, self.t, self.gamma, self.Cp)
            )
            path.append(node)

        # Expansion: add one untried child
        if not node.is_terminal() and not node.is_fully_expanded():
            action     = random.choice(node.untried_actions)
            next_state = node.state.take_action(action)
            node       = node.add_child(action, next_state)
            path.append(node)

        # Simulation: random rollout from node 
        rollout_actions = self._rollout(node)
        x_r = node.action_sequence + rollout_actions

        # Sample other robots sequences from their received distributions
        x_others = self._sample_others()

        # Evaluate local utility f^r
        joint = {**x_others, self.robot_id: x_r}
        F_t   = self.local_utility_fn(joint)

        # Backpropagation
        self._backprop(path, F_t)

    def _rollout(self, node):
        """Random rollout policy from node.state. Returns list of actions taken."""
        state   = node.state
        actions = []
        depth   = 0
        while not state.is_terminal_state() and depth < self.rollout_depth:
            legal = state.get_legal_actions()
            if not legal:
                break
            a = random.choice(legal)
            actions.append(a)
            state = state.take_action(a)
            depth += 1
        return actions

    def _backprop(self, path, F_t):
        """
        Backpropagate F_t along path (Algorithm 2, line 8).

        Updates undiscounted stats (visits, cum_reward) for visited nodes only.

        D-UCT requires the gamma decay to be applied to ALL nodes every iteration,
        not just the visited path. Nodes on the path get visited=True (adds 1 and F_t
        on top of the decay); every other node gets visited=False (decay only). This
        is what makes D-UCT properly de-emphasize stale statistics over time.
        """
        path_set = set(id(n) for n in path)

        # Collect every node in the tree via DFS
        all_nodes = []
        self._collect_nodes(self.root, all_nodes)

        for node in all_nodes:
            on_path = id(node) in path_set
            if on_path:
                node.visits     += 1
                node.cum_reward += F_t
            node.update_discounted(F_t, visited=on_path, gamma=self.gamma)

    # UPDATE DISTRIBUTION
    def _update_sample_space(self):
        """
        Select X̂^r_n as the num_seq nodes with highest disc_q in tree.
        Resets q^r_n to uniform when X̂ changes.
        """
        candidates = []
        self._collect_nodes(self.root, candidates)

        # Sort by discounted Q descending, take top num_seq (exclude root)
        candidates.sort(key=lambda n: n.disc_q(), reverse=True)
        new_X_hat = [
            tuple(n.action_sequence)
            for n in candidates
            if n.action_sequence   # exclude root (empty sequence)
        ][:self.num_seq]

        # Reset to uniform when sample space changes
        if set(new_X_hat) != set(self.X_hat):
            self.X_hat = new_X_hat
            n = max(len(self.X_hat), 1)
            self.q = {s: 1.0 / n for s in self.X_hat}

    def _collect_nodes(self, node, out):
        """DFS to collect all nodes in tree."""
        out.append(node)
        for child in node.children:
            self._collect_nodes(child, out)

    def _update_distribution(self):
        """
        Gradient descent on product distribution q^r_n.

        For each x^r ∈ X̂^r_n:
            q  <-  q - alpha * q * ( (E[f^r] - E[f^r | x^r]) / beta  +  H(q)  +  ln(q) )
        then normalize.

        This is the mirror descent / natural gradient step.
        """
        if not self.X_hat:
            return

        # Estimate E_{q_n}[f^r]
        E_f = self._estimate_expectation(fixed_xr=None)

        # Entropy H(q^r_n)
        H = -sum(p * math.log(p) for p in self.q.values() if p > 0)

        new_q = {}
        for xr_tup in self.X_hat:
            # E_{q_n}[f^r | x^r]  (Algorithm 3, line 4)
            E_f_given_xr = self._estimate_expectation(fixed_xr=xr_tup)

            q_val = self.q.get(xr_tup, 1.0 / len(self.X_hat))
            ln_q  = math.log(q_val) if q_val > 1e-12 else -100.0

            # Algorithm 3 line 5 -- first order additive approximation for compute purposes
            delta = self.alpha * q_val * (
                (E_f - E_f_given_xr) / self.beta  +  H  +  ln_q
            )
            new_q[xr_tup] = max(1e-12, q_val - delta)

        # Normalize
        total = sum(new_q.values())
        if total > 0:
            self.q = {k: v / total for k, v in new_q.items()}
        else:
            n = len(self.X_hat)
            self.q = {k: 1.0 / n for k in self.X_hat}

    def _estimate_expectation(self, fixed_xr=None):
        """
        MC estimate of E_{q_n}[f^r] or E_{q_n}[f^r | x^r = fixed_xr].

        Sample from joint distribution and average.
        """
        total = 0.0
        for _ in range(self.num_samples):
            xr      = list(fixed_xr) if fixed_xr is not None else self._sample_own()
            x_others = self._sample_others()
            joint   = {**x_others, self.robot_id: xr}
            total  += self.local_utility_fn(joint)
        return total / self.num_samples

    def _sample_own(self):
        """Sample x^r from own distribution q^r_n."""
        return list(self._sample_from_dist(self.q))

    def _sample_others(self):
        """Sample x^(r) from each other robot's received distribution."""
        return {
            r: list(self._sample_from_dist(self.received_dists[r]))
            for r in self.robot_ids if r != self.robot_id
        }

    @staticmethod
    def _sample_from_dist(dist):
        """
        Sample a sequence tuple from a {tuple: prob} dict.
        Returns [] (empty sequence) if distribution is empty — default policy.
        """
        if not dist:
            return []
        keys  = list(dist.keys())
        probs = list(dist.values())
        total = sum(probs)
        if total <= 0:
            return random.choice(keys)
        probs = [p / total for p in probs]

        # Manual categorical sample (no numpy dependency)
        r   = random.random()
        cum = 0.0
        for key, p in zip(keys, probs):
            cum += p
            if r <= cum:
                return key
        return keys[-1]

    # FALLBACK: GREEDY TREE EXTRACTION
    def _greedy_tree_sequence(self):
        """
        Extract best sequence by greedy Q-value descent through tree.
        Used when distribution is empty (before first sample space is built).
        """
        node = self.root
        seq  = []
        while node.children:
            visited = [c for c in node.children if c.visits > 0]
            if not visited:
                break
            best = max(visited, key=lambda c: c.q())
            seq.append(best.action)
            node = best
        return seq


# PART 3 — MULTI-ROBOT RUNNER
class DecMCTSTeam:
    """
    Convenience wrapper that runs Dec-MCTS for a team of robots,
    handles the communication phase and exposes per-round metrics.

    """

    def __init__(self, planners: dict):
        """
        planners : {robot_id: DecMCTS}  — one planner per robot

        """
        self.planners = planners

    def iterate_and_communicate(self, n_outer=1, comm_period=1):
        """
        Run n_outer outer iterations for all robots, broadcasting
        distributions every comm_period outer iterations.

        comm_period=1  : communicate after every outer iteration (default)
        comm_period=K  : communicate once every K outer iterations
        comm_period=-1 : never communicate (fully decentralized / no comms)
        """
        for i in range(1, n_outer + 1):
            for p in self.planners.values():
                p.iterate(1)

            if comm_period > 0 and i % comm_period == 0:
                for rid, planner in self.planners.items():
                    _, q = planner.get_distribution()
                    for other_id, other_planner in self.planners.items():
                        if other_id != rid:
                            other_planner.receive_dist_dict(rid, q)

    def best_sequences(self):
        """Return {robot_id: action_sequence} for all robots."""
        return {rid: p.best_action_sequence() for rid, p in self.planners.items()}

    def entropies(self):
        """Return {robot_id: H(q^r_n)} — useful for convergence monitoring."""
        result = {}
        for rid, p in self.planners.items():
            probs    = list(p.q.values())
            H        = -sum(pr * math.log(pr) for pr in probs if pr > 0)
            result[rid] = H
        return result