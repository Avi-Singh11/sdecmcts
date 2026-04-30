"""
tiger_online_benchmark.py
-------------------------

for t in 0..H-1:
    build/run Dec-MCTS from each agent's current belief
    execute only the first action from each selected sequence
    sample transition + observation
    update each agent's belief using its local observation
    replan

Thus each individual Dec-MCTS call is open-loop, but the executed controller is
observation-conditioned through belief updates and replanning.

Action encoding matches the RSSDA Tiger script:
    OPEN_LEFT  = 0
    OPEN_RIGHT = 1
    LISTEN     = 2

MOD='C' observations:
    - both listen: each local observation is 0.75 accurate
    - exactly one listens: listener is 0.75 accurate, non-listener is uniform
    - opening: tiger resets uniformly; observations are otherwise uniform except
      the single-listener correction above
"""

from __future__ import annotations

import os 
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decmcts import DecMCTS, DecMCTSTeam
import decmcts


# =============================================================================
# Constants
# =============================================================================

TIGER_LEFT = 0
TIGER_RIGHT = 1

OPEN_LEFT = 0
OPEN_RIGHT = 1
LISTEN = 2

N_AGENTS = 2
N_STATES = 2
ACT_PER_AGENT = 3
OBS_PER_AGENT = 2
N_ACTS = ACT_PER_AGENT ** N_AGENTS
N_OBS = OBS_PER_AGENT ** N_AGENTS

ACTION_NAME = {
    OPEN_LEFT: "OL",
    OPEN_RIGHT: "OR",
    LISTEN: "L",
}

OBS_NAME = {
    TIGER_LEFT: "hear-left",
    TIGER_RIGHT: "hear-right",
}

# =============================================================================
# Tiger Model
# =============================================================================

class TigerModel:
    def __init__(self):
        self.T = [0.0] * (N_ACTS * N_STATES * N_STATES)
        self.O = [0.0] * (N_ACTS * N_STATES * N_OBS)
        self.R = [0.0] * (N_ACTS * N_STATES)
        self.init_belief = [0.5, 0.5]
        self._build()

    @staticmethod
    def joint_action(a0: int, a1: int) -> int:
        return a0 + ACT_PER_AGENT * a1

    @staticmethod
    def split_action(a: int) -> Tuple[int, int]:
        return a % ACT_PER_AGENT, a // ACT_PER_AGENT

    @staticmethod
    def joint_obs(o0: int, o1: int) -> int:
        return o0 + OBS_PER_AGENT * o1

    @staticmethod
    def split_obs(o: int) -> Tuple[int, int]:
        return o % OBS_PER_AGENT, o // OBS_PER_AGENT

    def _t_idx(self, a: int, s: int, sp: int) -> int:
        return a * N_STATES * N_STATES + s * N_STATES + sp

    def _o_idx(self, a: int, sp: int, o: int) -> int:
        return a * N_STATES * N_OBS + sp * N_OBS + o

    def _r_idx(self, a: int, s: int) -> int:
        return a * N_STATES + s

    def _set_t(self, a: int, s: int, sp: int, p: float) -> None:
        self.T[self._t_idx(a, s, sp)] = p

    def _set_o(self, a: int, sp: int, o: int, p: float) -> None:
        self.O[self._o_idx(a, sp, o)] = p

    def _set_r(self, a: int, s: int, r: float) -> None:
        self.R[self._r_idx(a, s)] = r

    def _build(self) -> None:
        both_listen = self.joint_action(LISTEN, LISTEN)

        for a in range(N_ACTS):
            a0, a1 = self.split_action(a)

            if a == both_listen:
                for s in range(N_STATES):
                    self._set_r(a, s, -2.0)
                    for sp in range(N_STATES):
                        self._set_t(a, s, sp, 1.0 if sp == s else 0.0)

                # MOD='C': both-listen local observations are 0.75/0.25.
                for sp in range(N_STATES):
                    for o in range(N_OBS):
                        o0, o1 = self.split_obs(o)
                        p0 = 0.75 if o0 == sp else 0.25
                        p1 = 0.75 if o1 == sp else 0.25
                        self._set_o(a, sp, o, p0 * p1)
                continue

            # Any non-both-listen action: tiger resets uniformly.
            for s in range(N_STATES):
                r = (
                    -101 * ((a0 == s) * (a1 == LISTEN) +
                            (a1 == s) * (a0 == LISTEN))
                    -50 * ((a0 == s) * (a1 == s))
                    -100 * ((a0 != LISTEN) * (a0 != s) * (a1 == s) +
                            (a1 != LISTEN) * (a1 != s) * (a0 == s))
                    +9 * ((a0 != LISTEN) * (a0 != s) * (a1 == LISTEN) +
                          (a1 != LISTEN) * (a1 != s) * (a0 == LISTEN))
                    +20 * ((a0 != LISTEN) * (a0 != s) *
                           (a1 != LISTEN) * (a1 != s))
                )
                self._set_r(a, s, float(r))
                for sp in range(N_STATES):
                    self._set_t(a, s, sp, 0.5)

            # Default opening obs model: uniform joint observations.
            for sp in range(N_STATES):
                for o in range(N_OBS):
                    self._set_o(a, sp, o, 1.0 / N_OBS)

        # MOD='C': exactly one listener gets 0.75/0.25; non-listener gets uniform.
        for a in range(N_ACTS):
            a0, a1 = self.split_action(a)
            if not ((a0 == LISTEN) ^ (a1 == LISTEN)):
                continue

            for sp in range(N_STATES):
                for o in range(N_OBS):
                    o0, o1 = self.split_obs(o)
                    if a0 == LISTEN:
                        p_listen = 0.75 if o0 == sp else 0.25
                        self._set_o(a, sp, o, p_listen * 0.5)
                    else:
                        p_listen = 0.75 if o1 == sp else 0.25
                        self._set_o(a, sp, o, 0.5 * p_listen)

    def transition_prob(self, s: int, a: int, sp: int) -> float:
        return self.T[self._t_idx(a, s, sp)]

    def obs_prob(self, sp: int, a: int, o: int) -> float:
        return self.O[self._o_idx(a, sp, o)]

    def reward(self, s: int, a: int) -> float:
        return self.R[self._r_idx(a, s)]

    def sample_initial_state(self, rng: random.Random) -> int:
        return 0 if rng.random() < self.init_belief[0] else 1

    def sample_next_state(self, s: int, a: int, rng: random.Random) -> int:
        r = rng.random()
        cum = 0.0
        for sp in range(N_STATES):
            cum += self.transition_prob(s, a, sp)
            if r <= cum:
                return sp
        return N_STATES - 1

    def sample_joint_obs(self, sp: int, a: int, rng: random.Random) -> int:
        r = rng.random()
        cum = 0.0
        for o in range(N_OBS):
            cum += self.obs_prob(sp, a, o)
            if r <= cum:
                return o
        return N_OBS - 1

    def local_obs_prob(self, agent_id: int, sp: int, joint_a: int, local_o: int) -> float:
        """
        Marginal P(o_i | s', joint_a), summing over the other agent's obs.
        """
        total = 0.0
        for joint_o in range(N_OBS):
            o0, o1 = self.split_obs(joint_o)
            if (agent_id == 0 and o0 == local_o) or (agent_id == 1 and o1 == local_o):
                total += self.obs_prob(sp, joint_a, joint_o)
        return total

# =============================================================================
# Planning state and exact belief calculations
# =============================================================================

@dataclass(frozen=True)
class TigerPlanningState:
    depth: int
    max_horizon: int

    def get_legal_actions(self) -> List[int]:
        if self.is_terminal_state():
            return []
        return [OPEN_LEFT, OPEN_RIGHT, LISTEN]

    def take_action(self, action: int) -> "TigerPlanningState":
        return TigerPlanningState(self.depth + 1, self.max_horizon)

    def is_terminal_state(self) -> bool:
        return self.depth >= self.max_horizon


def normalize_belief(b: Sequence[float]) -> List[float]:
    total = float(sum(b))
    if total <= 0:
        return [0.5, 0.5]
    return [float(x) / total for x in b]


def predict_belief_open_loop(
    belief: Sequence[float],
    joint_a: int,
    model: TigerModel,
) -> List[float]:
    """
    Predict next belief without conditioning on observation:
        b'(s') = sum_s b(s) T(s' | s, a)
    Used inside open-loop sequence evaluation and heuristic rollouts.
    """
    out = [0.0, 0.0]
    for s in range(N_STATES):
        for sp in range(N_STATES):
            out[sp] += belief[s] * model.transition_prob(s, joint_a, sp)
    return normalize_belief(out)


def update_local_belief(
    belief: Sequence[float],
    joint_a: int,
    local_obs: int,
    agent_id: int,
    model: TigerModel,
) -> List[float]:
    """
    Bayes update using each agent's local observation:
        b'(s') proportional to P(o_i | s', a) sum_s T(s' | s,a)b(s)
    """
    pred = predict_belief_open_loop(belief, joint_a, model)
    post = [
        pred[sp] * model.local_obs_prob(agent_id, sp, joint_a, local_obs)
        for sp in range(N_STATES)
    ]
    return normalize_belief(post)


def expected_open_loop_return(
    belief: Sequence[float],
    joint_sequences: Dict[int, Sequence[int]],
    model: TigerModel,
    horizon: int,
    null_action: int = LISTEN,
) -> float:
    """
    Exact expected return for fixed open-loop joint action sequences from belief.

    Future observations are not used because the sequence is fixed. Belief evolves
    only through the transition model.
    """
    b = list(belief)
    total = 0.0

    seq0 = list(joint_sequences.get(0, []))
    seq1 = list(joint_sequences.get(1, []))

    for t in range(horizon):
        a0 = seq0[t] if t < len(seq0) else null_action
        a1 = seq1[t] if t < len(seq1) else null_action
        joint_a = model.joint_action(a0, a1)

        total += sum(b[s] * model.reward(s, joint_a) for s in range(N_STATES))
        b = predict_belief_open_loop(b, joint_a, model)

    return total


def make_local_utility_fn(
    agent_id: int,
    belief: Sequence[float],
    model: TigerModel,
    horizon: int,
    mode: str = "difference",
) -> Callable[[Dict[int, List[int]]], float]:
    """
    Local utility for Dec-MCTS.

    mode='difference':
        f^r(x) = g(x^r, x^{-r}) - g(null^r, x^{-r})
        This matches the Dec-MCTS paper's local utility idea.

    mode='global':
        f^r(x) = g(x)
        Useful for debugging if difference rewards create poor local equilibria.
    """
    if mode not in {"difference", "global"}:
        raise ValueError("mode must be 'difference' or 'global'.")

    b0 = list(belief)

    def g(joint: Dict[int, List[int]]) -> float:
        return expected_open_loop_return(b0, joint, model, horizon)

    def local_utility(joint: Dict[int, List[int]]) -> float:
        if mode == "global":
            return g(joint)

        null_joint = {rid: list(seq) for rid, seq in joint.items()}
        null_joint[agent_id] = [LISTEN] * horizon
        return g(joint) - g(null_joint)

    return local_utility

    def update_joint_belief(belief, joint_a, joint_obs, model):
        pred = predict_belief_open_loop(belief, joint_a, model)
        post = [
            pred[sp] * model.obs_prob(sp, joint_a, joint_obs)
            for sp in range(N_STATES)
        ]
        return normalize_belief(post)
    
   

# =============================================================================
# Heuristic
# =============================================================================

def heuristic_action_from_belief(
    belief: Sequence[float],
    open_threshold: float = 0.85,
) -> int:
    """
    Non-clairvoyant belief heuristic.

    If tiger is likely left, open right.
    If tiger is likely right, open left.
    Otherwise listen.
    """
    p_left = belief[TIGER_LEFT]
    p_right = belief[TIGER_RIGHT]

    if p_left >= open_threshold:
        return OPEN_RIGHT
    if p_right >= open_threshold:
        return OPEN_LEFT
    return LISTEN


def make_tiger_rollout_policy(
    belief: Sequence[float],
    model: TigerModel,
    horizon: int,
    open_threshold: float,
) -> Callable:
    """
    Create rollout_policy(planner, node, x_others) -> list[action].

    This heuristic never samples or uses the hidden true state. It only uses the
    planner's current belief and predicted open-loop transitions.
    """
    root_belief = list(belief)

    def rollout_policy(planner: DecMCTS, node, x_others: Dict[int, List[int]]) -> List[int]:
        b = list(root_belief)
        own_prefix = list(node.action_sequence)
        other_id = next(r for r in planner.robot_ids if r != planner.robot_id)
        other_seq = list(x_others.get(other_id, []))

        # Roll belief forward through the already chosen prefix.
        for t, my_a in enumerate(own_prefix):
            other_a = other_seq[t] if t < len(other_seq) else LISTEN
            if planner.robot_id == 0:
                joint_a = model.joint_action(my_a, other_a)
            else:
                joint_a = model.joint_action(other_a, my_a)
            b = predict_belief_open_loop(b, joint_a, model)

        # Complete the sequence with the belief heuristic.
        actions: List[int] = []
        for t in range(len(own_prefix), horizon):
            my_a = heuristic_action_from_belief(b, open_threshold=open_threshold)
            actions.append(my_a)

            other_a = other_seq[t] if t < len(other_seq) else LISTEN
            if planner.robot_id == 0:
                joint_a = model.joint_action(my_a, other_a)
            else:
                joint_a = model.joint_action(other_a, my_a)
            b = predict_belief_open_loop(b, joint_a, model)

        return actions

    return rollout_policy


# =============================================================================
# Online Dec-MCTS episode simulation
# =============================================================================

def run_planning_step(
    beliefs: Dict[int, List[float]],
    remaining_horizon: int,
    model: TigerModel,
    args,
    seed: int,
) -> Tuple[Dict[int, int], Dict[int, List[int]], Dict[int, float]]:
    """
    Run one online planning step from the agents' current beliefs and return
    first actions.
    """
    planners = {}

    for rid in range(N_AGENTS):
        init_state = TigerPlanningState(depth=0, max_horizon=remaining_horizon)
        local_fn = make_local_utility_fn(
            agent_id=rid,
            belief=beliefs[rid],
            model=model,
            horizon=remaining_horizon,
            mode=args.local_utility,
        )

        rollout_policy = make_tiger_rollout_policy(
            belief=beliefs[rid],
            model=model,
            horizon=remaining_horizon,
            open_threshold=args.open_threshold,
        )

        planners[rid] = DecMCTS(
            robot_id=rid,
            robot_ids=[0, 1],
            init_state=init_state,
            local_utility_fn=local_fn,
            rollout_policy=rollout_policy,
            default_sequence_fn=lambda _rid, H=remaining_horizon: [LISTEN] * H,
            gamma=args.gamma,
            cp=args.cp,
            rollout_depth=remaining_horizon,
            tau=args.tau,
            num_seq=args.num_seq,
            num_samples=args.num_samples,
            beta_init=args.beta_init,
            beta_decay=args.beta_decay,
            alpha=args.alpha,
            seed=seed + 1009 * rid,
        )

    team = DecMCTSTeam(planners)
    team.iterate_and_communicate(n_outer=args.outer_iters, comm_period=args.comm_period)

    raw_actions = {}
    actions = {}
    sequences = team.best_sequences()

    for rid in range(N_AGENTS):
        a = planners[rid].best_action()
        raw_a = LISTEN if a is None else int(a)
        raw_actions[rid] = raw_a

        if getattr(args, "guard_actions", False):
            actions[rid] = guard_tiger_action(
                belief=beliefs[rid],
                proposed_action=raw_a,
                open_threshold=args.open_threshold,
                force_open_when_confident=getattr(args, "force_open_when_confident", False),
            )
        else:
            actions[rid] = raw_a

    root_masses = {
        rid: root_action_masses(planner)
        for rid, planner in planners.items()
    }

    return actions, raw_actions, sequences, team.entropies(), root_masses

def guard_tiger_action(
    belief,
    proposed_action,
    open_threshold,
    force_open_when_confident=False,
    ):
        """
        Safety guard for online Tiger execution.

        If belief is not confident, veto OPEN actions and LISTEN instead.
        If belief is confident and an open is proposed, force the safe-looking door.
        Optionally force opening whenever belief is confident.
        """
        p_left = belief[TIGER_LEFT]
        p_right = belief[TIGER_RIGHT]
        confidence = max(p_left, p_right)

        safe_open = OPEN_RIGHT if p_left >= p_right else OPEN_LEFT

        if confidence < open_threshold:
            if proposed_action in (OPEN_LEFT, OPEN_RIGHT):
                return LISTEN
            return proposed_action

        # Confident belief: if opening, make sure it is the safe-looking door.
        if proposed_action in (OPEN_LEFT, OPEN_RIGHT):
            return safe_open

        if force_open_when_confident:
            return safe_open

        return proposed_action



def update_joint_belief(belief, joint_a, joint_obs, model):
    """
    Bayes update using the full joint observation:
        b'(s') proportional to P(o_joint | s', a) * sum_s T(s' | s,a)b(s)
    """
    pred = predict_belief_open_loop(belief, joint_a, model)
    post = [
        pred[sp] * model.obs_prob(sp, joint_a, joint_obs)
        for sp in range(N_STATES)
    ]
    return normalize_belief(post)


def simulate_online_episode(
    model: TigerModel,
    args,
    episode_seed: int,
    verbose: bool = False,
) -> float:
    rng = random.Random(episode_seed)
    true_state = model.sample_initial_state(rng)

    # Belief at the most recent communication synchronization point.
    common_belief = list(model.init_belief)

    # Joint action/observation history since the last communication event.
    pending_history = []

    # Private beliefs used for planning between communication events.
    beliefs = {
        0: list(model.init_belief),
        1: list(model.init_belief),
    }

    total_reward = 0.0

    if verbose:
        print(f"initial true_state={'LEFT' if true_state == TIGER_LEFT else 'RIGHT'}")
        print(f"initial beliefs={beliefs}")

    for t in range(args.horizon):
        remaining = args.horizon - t

        actions, raw_actions, sequences, entropies, root_masses = run_planning_step(
            beliefs=beliefs,
            remaining_horizon=remaining,
            model=model,
            args=args,
            seed=episode_seed + 7919 * t,
        )

        pre_beliefs = {
            0: list(beliefs[0]),
            1: list(beliefs[1]),
        }

        a0, a1 = actions[0], actions[1]
        joint_a = model.joint_action(a0, a1)

        reward = model.reward(true_state, joint_a)
        total_reward += reward

        next_state = model.sample_next_state(true_state, joint_a, rng)
        joint_o = model.sample_joint_obs(next_state, joint_a, rng)
        o0, o1 = model.split_obs(joint_o)

        # Each agent first updates using only its own private local observation.
        beliefs[0] = update_local_belief(beliefs[0], joint_a, o0, 0, model)
        beliefs[1] = update_local_belief(beliefs[1], joint_a, o1, 1, model)

        # Store the full joint action/observation so it can be shared later.
        pending_history.append((joint_a, joint_o))

        if should_env_communicate(args, t, a0, a1):
            for hist_joint_a, hist_joint_o in pending_history:
                common_belief = update_joint_belief(
                    common_belief,
                    hist_joint_a,
                    hist_joint_o,
                    model,
                )

            beliefs[0] = list(common_belief)
            beliefs[1] = list(common_belief)
            pending_history = []

        if verbose:
            print(
                f"t={t:02d} rem={remaining:02d} "
                f"a=({ACTION_NAME[a0]},{ACTION_NAME[a1]}) "
                f"r={reward:6.1f} "
                f"obs=({OBS_NAME[o0]},{OBS_NAME[o1]}) "
                f"next={'LEFT' if next_state == TIGER_LEFT else 'RIGHT'} "
                f"belief0={[round(x,3) for x in beliefs[0]]} "
                f"belief1={[round(x,3) for x in beliefs[1]]} "
                f"H={{{0}: {entropies.get(0,0):.3f}, {1}: {entropies.get(1,0):.3f}}}"
                f"raw_actions=({ACTION_NAME[raw_actions[0]]},{ACTION_NAME[raw_actions[1]]}) "
                f"exec_actions=({ACTION_NAME[a0]},{ACTION_NAME[a1]})"
            )

            print(f"    pre_belief0={[round(x, 3) for x in pre_beliefs[0]]}")
            print(f"    pre_belief1={[round(x, 3) for x in pre_beliefs[1]]}")
            print(f"    candidate_values_b0={candidate_values[0]}")
            print(f"    candidate_values_b1={candidate_values[1]}")
                
            print(f"    seq0={[ACTION_NAME[x] for x in sequences[0]]}")
            print(f"    seq1={[ACTION_NAME[x] for x in sequences[1]]}")
            print(f"    root_mass={root_masses}")

        true_state = next_state

    return total_reward


def confidence_interval_95(values: Sequence[float]) -> Tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n <= 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    se = math.sqrt(var / n)
    return mean, 1.96 * se

# HELPERS
def root_action_masses(planner):
    masses = {OPEN_LEFT: 0.0, OPEN_RIGHT: 0.0, LISTEN: 0.0}

    for seq, p in planner.q.items():
        if len(seq) > 0:
            masses[seq[0]] += p

    return {ACTION_NAME[a]: round(v, 4) for a, v in masses.items()}

        
def run_episode_worker(payload):
    ep, args_dict = payload
    args = argparse.Namespace(**args_dict)
    if ep == 0:
        print("WORKER using decmcts from:", decmcts.__file__, flush=True)

    model = TigerModel()
    ret = simulate_online_episode(
        model=model,
        args=args,
        episode_seed=args.seed + 104729 * ep,
        verbose=False,
    )
    return ep, ret

def should_env_communicate(args, t, a0, a1):
    if args.env_comm_mode == "none":
        return False

    if args.env_comm_mode == "both-listen":
        return a0 == LISTEN and a1 == LISTEN

    # periodic mode
    return args.env_comm_period > 0 and (t + 1) % args.env_comm_period == 0

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=50)

    # Dec-MCTS planning budget.
    parser.add_argument("--outer-iters", type=int, default=40)
    parser.add_argument("--tau", type=int, default=100)
    parser.add_argument("--num-seq", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--comm-period", type=int, default=1)
    parser.add_argument("--env-comm-period", type=int, default=1)

    # Dec-MCTS hyperparameters.
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--cp", type=float, default=0.5)
    parser.add_argument("--beta-init", type=float, default=2.0)
    parser.add_argument("--beta-decay", type=float, default=0.995)
    parser.add_argument("--alpha", type=float, default=0.02)

    # Tiger heuristic/evaluation.
    parser.add_argument("--open-threshold", type=float, default=0.85)
    parser.add_argument("--local-utility", choices=["difference", "global"], default="difference")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose-first", action="store_true")
    parser.add_argument("--guard-actions", action="store_true", help="Veto OPEN actions when belief confidence is below open_threshold.")

    parser.add_argument("--force-open-when-confident", action="store_true", help="With --guard-actions, open the safe-looking door whenever belief is confident.")

    parser.add_argument("--env-comm-mode", choices=["periodic", "both-listen", "none"], default="periodic", help="Execution-time observation sharing mode.")
    parser.add_argument("--diverse-sample-space", action="store_true")

    # Adding parallelism across all cores
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes. Use -1 for all CPU cores.",
    )

    args = parser.parse_args()

    model = TigerModel()

    print("Online Dec-MCTS Tiger benchmark")
    print(f"horizon={args.horizon}, episodes={args.episodes}")
    print(
        f"planning: outer_iters={args.outer_iters}, tau={args.tau}, "
        f"num_seq={args.num_seq}, num_samples={args.num_samples}")
    print(
        f"utility={args.local_utility}, open_threshold={args.open_threshold}, "
        f"env_comm_period={args.env_comm_period}, seed={args.seed}")
    print()

    print("MAIN using decmcts from:", decmcts.__file__, flush=True)

    returns = []

    # Keep verbose runs serial so the debug trace is readable.
    if args.jobs == 1 or args.verbose_first:
        for ep in range(args.episodes):
            ret = simulate_online_episode(
                model=model,
                args=args,
                episode_seed=args.seed + 104729 * ep,
                verbose=(args.verbose_first and ep == 0),
            )
            returns.append(ret)

            if (ep + 1) % max(1, args.episodes // 10) == 0 or ep == 0:
                mean, ci = confidence_interval_95(returns)
                print(
                    f"episode {ep + 1:4d}/{args.episodes}: "
                    f"last={ret:8.3f}, mean={mean:8.3f} ± {ci:.3f}",
                    flush=True,
                )

    else:
        n_jobs = os.cpu_count() if args.jobs == -1 else args.jobs
        n_jobs = max(1, int(n_jobs))

        args_dict = vars(args).copy()

        print(f"Running episodes in parallel with {n_jobs} workers", flush=True)

        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [
                ex.submit(run_episode_worker, (ep, args_dict))
                for ep in range(args.episodes)
            ]

            completed = 0
            for fut in as_completed(futures):
                ep, ret = fut.result()
                returns.append(ret)
                completed += 1

                if completed % max(1, args.episodes // 10) == 0 or completed == 1:
                    mean, ci = confidence_interval_95(returns)
                    print(
                        f"completed {completed:4d}/{args.episodes}: "
                        f"last_ep={ep:4d}, last={ret:8.3f}, "
                        f"mean={mean:8.3f} ± {ci:.3f}",
                        flush=True,
                    )

    mean, ci = confidence_interval_95(returns)
    print()
    print(f"Final mean return: {mean:.5f} ± {ci:.5f} (95% CI)")



if __name__ == "__main__":
    main()
