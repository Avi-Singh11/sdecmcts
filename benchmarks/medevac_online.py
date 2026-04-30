"""
medevac_online.py
---------------------------
Online/receding-horizon Maritime MEDEVAC benchmark

Execution-time communication is separate from planning-time Dec-MCTS
distribution exchange:
    --comm-period controls q-distribution exchange inside each planning call
    --env-comm-period controls observation/belief sharing across environment steps
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decmcts import DecMCTS, DecMCTSTeam


# =============================================================================
# Constants
# =============================================================================

G = 4
N_AGENTS = 2

WAIT = 0
ADVANCE = 1
EXCHANGE = 2

ACTION_NAME = {
    WAIT: "WAIT",
    ADVANCE: "ADVANCE",
    EXCHANGE: "EXCHANGE",
}

NOT_AT_TARGET = 0
AT_TARGET = 1

OBS_NAME = {
    NOT_AT_TARGET: "not-at-target",
    AT_TARGET: "at-target",
}

ACT_PER_AGENT = 3
OBS_PER_AGENT = 2
N_ACTIONS = ACT_PER_AGENT ** N_AGENTS
N_OBS = OBS_PER_AGENT ** N_AGENTS
N_STATES = (G * G) * (G * G) * 2

PATIENT = (1, 1)
HOSPITAL = (3, 3)
HELO_START = (0, 1)
SHIP_START = (1, 0)

P_MOVE_HELO = 0.95
P_MOVE_SHIP = 0.85
P_PICKUP = 0.95
P_DROPOFF = 0.95

STEP_COST = -0.3
WRONG_EXCHANGE_COST = -1.0
PICKUP_REWARD = 5.0
DROPOFF_REWARD = 12.0
PICKUP_MISMATCH_PENALTY = -6.0
DROPOFF_MISMATCH_PENALTY = -6.0


# =============================================================================
# State utilities
# =============================================================================

def pos_to_idx(x: int, y: int) -> int:
    return y * G + x


def idx_to_pos(idx: int) -> Tuple[int, int]:
    return idx % G, idx // G


def state_id(hx: int, hy: int, sx: int, sy: int, carry: int) -> int:
    return carry * (G * G * G * G) + pos_to_idx(hx, hy) * (G * G) + pos_to_idx(sx, sy)


def id_to_state(s: int) -> Tuple[int, int, int, int, int]:
    carry = 1 if s >= (G * G * G * G) else 0
    rem = s - carry * (G * G * G * G)
    hpos = rem // (G * G)
    spos = rem % (G * G)
    hx, hy = idx_to_pos(hpos)
    sx, sy = idx_to_pos(spos)
    return hx, hy, sx, sy, carry


def target_for_carry(carry: int) -> Tuple[int, int]:
    return PATIENT if carry == 0 else HOSPITAL


def at_patient(x: int, y: int) -> bool:
    return (x, y) == PATIENT


def at_hospital(x: int, y: int) -> bool:
    return (x, y) == HOSPITAL


def at_target_for_agent(x: int, y: int, carry: int) -> int:
    return AT_TARGET if (x, y) == target_for_carry(carry) else NOT_AT_TARGET


def next_step_toward(x: int, y: int, tx: int, ty: int, prefer: str) -> Tuple[int, int]:
    if (x, y) == (tx, ty):
        return x, y
    if prefer == "H":
        if x < tx:
            return x + 1, y
        if y < ty:
            return x, y + 1
    else:
        if y < ty:
            return x, y + 1
        if x < tx:
            return x + 1, y
    return x, y


def advance_helo(x: int, y: int, carry: int) -> Tuple[int, int, float]:
    tx, ty = target_for_carry(carry)
    nx, ny = next_step_toward(x, y, tx, ty, prefer="H")
    p_succ = P_MOVE_HELO if (nx, ny) != (x, y) else 1.0
    return nx, ny, p_succ


def advance_ship(x: int, y: int, carry: int) -> Tuple[int, int, float]:
    tx, ty = target_for_carry(carry)
    nx, ny = next_step_toward(x, y, tx, ty, prefer="V")
    p_succ = P_MOVE_SHIP if (nx, ny) != (x, y) else 1.0
    return nx, ny, p_succ


# =============================================================================
# MEDEVAC model
# =============================================================================

class MedevacModel:
    """Sparse generative model for Maritime MEDEVAC."""

    def __init__(self):
        self.rewards: List[float] = [0.0] * (N_ACTIONS * N_STATES)
        self.transitions: List[Tuple[array, array]] = []
        self.obs_of_state_action: List[int] = [0] * (N_ACTIONS * N_STATES)
        self.init_belief = [0.0] * N_STATES
        self.init_state = state_id(HELO_START[0], HELO_START[1], SHIP_START[0], SHIP_START[1], 0)
        self.init_belief[self.init_state] = 1.0
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

    def reward(self, s: int, joint_a: int) -> float:
        return self.rewards[joint_a * N_STATES + s]

    def transition_sparse(self, s: int, joint_a: int) -> Tuple[array, array]:
        return self.transitions[joint_a * N_STATES + s]

    def obs_from_next_state(self, sp: int, joint_a: int) -> int:
        return self.obs_of_state_action[joint_a * N_STATES + sp]

    def sample_initial_state(self, rng: random.Random) -> int:
        return self.init_state

    def sample_next_state(self, s: int, joint_a: int, rng: random.Random) -> int:
        indices, probs = self.transition_sparse(s, joint_a)
        r = rng.random()
        cum = 0.0
        for sp, p in zip(indices, probs):
            cum += p
            if r <= cum:
                return int(sp)
        return int(indices[-1])

    def _build(self) -> None:
        self.transitions = [None] * (N_ACTIONS * N_STATES)  # type: ignore

        for a0 in range(ACT_PER_AGENT):
            for a1 in range(ACT_PER_AGENT):
                joint_a = self.joint_action(a0, a1)
                for s in range(N_STATES):
                    hx, hy, sx, sy, carry = id_to_state(s)
                    r = STEP_COST

                    helo_at_pat = at_patient(hx, hy)
                    ship_at_pat = at_patient(sx, sy)
                    helo_at_hosp = at_hospital(hx, hy)
                    ship_at_hosp = at_hospital(sx, sy)

                    if a0 == EXCHANGE and not (helo_at_pat or helo_at_hosp):
                        r += WRONG_EXCHANGE_COST
                    if a1 == EXCHANGE and not (ship_at_pat or ship_at_hosp):
                        r += WRONG_EXCHANGE_COST

                    if carry == 0:
                        if helo_at_pat and ship_at_pat and a0 == EXCHANGE and a1 == EXCHANGE:
                            r += P_PICKUP * PICKUP_REWARD
                        if (
                            (a0 == EXCHANGE and helo_at_pat and not (a1 == EXCHANGE and ship_at_pat))
                            or (a1 == EXCHANGE and ship_at_pat and not (a0 == EXCHANGE and helo_at_pat))
                        ):
                            r += PICKUP_MISMATCH_PENALTY
                    else:
                        if helo_at_hosp and ship_at_hosp and a0 == EXCHANGE and a1 == EXCHANGE:
                            r += P_DROPOFF * DROPOFF_REWARD
                        if (
                            (a0 == EXCHANGE and helo_at_hosp and not (a1 == EXCHANGE and ship_at_hosp))
                            or (a1 == EXCHANGE and ship_at_hosp and not (a0 == EXCHANGE and helo_at_hosp))
                        ):
                            r += DROPOFF_MISMATCH_PENALTY

                    self.rewards[joint_a * N_STATES + s] = r

                    if a0 == ADVANCE:
                        nx_h, ny_h, p_succ_h = advance_helo(hx, hy, carry)
                    else:
                        nx_h, ny_h, p_succ_h = hx, hy, 1.0
                    if a1 == ADVANCE:
                        nx_s, ny_s, p_succ_s = advance_ship(sx, sy, carry)
                    else:
                        nx_s, ny_s, p_succ_s = sx, sy, 1.0

                    cases = [
                        (nx_h, ny_h, nx_s, ny_s, p_succ_h * p_succ_s),
                        (nx_h, ny_h, sx, sy, p_succ_h * (1.0 - p_succ_s)),
                        (hx, hy, nx_s, ny_s, (1.0 - p_succ_h) * p_succ_s),
                        (hx, hy, sx, sy, (1.0 - p_succ_h) * (1.0 - p_succ_s)),
                    ]

                    if a0 == EXCHANGE:
                        cases = [(hx, hy, xb, yb, p if (xp == hx and yp == hy) else 0.0)
                                 for (xp, yp, xb, yb, p) in cases]
                    if a1 == EXCHANGE:
                        cases = [(xp, yp, sx, sy, p if (xb == sx and yb == sy) else 0.0)
                                 for (xp, yp, xb, yb, p) in cases]

                    accum: Dict[int, float] = {}
                    for hx2, hy2, sx2, sy2, p_case in cases:
                        if p_case <= 0.0:
                            continue
                        if carry == 0 and helo_at_pat and ship_at_pat and a0 == EXCHANGE and a1 == EXCHANGE:
                            s_succ = state_id(hx2, hy2, sx2, sy2, 1)
                            s_fail = state_id(hx2, hy2, sx2, sy2, 0)
                            accum[s_succ] = accum.get(s_succ, 0.0) + p_case * P_PICKUP
                            accum[s_fail] = accum.get(s_fail, 0.0) + p_case * (1.0 - P_PICKUP)
                        elif carry == 1 and helo_at_hosp and ship_at_hosp and a0 == EXCHANGE and a1 == EXCHANGE:
                            s_succ = state_id(hx2, hy2, sx2, sy2, 0)
                            s_fail = state_id(hx2, hy2, sx2, sy2, 1)
                            accum[s_succ] = accum.get(s_succ, 0.0) + p_case * P_DROPOFF
                            accum[s_fail] = accum.get(s_fail, 0.0) + p_case * (1.0 - P_DROPOFF)
                        else:
                            s2 = state_id(hx2, hy2, sx2, sy2, carry)
                            accum[s2] = accum.get(s2, 0.0) + p_case

                    indices = array("i", [])
                    probs = array("d", [])
                    for sp, p in accum.items():
                        if p > 0:
                            indices.append(sp)
                            probs.append(p)
                    self.transitions[joint_a * N_STATES + s] = (indices, probs)

        for joint_a in range(N_ACTIONS):
            for sp in range(N_STATES):
                hx, hy, sx, sy, carry = id_to_state(sp)
                o0 = at_target_for_agent(hx, hy, carry)
                o1 = at_target_for_agent(sx, sy, carry)
                self.obs_of_state_action[joint_a * N_STATES + sp] = self.joint_obs(o0, o1)


# =============================================================================
# Planning state and belief updates
# =============================================================================

@dataclass(frozen=True)
class MedevacPlanningState:
    depth: int
    max_horizon: int

    def get_legal_actions(self) -> List[int]:
        return [] if self.is_terminal_state() else [WAIT, ADVANCE, EXCHANGE]

    def take_action(self, action: int) -> "MedevacPlanningState":
        return MedevacPlanningState(self.depth + 1, self.max_horizon)

    def is_terminal_state(self) -> bool:
        return self.depth >= self.max_horizon


def normalize_belief(b: Sequence[float]) -> List[float]:
    total = float(sum(b))
    if total <= 0:
        out = [0.0] * N_STATES
        out[state_id(HELO_START[0], HELO_START[1], SHIP_START[0], SHIP_START[1], 0)] = 1.0
        return out
    return [float(x) / total for x in b]


def predict_belief_open_loop(belief: Sequence[float], joint_a: int, model: MedevacModel) -> List[float]:
    out = [0.0] * N_STATES
    for s, bs in enumerate(belief):
        if bs <= 0.0:
            continue
        indices, probs = model.transition_sparse(s, joint_a)
        for sp, p in zip(indices, probs):
            out[int(sp)] += bs * p
    return normalize_belief(out)


def update_local_belief(belief: Sequence[float], joint_a: int, local_obs: int, agent_id: int, model: MedevacModel) -> List[float]:
    pred = predict_belief_open_loop(belief, joint_a, model)
    post = [0.0] * N_STATES
    for sp, p in enumerate(pred):
        if p <= 0.0:
            continue
        joint_o = model.obs_from_next_state(sp, joint_a)
        o0, o1 = model.split_obs(joint_o)
        obs_i = o0 if agent_id == 0 else o1
        if obs_i == local_obs:
            post[sp] = p
    return normalize_belief(post)


def update_joint_belief(belief: Sequence[float], joint_a: int, joint_obs: int, model: MedevacModel) -> List[float]:
    pred = predict_belief_open_loop(belief, joint_a, model)
    post = [0.0] * N_STATES
    for sp, p in enumerate(pred):
        if p <= 0.0:
            continue
        if model.obs_from_next_state(sp, joint_a) == joint_obs:
            post[sp] = p
    return normalize_belief(post)


def expected_open_loop_return(belief: Sequence[float], joint_sequences: Dict[int, Sequence[int]], model: MedevacModel,
                              horizon: int, null_action: int = WAIT) -> float:
    b = list(belief)
    total = 0.0
    seq0 = list(joint_sequences.get(0, []))
    seq1 = list(joint_sequences.get(1, []))
    for t in range(horizon):
        a0 = seq0[t] if t < len(seq0) else null_action
        a1 = seq1[t] if t < len(seq1) else null_action
        joint_a = model.joint_action(a0, a1)
        total += sum(bs * model.reward(s, joint_a) for s, bs in enumerate(b) if bs > 0.0)
        b = predict_belief_open_loop(b, joint_a, model)
    return total


def make_local_utility_fn(agent_id: int, belief: Sequence[float], model: MedevacModel, horizon: int,
                          mode: str = "difference") -> Callable[[Dict[int, List[int]]], float]:
    if mode not in {"difference", "global"}:
        raise ValueError("mode must be 'difference' or 'global'.")
    b0 = list(belief)

    def g(joint: Dict[int, List[int]]) -> float:
        return expected_open_loop_return(b0, joint, model, horizon)

    def local_utility(joint: Dict[int, List[int]]) -> float:
        if mode == "global":
            return g(joint)
        null_joint = {rid: list(seq) for rid, seq in joint.items()}
        null_joint[agent_id] = [WAIT] * horizon
        return g(joint) - g(null_joint)

    return local_utility


# =============================================================================
# Rollout heuristic
# =============================================================================

def marginal_progress_stats(belief: Sequence[float], agent_id: int) -> Tuple[float, float, float]:
    p_self = 0.0
    p_other = 0.0
    p_both = 0.0
    for s, bs in enumerate(belief):
        if bs <= 0.0:
            continue
        hx, hy, sx, sy, carry = id_to_state(s)
        h_at = at_target_for_agent(hx, hy, carry) == AT_TARGET
        s_at = at_target_for_agent(sx, sy, carry) == AT_TARGET
        if agent_id == 0:
            self_at, other_at = h_at, s_at
        else:
            self_at, other_at = s_at, h_at
        if self_at:
            p_self += bs
        if other_at:
            p_other += bs
        if self_at and other_at:
            p_both += bs
    return p_self, p_other, p_both


def heuristic_action_from_belief(belief: Sequence[float], agent_id: int,
                                 exchange_threshold: float = 0.80,
                                 wait_if_self_ready_threshold: float = 0.70) -> int:
    p_self, _p_other, p_both = marginal_progress_stats(belief, agent_id)
    if p_both >= exchange_threshold:
        return EXCHANGE
    if p_self >= wait_if_self_ready_threshold:
        return WAIT
    return ADVANCE


def make_medevac_rollout_policy(belief: Sequence[float], model: MedevacModel, horizon: int,
                                exchange_threshold: float,
                                wait_if_self_ready_threshold: float) -> Callable:
    root_belief = list(belief)

    def rollout_policy(planner: DecMCTS, node, x_others: Dict[int, List[int]]) -> List[int]:
        b = list(root_belief)
        own_prefix = list(node.action_sequence)
        other_id = next(r for r in planner.robot_ids if r != planner.robot_id)
        other_seq = list(x_others.get(other_id, []))
        for t, my_a in enumerate(own_prefix):
            other_a = other_seq[t] if t < len(other_seq) else WAIT
            joint_a = model.joint_action(my_a, other_a) if planner.robot_id == 0 else model.joint_action(other_a, my_a)
            b = predict_belief_open_loop(b, joint_a, model)
        actions: List[int] = []
        for t in range(len(own_prefix), horizon):
            my_a = heuristic_action_from_belief(
                b,
                agent_id=planner.robot_id,
                exchange_threshold=exchange_threshold,
                wait_if_self_ready_threshold=wait_if_self_ready_threshold,
            )
            actions.append(my_a)
            other_a = other_seq[t] if t < len(other_seq) else WAIT
            joint_a = model.joint_action(my_a, other_a) if planner.robot_id == 0 else model.joint_action(other_a, my_a)
            b = predict_belief_open_loop(b, joint_a, model)
        return actions

    return rollout_policy


# =============================================================================
# Optional execution guard
# =============================================================================

def guard_medevac_action(belief: Sequence[float], agent_id: int, proposed_action: int,
                         exchange_threshold: float, wait_if_self_ready_threshold: float) -> int:
    p_self, _p_other, p_both = marginal_progress_stats(belief, agent_id)
    if proposed_action == EXCHANGE and p_both < exchange_threshold:
        return WAIT if p_self >= wait_if_self_ready_threshold else ADVANCE
    return proposed_action


# =============================================================================
# Planning and episode simulation
# =============================================================================

def root_action_masses(planner: DecMCTS) -> Dict[str, float]:
    masses = {WAIT: 0.0, ADVANCE: 0.0, EXCHANGE: 0.0}
    for seq, p in planner.q.items():
        if len(seq) > 0:
            masses[seq[0]] += p
    return {ACTION_NAME[a]: round(v, 4) for a, v in masses.items()}


def run_planning_step(beliefs: Dict[int, List[float]], remaining_horizon: int, model: MedevacModel,
                      args, seed: int) -> Tuple[Dict[int, int], Dict[int, List[int]], Dict[int, float], Dict[int, Dict[str, float]]]:
    planners: Dict[int, DecMCTS] = {}
    for rid in range(N_AGENTS):
        init_state = MedevacPlanningState(depth=0, max_horizon=remaining_horizon)
        local_fn = make_local_utility_fn(
            agent_id=rid,
            belief=beliefs[rid],
            model=model,
            horizon=remaining_horizon,
            mode=args.local_utility,
        )
        rollout_policy = make_medevac_rollout_policy(
            belief=beliefs[rid],
            model=model,
            horizon=remaining_horizon,
            exchange_threshold=args.exchange_threshold,
            wait_if_self_ready_threshold=args.wait_if_self_ready_threshold,
        )
        planners[rid] = DecMCTS(
            robot_id=rid,
            robot_ids=[0, 1],
            init_state=init_state,
            local_utility_fn=local_fn,
            rollout_policy=rollout_policy,
            default_sequence_fn=lambda _rid, H=remaining_horizon: [WAIT] * H,
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

    actions: Dict[int, int] = {}
    sequences = team.best_sequences()
    for rid in range(N_AGENTS):
        a = planners[rid].best_action()
        raw_a = WAIT if a is None else int(a)
        if getattr(args, "guard_actions", False):
            actions[rid] = guard_medevac_action(
                belief=beliefs[rid],
                agent_id=rid,
                proposed_action=raw_a,
                exchange_threshold=args.exchange_threshold,
                wait_if_self_ready_threshold=args.wait_if_self_ready_threshold,
            )
        else:
            actions[rid] = raw_a

    root_masses = {rid: root_action_masses(planner) for rid, planner in planners.items()}
    return actions, sequences, team.entropies(), root_masses


def belief_support_summary(belief: Sequence[float], max_items: int = 4) -> str:
    items = [(p, s) for s, p in enumerate(belief) if p > 1e-9]
    items.sort(reverse=True)
    top = []
    for p, s in items[:max_items]:
        top.append(f"{id_to_state(s)}:{p:.3f}")
    tail = "" if len(items) <= max_items else f", ... {len(items)} states"
    return "[" + ", ".join(top) + tail + "]"


def simulate_online_episode(model: MedevacModel, args, episode_seed: int, verbose: bool = False) -> float:
    rng = random.Random(episode_seed)
    true_state = model.sample_initial_state(rng)
    common_belief = list(model.init_belief)
    pending_history: List[Tuple[int, int]] = []
    beliefs = {0: list(model.init_belief), 1: list(model.init_belief)}
    total_reward = 0.0

    if verbose:
        print(f"initial true_state={id_to_state(true_state)}")
        print(f"initial beliefs: support={belief_support_summary(beliefs[0])}")

    for t in range(args.horizon):
        remaining = args.horizon - t
        actions, sequences, entropies, root_masses = run_planning_step(
            beliefs=beliefs,
            remaining_horizon=remaining,
            model=model,
            args=args,
            seed=episode_seed + 7919 * t,
        )
        a0, a1 = actions[0], actions[1]
        joint_a = model.joint_action(a0, a1)
        reward = model.reward(true_state, joint_a)
        total_reward += reward
        next_state = model.sample_next_state(true_state, joint_a, rng)
        joint_o = model.obs_from_next_state(next_state, joint_a)
        o0, o1 = model.split_obs(joint_o)
        beliefs[0] = update_local_belief(beliefs[0], joint_a, o0, 0, model)
        beliefs[1] = update_local_belief(beliefs[1], joint_a, o1, 1, model)
        pending_history.append((joint_a, joint_o))

        if args.env_comm_period > 0 and (t + 1) % args.env_comm_period == 0:
            for hist_joint_a, hist_joint_o in pending_history:
                common_belief = update_joint_belief(common_belief, hist_joint_a, hist_joint_o, model)
            beliefs[0] = list(common_belief)
            beliefs[1] = list(common_belief)
            pending_history = []

        if verbose:
            print(
                f"t={t:02d} rem={remaining:02d} "
                f"a=({ACTION_NAME[a0]},{ACTION_NAME[a1]}) "
                f"r={reward:7.3f} "
                f"obs=({OBS_NAME[o0]},{OBS_NAME[o1]}) "
                f"next={id_to_state(next_state)} "
                f"H={{{0}: {entropies.get(0,0):.3f}, {1}: {entropies.get(1,0):.3f}}}"
            )
            print(f"    belief0={belief_support_summary(beliefs[0])}")
            print(f"    belief1={belief_support_summary(beliefs[1])}")
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


def run_episode_worker(payload):
    ep, args_dict = payload
    args = argparse.Namespace(**args_dict)
    model = MedevacModel()
    ret = simulate_online_episode(
        model=model,
        args=args,
        episode_seed=args.seed + 104729 * ep,
        verbose=False,
    )
    return ep, ret


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=50)

    parser.add_argument("--outer-iters", type=int, default=40)
    parser.add_argument("--tau", type=int, default=100)
    parser.add_argument("--num-seq", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--comm-period", type=int, default=1)
    parser.add_argument("--env-comm-period", type=int, default=1)

    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--cp", type=float, default=0.5)
    parser.add_argument("--beta-init", type=float, default=2.0)
    parser.add_argument("--beta-decay", type=float, default=0.995)
    parser.add_argument("--alpha", type=float, default=0.02)

    parser.add_argument("--exchange-threshold", type=float, default=0.80)
    parser.add_argument("--wait-if-self-ready-threshold", type=float, default=0.70)
    parser.add_argument("--local-utility", choices=["difference", "global"], default="difference")
    parser.add_argument("--guard-actions", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose-first", action="store_true")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel worker processes. Use -1 for all CPU cores.")

    args = parser.parse_args()
    model = MedevacModel()

    print("Online Dec-MCTS MEDEVAC benchmark")
    print(f"horizon={args.horizon}, episodes={args.episodes}")
    print(
        f"planning: outer_iters={args.outer_iters}, tau={args.tau}, "
        f"num_seq={args.num_seq}, num_samples={args.num_samples}"
    )
    print(
        f"utility={args.local_utility}, exchange_threshold={args.exchange_threshold}, "
        f"env_comm_period={args.env_comm_period}, seed={args.seed}"
    )
    print(f"guard_actions={args.guard_actions}")
    print()

    returns: List[float] = []
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
            futures = [ex.submit(run_episode_worker, (ep, args_dict)) for ep in range(args.episodes)]
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
    print("This is an online/receding-horizon Dec-MCTS controller baseline.")


if __name__ == "__main__":
    main()
