"""
run_pomdp_benchmarks.py
-----------------------
Evaluates Dec-MCTS on the Tiger Dec-POMDP benchmark, using Mahdi's exact
T/O/R formulation (MOD='C', TRIG_SEMI).

Reference values from the RS-SDA* paper (AAMAS 2026, Al-Husseini et al.):
  h=8:  RS-MAA* 12.217   RS-SDA* 27.215   Cen 47.717
  h=9:  RS-MAA* 15.572   RS-SDA* 30.905   Cen 53.474
  h=10: RS-MAA* 15.184   RS-SDA* 34.724   Cen 60.510

Dec-MCTS should fall BETWEEN RS-MAA* (fully decentralized lower bound) and
RS-SDA* (semi-decentralized upper bound).

Planners:
  DecMCTS (free comm)    — communicate every outer iteration
  DecMCTS-gated p=1      — communicate every iteration only if benchmark allows
  DecMCTS-gated p=5      — communicate every 5 iterations if benchmark allows
  DecMCTS-gated (none)   — no communication (fully decentralized baseline)

Usage:
    python run_pomdp_benchmarks.py [--horizons 8 9 10] [--trials K]
                                   [--dec-outer N] [--dec-tau T] [--seed S]
                                   [--n-particles P] [--n-mc M]
"""

import sys
import os
import time
import random
import argparse

# Allow importing from Scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decmcts import DecMCTS, DecMCTSTeam
from benchmarks.pomdp_tiger import (
    make_tiger_pomdp_benchmark,
    TigerModel, TigerBeliefState,
    sample_obs, update_particles, make_objectives_from_particles,
    _simulate_episode,
    N_PARTICLES, N_MC_SAMPLES, N_AGENTS, N_STATES, LISTEN,
    ACT_PER_AGENT, DEFAULT_HORIZON,
)


# ── DEFAULTS ──────────────────────────────────────────────────────────────────

DEFAULT_HORIZONS   = [3, 9, 10]
DEFAULT_TRIALS     = 5
DEFAULT_SEED       = 42
DEFAULT_DEC_OUTER  = 80     # outer iterations per Dec-MCTS run
DEFAULT_DEC_TAU    = 10     # UCT temperature
DEFAULT_NUM_SEQ    = 10     # sequences sampled per distribution draw
DEFAULT_NUM_SAMPLES = 20    # samples per local_obj_fn call inside DecMCTS
GATED_COMM_PERIODS = [1, 5, -1]

# Reference values from RS-SDA* paper table (AAMAS 2026)
PAPER_REFS = {
    3:  {"rs_maa_star": 12.217, "rs_sda_star": 27.215, "cen": 47.717},
    9:  {"rs_maa_star": 15.572, "rs_sda_star": 30.905, "cen": 53.474},
    10: {"rs_maa_star": 15.184, "rs_sda_star": 34.724, "cen": 60.510},
}


# ── FORMATTING ────────────────────────────────────────────────────────────────

def section(title):
    w = 72
    print(f"\n{'═'*w}")
    print(f"  {title}")
    print(f"{'═'*w}")


def row(label, *vals, width=32):
    vals_str = "  ".join(f"{v:>10}" for v in vals)
    print(f"  {label:<{width}}  {vals_str}")


# ── PLANNER RUNNERS ───────────────────────────────────────────────────────────

def _make_planners(robot_ids, init_states, local_obj_fns, tau,
                   num_seq, num_samples, seed):
    random.seed(seed)
    return {
        rid: DecMCTS(
            robot_id         = rid,
            robot_ids        = robot_ids,
            init_state       = init_states[rid],
            local_utility_fn = local_obj_fns[rid],
            tau              = tau,
            num_seq          = num_seq,
            num_samples      = num_samples,
        )
        for rid in robot_ids
    }


def run_decmcts_free(robot_ids, init_states, global_obj, local_obj_fns,
                     n_outer, tau, num_seq, num_samples, seed):
    """Dec-MCTS with unrestricted communication (communicate every iteration)."""
    planners = _make_planners(robot_ids, init_states, local_obj_fns,
                               tau, num_seq, num_samples, seed)
    t0 = time.perf_counter()
    DecMCTSTeam(planners).iterate_and_communicate(n_outer=n_outer, comm_period=1)
    elapsed = time.perf_counter() - t0

    paths = {
        rid: [init_states[rid].vertex] + planners[rid].best_action_sequence()
        for rid in robot_ids
    }
    return global_obj(paths), elapsed


def run_decmcts_gated(robot_ids, init_states, global_obj, local_obj_fns,
                      can_communicate, n_outer, tau, num_seq, num_samples,
                      comm_period, seed):
    """
    Dec-MCTS with communication gated by BOTH the fixed period AND the
    benchmark's physical comm constraint (can_communicate).

    For Tiger POMDP, can_communicate always returns True, so the gating
    reduces to a pure period gate — useful for ablating communication frequency.

    comm_period=-1 means never communicate.
    """
    planners = _make_planners(robot_ids, init_states, local_obj_fns,
                               tau, num_seq, num_samples, seed)

    def _post_first_action_states():
        result = {}
        for rid in robot_ids:
            seq = planners[rid].best_action_sequence()
            s = init_states[rid]
            if seq:
                s = s.take_action(seq[0])
            result[rid] = s
        return result

    t0 = time.perf_counter()
    for i in range(1, n_outer + 1):
        for p in planners.values():
            p.iterate(1)

        if comm_period > 0 and i % comm_period == 0:
            post_states = _post_first_action_states()
            s0, s1 = post_states[robot_ids[0]], post_states[robot_ids[1]]
            if can_communicate(s0, s1):
                for rid, planner in planners.items():
                    _, q = planner.get_distribution()
                    for other_id, other_planner in planners.items():
                        if other_id != rid:
                            other_planner.receive_dist_dict(rid, q)

    elapsed = time.perf_counter() - t0
    paths = {
        rid: [init_states[rid].vertex] + planners[rid].best_action_sequence()
        for rid in robot_ids
    }
    return global_obj(paths), elapsed


# ── ONLINE DEC-MCTS ───────────────────────────────────────────────────────────

def run_decmcts_online(horizon, n_outer_per_step, tau, num_seq, num_samples,
                       n_particles, n_mc, comm_period, seed):
    """
    Online (receding-horizon) Dec-MCTS for Tiger POMDP.

    At each of the `horizon` decision steps:
      1. Re-plan from the current particle-filter belief (n_outer_per_step
         Dec-MCTS outer iterations, communicating every comm_period iterations).
      2. Execute the first action from each agent's best sequence.
      3. Sample the true joint observation from O(·|true_state, joint_a).
      4. Update each agent's particle filter with the observation.
      5. Transition true state through T.

    Both agents observe the full joint observation (Tiger: same room → shared obs).
    comm_period=-1 disables communication.
    """
    random.seed(seed)
    robot_ids = [0, 1]
    model     = TigerModel()

    # Sample true initial state (unknown to agents)
    r = random.random()
    true_state = 0 if r < model.init_belief[0] else 1

    # Uniform particle filters for both agents
    base = n_particles // N_STATES
    particles = {
        rid: [s for s in range(N_STATES) for _ in range(base)]
        for rid in robot_ids
    }

    total_reward = 0.0

    for step in range(horizon):
        remaining = horizon - step

        # Build belief states and objectives from current particles
        # (agents share particles since they share observations in Tiger)
        current_particles = particles[0]  # both agents have same belief
        global_obj_step, local_obj_fns = make_objectives_from_particles(
            current_particles, model, n_mc
        )

        belief_states = {
            rid: TigerBeliefState(
                list(current_particles), 0, remaining, model, rid
            )
            for rid in robot_ids
        }

        # Build fresh planners at this step
        planners = _make_planners(
            robot_ids, belief_states, local_obj_fns,
            tau, num_seq, num_samples, seed=seed + step * 7
        )

        # Run Dec-MCTS for n_outer_per_step iterations
        for i in range(1, n_outer_per_step + 1):
            for p in planners.values():
                p.iterate(1)
            if comm_period > 0 and i % comm_period == 0:
                for rid, planner in planners.items():
                    _, q = planner.get_distribution()
                    for other_id, other_planner in planners.items():
                        if other_id != rid:
                            other_planner.receive_dist_dict(rid, q)

        # Extract first action from each planner's best sequence
        actions = {}
        for rid in robot_ids:
            seq = planners[rid].best_action_sequence()
            actions[rid] = seq[0] if seq else LISTEN

        joint_a = actions[0] + actions[1] * ACT_PER_AGENT

        # Reward from true state
        total_reward += model.reward(joint_a, true_state)

        # Sample joint observation
        obs = sample_obs(model, joint_a, true_state)

        # Transition true state
        true_state = model.sample_next_state(true_state, joint_a)

        # Update particle filters (both agents share joint observation in Tiger)
        for rid in robot_ids:
            particles[rid] = update_particles(
                particles[rid], joint_a, obs, model, n_resample=n_particles
            )

    return total_reward


# ── SINGLE HORIZON RUN ────────────────────────────────────────────────────────

def run_horizon(horizon, trials, dec_outer, dec_tau, num_seq, num_samples,
                n_particles, n_mc, seed, online_outer=None, skip_offline=False):
    section(f"TIGER POMDP  h={horizon}")

    refs = PAPER_REFS.get(horizon, {})
    if refs:
        print(f"  Paper refs (AAMAS 2026):  "
              f"RS-MAA*={refs['rs_maa_star']:.3f}  "
              f"RS-SDA*={refs['rs_sda_star']:.3f}  "
              f"Cen={refs['cen']:.3f}")
        print(f"  Target range for Dec-MCTS: "
              f"[{refs['rs_maa_star']:.3f}, {refs['rs_sda_star']:.3f}]")
    print(f"\n  Trials={trials}  dec_outer={dec_outer}  dec_tau={dec_tau}  "
          f"n_particles={n_particles}  n_mc={n_mc}\n")

    n_online = online_outer if online_outer is not None else dec_outer // horizon

    free_rewards, free_times = [], []
    gated = {cp: {"rewards": [], "times": []} for cp in GATED_COMM_PERIODS}
    online_rewards, online_times       = [], []
    online_nocomm_rewards, online_nocomm_times = [], []

    print(f"  Running {trials} trials  (online {n_online} iters/step) ...", flush=True)

    for t in range(trials):
        trial_seed = seed + t * 137

        robot_ids, init_states, global_obj, local_obj_fns, can_comm = \
            make_tiger_pomdp_benchmark(
                horizon      = horizon,
                n_particles  = n_particles,
                n_mc         = n_mc,
            )

        # Free communication (offline)
        if not skip_offline:
            r, elapsed = run_decmcts_free(
                robot_ids, init_states, global_obj, local_obj_fns,
                dec_outer, dec_tau, num_seq, num_samples, trial_seed,
            )
            free_rewards.append(r)
            free_times.append(elapsed)

            # Gated variants
            for cp in GATED_COMM_PERIODS:
                r, elapsed = run_decmcts_gated(
                    robot_ids, init_states, global_obj, local_obj_fns,
                    can_comm, dec_outer, dec_tau, num_seq, num_samples,
                    cp, trial_seed,
                )
                gated[cp]["rewards"].append(r)
                gated[cp]["times"].append(elapsed)

        # Online (receding-horizon) — free comm
        t0 = time.perf_counter()
        r = run_decmcts_online(
            horizon, n_online, dec_tau, num_seq, num_samples,
            n_particles, n_mc, comm_period=1, seed=trial_seed,
        )
        online_rewards.append(r)
        online_times.append(time.perf_counter() - t0)

        # Online — no comm
        t0 = time.perf_counter()
        r = run_decmcts_online(
            horizon, n_online, dec_tau, num_seq, num_samples,
            n_particles, n_mc, comm_period=-1, seed=trial_seed,
        )
        online_nocomm_rewards.append(r)
        online_nocomm_times.append(time.perf_counter() - t0)

        offline_str = f"  offline={free_rewards[-1]:>8.2f}" if free_rewards else ""
        print(f"  trial {t+1:>3}/{trials}  "
              f"online_free={online_rewards[-1]:>8.2f}  "
              f"online_nocomm={online_nocomm_rewards[-1]:>8.2f}"
              f"{offline_str}",
              flush=True)

    def med(xs):
        s = sorted(xs)
        return s[len(s) // 2]

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    row("Planner", "Median R", "Avg R", "Avg t(s)")
    print("  " + "─" * 66)

    if free_rewards:
        print("  [OFFLINE — fixed action sequence, evaluated vs initial belief]")
        row("  DecMCTS offline (free comm)",
            f"{med(free_rewards):.3f}",
            f"{avg(free_rewards):.3f}",
            f"{avg(free_times):.2f}s")
        for cp in GATED_COMM_PERIODS:
            rs = gated[cp]["rewards"]
            ts = gated[cp]["times"]
            label = ("  DecMCTS-gated (no comm)"    if cp == -1
                     else f"  DecMCTS-gated (period={cp})")
            row(label,
                f"{med(rs):.3f}",
                f"{avg(rs):.3f}",
                f"{avg(ts):.2f}s")

    print(f"  [ONLINE — replanning every step, {n_online} iters/step, belief updated]")
    row("  DecMCTS online (free comm)",
        f"{med(online_rewards):.3f}",
        f"{avg(online_rewards):.3f}",
        f"{avg(online_times):.2f}s")
    row("  DecMCTS online (no comm)",
        f"{med(online_nocomm_rewards):.3f}",
        f"{avg(online_nocomm_rewards):.3f}",
        f"{avg(online_nocomm_times):.2f}s")

    # Reference rows (paper values)
    if refs:
        print("  " + "─" * 66)
        row("── Paper refs ──", "", "", "")
        row("RS-MAA* (lower bound)",    f"{refs['rs_maa_star']:.3f}", "", "")
        row("RS-SDA* (semi-dec UB)",    f"{refs['rs_sda_star']:.3f}", "", "")
        row("Centralized (paper)",      f"{refs['cen']:.3f}",         "", "")

    # Gap analysis vs online result
    if refs:
        online_med = med(online_rewards)
        lo, hi = refs["rs_maa_star"], refs["rs_sda_star"]
        print(f"\n  Gap analysis (online free-comm median = {online_med:.3f}):")
        if online_med < lo:
            print(f"    BELOW RS-MAA* by {lo - online_med:.3f}  "
                  f"(underperforming fully-decentralized lower bound; "
                  f"try more iters/step)")
        elif online_med > hi:
            print(f"    ABOVE RS-SDA* by {online_med - hi:.3f}  "
                  f"(exceeds semi-dec UB — MC noise or very lucky trial)")
        else:
            pct = 100 * (online_med - lo) / (hi - lo) if hi != lo else 0
            print(f"    Within target range  ({pct:.1f}% of the way from "
                  f"RS-MAA* to RS-SDA*)")

    return {
        "free":          {"rewards": free_rewards,          "times": free_times},
        "gated":         gated,
        "online":        {"rewards": online_rewards,        "times": online_times},
        "online_nocomm": {"rewards": online_nocomm_rewards, "times": online_nocomm_times},
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dec-MCTS on Tiger Dec-POMDP — vs RS-MAA*/RS-SDA* paper refs"
    )
    parser.add_argument("--horizons",    nargs="+", type=int,
                        default=DEFAULT_HORIZONS)
    parser.add_argument("--trials",      type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--dec-outer",   type=int, default=DEFAULT_DEC_OUTER)
    parser.add_argument("--dec-tau",     type=int, default=DEFAULT_DEC_TAU)
    parser.add_argument("--num-seq",     type=int, default=DEFAULT_NUM_SEQ)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--n-particles", type=int, default=N_PARTICLES)
    parser.add_argument("--n-mc",        type=int, default=N_MC_SAMPLES)
    parser.add_argument("--online-outer", type=int, default=None,
                        help="Dec-MCTS iters per step in online mode "
                             "(default: dec-outer // horizon)")
    parser.add_argument("--seed",        type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-offline", action="store_true",
                        help="Skip offline Dec-MCTS variants (saves ~80%% of runtime)")
    parser.add_argument("--output",      type=str, default=None,
                        help="Save full console output to this file (in addition to stdout)")
    args = parser.parse_args()

    # Tee stdout to file if requested
    if args.output:
        import sys as _sys
        class _Tee:
            def __init__(self, *streams): self.streams = streams
            def write(self, s):
                for st in self.streams: st.write(s)
            def flush(self):
                for st in self.streams: st.flush()
        _f = open(args.output, "w")
        _sys.stdout = _Tee(_sys.__stdout__, _f)

    section("Dec-MCTS on Tiger Dec-POMDP (Mahdi's T/O/R formulation)")
    print(f"  Horizons    : {args.horizons}")
    print(f"  Trials      : {args.trials}")
    print(f"  Dec outer   : {args.dec_outer}   tau: {args.dec_tau}")
    print(f"  Online iters/step: {args.online_outer or 'dec-outer // horizon'}")
    print(f"  n_particles : {args.n_particles}   n_mc: {args.n_mc}")
    print(f"  Seed        : {args.seed}")
    print()
    print("  Belief states use a particle filter; reward estimated by MC sampling.")
    print("  Dec-MCTS should land BETWEEN RS-MAA* and RS-SDA* reference values.")

    all_results = {}
    for h in args.horizons:
        all_results[h] = run_horizon(
            horizon      = h,
            trials       = args.trials,
            dec_outer    = args.dec_outer,
            dec_tau      = args.dec_tau,
            num_seq      = args.num_seq,
            num_samples  = args.num_samples,
            n_particles  = args.n_particles,
            n_mc         = args.n_mc,
            seed         = args.seed,
            online_outer = args.online_outer,
            skip_offline = args.skip_offline,
        )

    # ── Cross-horizon summary ─────────────────────────────────────────────────
    section("CROSS-HORIZON SUMMARY")
    print()
    row("h", "Online-free", "Online-nocomm", "Offline-free",
        "RS-MAA*", "RS-SDA*", width=4)
    print("  " + "─" * 80)

    def med(xs):
        s = sorted(xs)
        return s[len(s) // 2]

    for h in args.horizons:
        res  = all_results[h]
        refs = PAPER_REFS.get(h, {})
        lo = f"{refs.get('rs_maa_star', '?'):.3f}" if refs else "?"
        hi = f"{refs.get('rs_sda_star', '?'):.3f}" if refs else "?"
        row(str(h),
            f"{med(res['online']['rewards']):.3f}",
            f"{med(res['online_nocomm']['rewards']):.3f}",
            f"{med(res['free']['rewards']):.3f}" if res['free']['rewards'] else "  skipped",
            lo, hi,
            width=4)

    print()


if __name__ == "__main__":
    main()
