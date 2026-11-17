"""HW6 Part B: FrozenLake, no training, seeded policy comparisons."""
from pathlib import Path
import json
import platform

import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

MAP = ["SFFF", "FHFH", "FFFH", "HFFG"]
ACTIONS = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
# Predefined, state-dependent fixed policy. Never updated during simulations.
FIXED_POLICY = {
    0: 1,   # S: down
    1: 2,   # right
    2: 1,   # down
    3: 0,   # left
    4: 1,   # down
    6: 1,   # down
    8: 2,   # right
    9: 2,   # right
    10: 1,  # down
    13: 2,  # right
    14: 2,  # right (goal)
}

NONTERMINAL = {
    4 * row + col for row in range(4) for col in range(4)
    if MAP[row][col] not in {"G", "H"}
}
assert set(FIXED_POLICY) == NONTERMINAL


def run_episode(slippery: bool, policy: str, episode_seed: int, trace=False):
    """Use the Gymnasium API and respect the 100-step cap."""
    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=slippery,
        render_mode="ansi" if trace else None,
    )
    try:
        state, _ = env.reset(seed=int(episode_seed))
        random_actions = np.random.default_rng(int(episode_seed))
        total_reward = 0.0
        lines = [f"Start: seed={episode_seed}, slippery={slippery}, policy={policy}"]
        if trace:
            lines.append(str(env.render()))
        for steps in range(1, 101):
            action = (
                int(random_actions.integers(0, 4))
                if policy == "random"
                else int(FIXED_POLICY[int(state)])
            )
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            if trace:
                lines.append(
                    f"Step {steps}: state={state}, action={ACTIONS[action]}, "
                    f"next_state={next_state}, reward={reward}, "
                    f"terminated={terminated}, truncated={truncated}\n"
                    f"{env.render()}"
                )
            state = int(next_state)
            # A terminal outcome on step 100 has priority over the time limit.
            if terminated or truncated or steps == 100:
                if state == 15:
                    outcome = "goal"
                elif state in {5, 7, 11, 12}:
                    outcome = "hole"
                else:
                    outcome = "time_limit"
                break
        result = {
            "slippery": bool(slippery),
            "policy": policy,
            "episode_seed": int(episode_seed),
            "steps": int(steps),
            "total_reward": total_reward,
            "outcome": outcome,
            "final_state": int(state),
        }
        if trace:
            lines.append(f"FINAL: {outcome}, steps={steps}, total_reward={total_reward}")
        return result, "\n".join(lines) if trace else ""
    finally:
        env.close()


def main():
    # Written down BEFORE all experiments.
    print("FIXED POLICY (state -> action):")
    for s in sorted(FIXED_POLICY):
        print(f"  {s:2}: {ACTIONS[FIXED_POLICY[s]]}")
    (OUT / "part_b_fixed_policy.json").write_text(
        json.dumps({str(s): ACTIONS[a] for s, a in FIXED_POLICY.items()}, indent=2),
        encoding="utf-8",
    )

    rows = []
    for slippery in [False, True]:
        for policy in ["random", "fixed"]:
            for batch_seed in [11, 22, 33]:
                for i in range(50):
                    episode_seed = 1000 * batch_seed + i
                    result, _ = run_episode(slippery, policy, episode_seed)
                    result.update({
                        "condition": f"slippery={slippery};policy={policy}",
                        "batch_seed": batch_seed,
                    })
                    rows.append(result)

    episodes = pd.DataFrame(rows)
    assert len(episodes) == 600
    episodes.to_csv(OUT / "part_b_episodes.csv", index=False)

    summaries = []
    for (slippery, policy, batch), g in episodes.groupby(
        ["slippery", "policy", "batch_seed"], sort=True
    ):
        goals = int((g.outcome == "goal").sum())
        holes = int((g.outcome == "hole").sum())
        limits = int((g.outcome == "time_limit").sum())
        assert goals + holes + limits == 50
        summaries.append({
            "Slippery?": slippery,
            "Policy": policy,
            "Batch seed": int(batch),
            "Goals /50": goals,
            "Holes /50": holes,
            "Time limits /50": limits,
            "Success %": round(100 * goals / 50, 2),
            "Mean steps": round(float(g.steps.mean()), 2),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "part_b_batch_summary.csv", index=False)
    print("\nRESULTS (all 12 batches):")
    print(summary.to_string(index=False))

    # Replay a successful and an unsuccessful episode from known seeds.
    successes = episodes[episodes.outcome == "goal"]
    failures = episodes[episodes.outcome != "goal"]
    for selected, filename in [
        (successes, "part_b_success_trace.txt"),
        (failures, "part_b_failure_trace.txt"),
    ]:
        if selected.empty:
            print("No episode for", filename)
            continue
        row = selected.iloc[0]
        replay, trace = run_episode(
            bool(row.slippery), str(row.policy), int(row.episode_seed), trace=True
        )
        assert replay["outcome"] == row.outcome
        (OUT / filename).write_text(trace, encoding="utf-8")

    versions = {
        "python": platform.python_version(),
        "gymnasium": gym.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": plt.matplotlib.__version__,
        "install": "python -m pip install gymnasium pandas numpy matplotlib",
        "run": "python part_b.py",
    }
    (OUT / "part_b_versions.json").write_text(
        json.dumps(versions, indent=2), encoding="utf-8"
    )

    plot_data = summary.pivot_table(
        index=["Slippery?", "Policy"], columns="Batch seed", values="Success %"
    )
    ax = plot_data.plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("Environment and policy")
    ax.set_title("FrozenLake success rate by condition and batch")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "part_b_success_rates.png", dpi=170)
    plt.close()

    print("\nSaved files to:", OUT)
    for file in sorted(OUT.glob("part_b_*")):
        print("-", file.name)
    print("\nVersions:", json.dumps(versions, ensure_ascii=False))


if __name__ == "__main__":
    main()
