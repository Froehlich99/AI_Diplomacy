"""
Cross-Game Memory Effectiveness Analysis

Measures whether AI agents improved across sequential games (1→2→3) within each experiment.
- Games 2+ receive memory from previous games
- Experiments 1-3: auto-compaction memory (last diary summary injected)
- Experiment 4: dedicated end-of-game reflection prompt

Metrics tracked per model across games:
1. Supply centers at game end
2. Survival (eliminated or not)
3. Order validity rate (from llm_responses.csv)
"""

import json
import csv
import os
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- Configuration ---
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = Path(__file__).parent.parent / "documentation" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    "Exp1 (auto-compact)": RESULTS_DIR / "3game_experiment",
    "Exp2 (auto-compact)": RESULTS_DIR / "3game_experiment_v2",
    "Exp3 (auto-compact)": RESULTS_DIR / "3game_experiment_v3",
    "Exp4 (reflection)": RESULTS_DIR / "3game_experiment_v4",
}

# Models in v4 differ from v1-3. We'll use short names for display.
def short_model_name(full_name: str) -> str:
    """Extract a readable short name from the full model path."""
    name = full_name.split("/")[-1]
    # Further shorten common prefixes
    replacements = {
        "grok-4.1-fast": "Grok-4.1",
        "gemma-4-31b-it": "Gemma-4-31B",
        "gemini-2.5-flash-lite": "Gemini-2.5-FL",
        "qwen3.5-27b": "Qwen3.5-27B",
        "qwen3.6-plus": "Qwen3.6+",
        "gpt-oss-120b": "GPT-OSS-120B",
        "claude-haiku-4.5": "Claude-Haiku",
        "gpt-5.4": "GPT-5.4",
        "claude-opus-4.6": "Claude-Opus",
        "deepseek-v4-pro": "DeepSeek-V4",
    }
    return replacements.get(name, name)


def infer_v4_game2_mapping(game1_map: dict, models_list: list) -> dict:
    """Infer game2 power_model_map for v4 experiment from rotation pattern."""
    powers = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
    # game1 uses offset 0, game2 uses offset 1
    game2_map = {}
    for i, power in enumerate(powers):
        model_idx = (i + 1) % 7
        game2_map[power] = models_list[model_idx]
    return game2_map


def load_experiment_data(exp_name: str, exp_path: Path) -> dict:
    """Load experiment summary and compute per-model metrics across games."""
    summary_path = exp_path / "experiment_summary.json"
    if not summary_path.exists():
        print(f"  WARNING: {summary_path} not found, skipping")
        return None

    with open(summary_path) as f:
        summary = json.load(f)

    results = summary["results"]
    game_ids = sorted(results.keys())  # game1, game2, game3

    # Fix missing power_model_map for v4 game2
    if exp_name == "Exp4 (reflection)" and not results.get("game2", {}).get("power_model_map"):
        config_path = exp_path / "experiment_config.json"
        with open(config_path) as f:
            config = json.load(f)
        models_list = config["models"]
        game1_map = results["game1"]["power_model_map"]
        results["game2"]["power_model_map"] = infer_v4_game2_mapping(game1_map, models_list)

    # Collect per-model data across games
    model_data = defaultdict(lambda: {"games": {}})

    for game_id in game_ids:
        game = results[game_id]
        power_model_map = game.get("power_model_map", {})
        supply_centers = game.get("supply_centers", {})

        if not power_model_map:
            print(f"  WARNING: No power_model_map for {game_id} in {exp_name}")
            continue

        # Get order validity from llm_responses.csv
        order_validity = compute_order_validity(exp_path / game_id)

        # Get survival info from game JSON
        survival_info = compute_survival_info(exp_path / game_id)

        for power, model in power_model_map.items():
            model_short = short_model_name(model)
            sc_count = supply_centers.get(power, {}).get("count", 0)
            eliminated = supply_centers.get(power, {}).get("eliminated", False)

            game_num = int(game_id.replace("game", ""))
            model_data[model_short]["games"][game_num] = {
                "supply_centers": sc_count,
                "eliminated": eliminated,
                "power": power,
                "order_validity": order_validity.get(power, None),
                "survival_phases": survival_info.get(power, None),
            }

    return dict(model_data)


def compute_order_validity(game_path: Path) -> dict:
    """Compute order validity rate per power from llm_responses.csv."""
    csv_path = game_path / "llm_responses.csv"
    if not csv_path.exists():
        return {}

    power_orders = defaultdict(lambda: {"total": 0, "valid": 0})

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("response_type") != "order_generation":
                continue
            power = row.get("power", "")
            success = row.get("success", "")
            if not power:
                continue
            power_orders[power]["total"] += 1
            # Valid if success field starts with "Success" or "TRUE"
            if success.startswith("Success") or success == "TRUE":
                power_orders[power]["valid"] += 1

    result = {}
    for power, counts in power_orders.items():
        if counts["total"] > 0:
            result[power] = counts["valid"] / counts["total"]
    return result


def compute_survival_info(game_path: Path) -> dict:
    """Compute survival duration (number of phases survived) per power."""
    json_path = game_path / "lmvsgame.json"
    if not json_path.exists():
        return {}

    with open(json_path) as f:
        data = json.load(f)

    phases = data.get("phases", [])
    total_phases = len(phases)
    powers = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

    # Track when each power first has 0 centers
    elimination_phase = {}
    for power in powers:
        elimination_phase[power] = total_phases  # survived to end by default

    for i, phase in enumerate(phases):
        if "state" not in phase or "centers" not in phase["state"]:
            continue
        centers = phase["state"]["centers"]
        for power in powers:
            if power in centers and len(centers[power]) == 0:
                if power not in elimination_phase or elimination_phase[power] == total_phases:
                    elimination_phase[power] = i

    return elimination_phase


def print_summary_table(all_data: dict):
    """Print a formatted summary table of results."""
    print("\n" + "=" * 100)
    print("CROSS-GAME MEMORY EFFECTIVENESS ANALYSIS")
    print("=" * 100)

    for exp_name, model_data in all_data.items():
        if model_data is None:
            continue
        print(f"\n{'─' * 100}")
        print(f"  {exp_name}")
        print(f"{'─' * 100}")
        print(f"  {'Model':<18} │ {'Game':<6} │ {'Power':<10} │ {'SC':<4} │ {'Elim?':<6} │ {'Order Val%':<10} │ {'Phases':<8}")
        print(f"  {'─'*18}─┼─{'─'*6}─┼─{'─'*10}─┼─{'─'*4}─┼─{'─'*6}─┼─{'─'*10}─┼─{'─'*8}")

        for model in sorted(model_data.keys()):
            games = model_data[model]["games"]
            for game_num in sorted(games.keys()):
                g = games[game_num]
                validity_str = f"{g['order_validity']*100:.0f}%" if g["order_validity"] is not None else "N/A"
                phases_str = f"{g['survival_phases']}" if g["survival_phases"] is not None else "N/A"
                elim_str = "YES" if g["eliminated"] else "no"
                print(f"  {model:<18} │ G{game_num:<5} │ {g['power']:<10} │ {g['supply_centers']:<4} │ {elim_str:<6} │ {validity_str:<10} │ {phases_str:<8}")

    # Print delta analysis
    print("\n\n" + "=" * 100)
    print("IMPROVEMENT ANALYSIS (Delta from Game 1 → Games 2/3)")
    print("=" * 100)

    # Separate by memory type
    for memory_type, exp_names in [
        ("Auto-Compaction Memory (Exp 1-3)", ["Exp1 (auto-compact)", "Exp2 (auto-compact)", "Exp3 (auto-compact)"]),
        ("Reflection Prompt Memory (Exp 4)", ["Exp4 (reflection)"]),
    ]:
        print(f"\n  {memory_type}")
        print(f"  {'─' * 80}")
        print(f"  {'Model':<18} │ {'Experiment':<20} │ {'SC G1→G2':<12} │ {'SC G1→G3':<12} │ {'Val G1→G2':<12} │ {'Val G1→G3':<12}")
        print(f"  {'─'*18}─┼─{'─'*20}─┼─{'─'*12}─┼─{'─'*12}─┼─{'─'*12}─┼─{'─'*12}")

        for exp_name in exp_names:
            model_data = all_data.get(exp_name)
            if model_data is None:
                continue
            for model in sorted(model_data.keys()):
                games = model_data[model]["games"]
                if 1 not in games:
                    continue
                g1_sc = games[1]["supply_centers"]
                g1_val = games[1]["order_validity"]

                # Delta to game 2
                if 2 in games:
                    delta_sc_2 = games[2]["supply_centers"] - g1_sc
                    sc2_str = f"{delta_sc_2:+d}"
                    if g1_val is not None and games[2]["order_validity"] is not None:
                        delta_val_2 = (games[2]["order_validity"] - g1_val) * 100
                        val2_str = f"{delta_val_2:+.1f}%"
                    else:
                        val2_str = "N/A"
                else:
                    sc2_str = "N/A"
                    val2_str = "N/A"

                # Delta to game 3
                if 3 in games:
                    delta_sc_3 = games[3]["supply_centers"] - g1_sc
                    sc3_str = f"{delta_sc_3:+d}"
                    if g1_val is not None and games[3]["order_validity"] is not None:
                        delta_val_3 = (games[3]["order_validity"] - g1_val) * 100
                        val3_str = f"{delta_val_3:+.1f}%"
                    else:
                        val3_str = "N/A"
                else:
                    sc3_str = "—"
                    val3_str = "—"

                print(f"  {model:<18} │ {exp_name:<20} │ {sc2_str:<12} │ {sc3_str:<12} │ {val2_str:<12} │ {val3_str:<12}")

    # Aggregate summary
    print("\n\n" + "=" * 100)
    print("AGGREGATE SUMMARY")
    print("=" * 100)

    for memory_type, exp_names in [
        ("Auto-Compaction (Exp 1-3)", ["Exp1 (auto-compact)", "Exp2 (auto-compact)", "Exp3 (auto-compact)"]),
        ("Reflection Prompt (Exp 4)", ["Exp4 (reflection)"]),
    ]:
        sc_deltas_g2 = []
        sc_deltas_g3 = []
        val_deltas_g2 = []
        val_deltas_g3 = []

        for exp_name in exp_names:
            model_data = all_data.get(exp_name)
            if model_data is None:
                continue
            for model in model_data.values():
                games = model["games"]
                if 1 not in games:
                    continue
                g1_sc = games[1]["supply_centers"]
                g1_val = games[1]["order_validity"]
                if 2 in games:
                    sc_deltas_g2.append(games[2]["supply_centers"] - g1_sc)
                    if g1_val is not None and games[2]["order_validity"] is not None:
                        val_deltas_g2.append(games[2]["order_validity"] - g1_val)
                if 3 in games:
                    sc_deltas_g3.append(games[3]["supply_centers"] - g1_sc)
                    if g1_val is not None and games[3]["order_validity"] is not None:
                        val_deltas_g3.append(games[3]["order_validity"] - g1_val)

        print(f"\n  {memory_type}:")
        if sc_deltas_g2:
            print(f"    SC delta G1→G2: mean={np.mean(sc_deltas_g2):+.2f}, median={np.median(sc_deltas_g2):+.1f} (n={len(sc_deltas_g2)})")
            improved = sum(1 for d in sc_deltas_g2 if d > 0)
            declined = sum(1 for d in sc_deltas_g2 if d < 0)
            same = sum(1 for d in sc_deltas_g2 if d == 0)
            print(f"      Improved: {improved}/{len(sc_deltas_g2)}, Declined: {declined}/{len(sc_deltas_g2)}, Same: {same}/{len(sc_deltas_g2)}")
        if sc_deltas_g3:
            print(f"    SC delta G1→G3: mean={np.mean(sc_deltas_g3):+.2f}, median={np.median(sc_deltas_g3):+.1f} (n={len(sc_deltas_g3)})")
            improved = sum(1 for d in sc_deltas_g3 if d > 0)
            declined = sum(1 for d in sc_deltas_g3 if d < 0)
            same = sum(1 for d in sc_deltas_g3 if d == 0)
            print(f"      Improved: {improved}/{len(sc_deltas_g3)}, Declined: {declined}/{len(sc_deltas_g3)}, Same: {same}/{len(sc_deltas_g3)}")
        if val_deltas_g2:
            print(f"    Order validity delta G1→G2: mean={np.mean(val_deltas_g2)*100:+.1f}%")
        if val_deltas_g3:
            print(f"    Order validity delta G1→G3: mean={np.mean(val_deltas_g3)*100:+.1f}%")


def create_figure(all_data: dict):
    """Create visualization of memory effectiveness across games."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Cross-Game Memory Effectiveness: Did Agents Improve?", fontsize=14, fontweight="bold")

    # Color palette for models
    all_models = set()
    for exp_data in all_data.values():
        if exp_data:
            all_models.update(exp_data.keys())
    all_models = sorted(all_models)
    cmap = plt.colormaps.get_cmap("tab10").resampled(len(all_models))
    model_colors = {m: cmap(i) for i, m in enumerate(all_models)}

    # --- Plot 1: Supply Centers across games (Auto-Compaction) ---
    ax = axes[0, 0]
    ax.set_title("Supply Centers Across Games\n(Auto-Compaction Memory, Exp 1-3)", fontsize=11)
    auto_compact_exps = ["Exp1 (auto-compact)", "Exp2 (auto-compact)", "Exp3 (auto-compact)"]

    for exp_name in auto_compact_exps:
        model_data = all_data.get(exp_name)
        if model_data is None:
            continue
        for model, data in model_data.items():
            games = data["games"]
            game_nums = sorted(games.keys())
            sc_values = [games[g]["supply_centers"] for g in game_nums]
            ax.plot(game_nums, sc_values, marker="o", color=model_colors[model],
                    alpha=0.5, linewidth=1.5, markersize=5)

    ax.set_xlabel("Game Number")
    ax.set_ylabel("Supply Centers at End")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Game 1\n(no memory)", "Game 2\n(with memory)", "Game 3\n(with memory)"])
    ax.axhline(y=34/7, color="gray", linestyle="--", alpha=0.5, label="Fair share (4.9)")
    ax.set_ylim(-0.5, 18)
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Supply Centers (Reflection Prompt) ---
    ax = axes[0, 1]
    ax.set_title("Supply Centers Across Games\n(Reflection Prompt Memory, Exp 4)", fontsize=11)

    model_data = all_data.get("Exp4 (reflection)")
    if model_data:
        for model, data in model_data.items():
            games = data["games"]
            game_nums = sorted(games.keys())
            sc_values = [games[g]["supply_centers"] for g in game_nums]
            ax.plot(game_nums, sc_values, marker="o", color=model_colors[model],
                    alpha=0.7, linewidth=2, markersize=7, label=model)

    ax.set_xlabel("Game Number")
    ax.set_ylabel("Supply Centers at End")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Game 1\n(no memory)", "Game 2\n(with memory)", "Game 3\n(with memory)"])
    ax.axhline(y=34/7, color="gray", linestyle="--", alpha=0.5, label="Fair share (4.9)")
    ax.set_ylim(-0.5, 18)
    ax.legend(fontsize=7, loc="upper left", ncol=1)
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Aggregate SC Deltas (bar chart) ---
    ax = axes[1, 0]
    ax.set_title("Average SC Change from Game 1\n(by model, all experiments)", fontsize=11)

    # Aggregate per model across auto-compaction experiments
    model_deltas_auto = defaultdict(list)
    model_deltas_refl = defaultdict(list)

    for exp_name in auto_compact_exps:
        model_data = all_data.get(exp_name)
        if model_data is None:
            continue
        for model, data in model_data.items():
            games = data["games"]
            if 1 in games:
                g1_sc = games[1]["supply_centers"]
                for gn in [2, 3]:
                    if gn in games:
                        model_deltas_auto[model].append(games[gn]["supply_centers"] - g1_sc)

    exp4_data = all_data.get("Exp4 (reflection)")
    if exp4_data:
        for model, data in exp4_data.items():
            games = data["games"]
            if 1 in games:
                g1_sc = games[1]["supply_centers"]
                for gn in [2, 3]:
                    if gn in games:
                        model_deltas_refl[model].append(games[gn]["supply_centers"] - g1_sc)

    # Combine and plot
    all_models_in_data = sorted(set(list(model_deltas_auto.keys()) + list(model_deltas_refl.keys())))
    x = np.arange(len(all_models_in_data))
    width = 0.35

    auto_means = [np.mean(model_deltas_auto.get(m, [0])) if m in model_deltas_auto else None for m in all_models_in_data]
    refl_means = [np.mean(model_deltas_refl.get(m, [0])) if m in model_deltas_refl else None for m in all_models_in_data]

    bars1 = []
    bars2 = []
    for i, m in enumerate(all_models_in_data):
        if auto_means[i] is not None:
            b = ax.bar(x[i] - width/2, auto_means[i], width, color=model_colors[m], alpha=0.6, edgecolor="black", linewidth=0.5)
            bars1.append(b)
        if refl_means[i] is not None:
            b = ax.bar(x[i] + width/2, refl_means[i], width, color=model_colors[m], alpha=1.0, edgecolor="black", linewidth=0.5, hatch="//")
            bars2.append(b)

    ax.set_xticks(x)
    ax.set_xticklabels(all_models_in_data, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Avg SC Delta (from Game 1)")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")

    # Legend for bar types
    auto_patch = mpatches.Patch(facecolor="gray", alpha=0.6, edgecolor="black", label="Auto-compact (Exp 1-3)")
    refl_patch = mpatches.Patch(facecolor="gray", alpha=1.0, edgecolor="black", hatch="//", label="Reflection (Exp 4)")
    ax.legend(handles=[auto_patch, refl_patch], fontsize=8, loc="upper right")

    # --- Plot 4: Order Validity across games ---
    ax = axes[1, 1]
    ax.set_title("Order Validity Rate Across Games\n(All Experiments)", fontsize=11)

    for exp_name, model_data in all_data.items():
        if model_data is None:
            continue
        linestyle = "-" if "auto" in exp_name else "--"
        for model, data in model_data.items():
            games = data["games"]
            game_nums = sorted(games.keys())
            val_values = []
            valid_nums = []
            for g in game_nums:
                v = games[g].get("order_validity")
                if v is not None:
                    val_values.append(v * 100)
                    valid_nums.append(g)
            if val_values:
                ax.plot(valid_nums, val_values, marker="s", color=model_colors[model],
                        alpha=0.5, linewidth=1.5, markersize=4, linestyle=linestyle)

    ax.set_xlabel("Game Number")
    ax.set_ylabel("Order Validity Rate (%)")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Game 1\n(no memory)", "Game 2\n(with memory)", "Game 3\n(with memory)"])
    ax.set_ylim(50, 105)
    ax.grid(True, alpha=0.3)

    # Add legend distinguishing experiment types
    solid_line = plt.Line2D([0], [0], color="gray", linewidth=2, linestyle="-", label="Auto-compact (Exp 1-3)")
    dashed_line = plt.Line2D([0], [0], color="gray", linewidth=2, linestyle="--", label="Reflection (Exp 4)")
    ax.legend(handles=[solid_line, dashed_line], fontsize=8, loc="lower right")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "memory_effectiveness.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved to: {output_path}")


def main():
    print("Loading experiment data...")
    all_data = {}
    for exp_name, exp_path in EXPERIMENTS.items():
        print(f"\n  Processing: {exp_name} ({exp_path.name})")
        all_data[exp_name] = load_experiment_data(exp_name, exp_path)

    print_summary_table(all_data)
    create_figure(all_data)


if __name__ == "__main__":
    main()
