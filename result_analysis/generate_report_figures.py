#!/usr/bin/env python3
"""
Generate publication-quality figures for the AI Diplomacy report.
Analyses all 4 experiments combined, with per-experiment breakdowns where relevant.

Experiment groups:
  - exp1-3: Auto-compaction for cross-game memory
  - exp4: Dedicated reflection prompt for cross-game memory

Model lineups:
  - exp1-2: Same lineup (gemini-2.5-flash-lite, claude-haiku-4.5, etc.)
  - exp3: Swapped claude-haiku-4.5 -> claude-opus-4.6
  - exp4: Swapped to gpt-5.4, deepseek-v4-pro (no gemini-flash-lite, no gpt-oss-120b)

Usage:
  python result_analysis/generate_report_figures.py
"""

import os
import sys
import json
import re
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "documentation", "images")

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENTS = {
    "exp1": ("3game_experiment", ["game1", "game2", "game3"]),
    "exp2": ("3game_experiment_v2", ["game1", "game2", "game3"]),
    "exp3": ("3game_experiment_v3", ["game1", "game2"]),
    "exp4": ("3game_experiment_v4", ["game1", "game2", "game3"]),
}

# Memory strategy grouping
MEMORY_GROUPS = {
    "Auto-Compaction (Exp 1-3)": ["exp1", "exp2", "exp3"],
    "Reflection Prompt (Exp 4)": ["exp4"],
}

# Styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

SAVE_KWARGS = {"dpi": 150, "bbox_inches": "tight"}


def short_model_name(model_str):
    """Extract short model name from full openrouter path."""
    if not model_str or pd.isna(model_str):
        return "unknown"
    # Remove 'openrouter:' prefix and org prefix
    name = str(model_str).replace("openrouter:", "")
    parts = name.split("/")
    return parts[-1] if parts else name


# ============================================================================
# Data Loading
# ============================================================================

def load_all_data():
    """Load all data from all 4 experiments."""
    print("=" * 60)
    print("LOADING DATA FROM ALL EXPERIMENTS")
    print("=" * 60)

    all_responses = []
    all_deception = []
    all_game_data = {}
    all_summaries = {}
    all_model_maps = []
    all_sc_data = []

    for exp_id, (folder, games) in EXPERIMENTS.items():
        exp_path = os.path.join(RESULTS_DIR, folder)
        print(f"\n--- {exp_id}: {folder} ({len(games)} games) ---")

        # Load experiment summary
        summary_path = os.path.join(exp_path, "experiment_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary = json.load(f)
            all_summaries[exp_id] = summary

            # Extract model maps and SC data
            for game_name, game_info in summary.get("results", {}).items():
                composite_key = f"{exp_id}_{game_name}"

                if "power_model_map" in game_info and game_info["power_model_map"]:
                    for power, model in game_info["power_model_map"].items():
                        all_model_maps.append({
                            "experiment": exp_id,
                            "game": game_name,
                            "composite_game": composite_key,
                            "power": power,
                            "model_full": model,
                            "model": short_model_name(model),
                        })

                if "supply_centers" in game_info:
                    for power, sc_info in game_info["supply_centers"].items():
                        all_sc_data.append({
                            "experiment": exp_id,
                            "game": game_name,
                            "composite_game": composite_key,
                            "power": power,
                            "final_sc": sc_info["count"],
                            "eliminated": sc_info.get("eliminated", False),
                        })

        # Load llm_responses.csv from each game
        for game in games:
            csv_path = os.path.join(exp_path, game, "llm_responses.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df["experiment"] = exp_id
                df["game"] = game
                df["composite_game"] = f"{exp_id}_{game}"
                all_responses.append(df)

        # Load deception scores
        dec_path = os.path.join(exp_path, "deception_scores.csv")
        if os.path.exists(dec_path):
            dec_df = pd.read_csv(dec_path)
            dec_df["experiment"] = exp_id
            all_deception.append(dec_df)

        # Load game JSON data
        for game in games:
            json_path = os.path.join(exp_path, game, "lmvsgame.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    gdata = json.load(f)
                all_game_data[f"{exp_id}_{game}"] = gdata

    # Combine dataframes
    resp_df = pd.concat(all_responses, ignore_index=True) if all_responses else pd.DataFrame()
    dec_df = pd.concat(all_deception, ignore_index=True) if all_deception else pd.DataFrame()
    model_df = pd.DataFrame(all_model_maps)
    sc_df = pd.DataFrame(all_sc_data)

    print(f"\n{'=' * 60}")
    print(f"LOADED TOTALS:")
    print(f"  LLM responses:   {len(resp_df)} rows")
    print(f"  Deception scores: {len(dec_df)} rows")
    print(f"  Game JSONs:       {len(all_game_data)} games")
    print(f"  Model mappings:   {len(model_df)} entries")
    print(f"  SC outcomes:      {len(sc_df)} entries")
    print(f"{'=' * 60}\n")

    return resp_df, dec_df, all_game_data, model_df, sc_df, all_summaries


# ============================================================================
# Trust Analysis
# ============================================================================

def extract_trust_data(resp_df):
    """Extract trust scores from LLM responses."""
    print("Extracting trust scores from LLM responses...")

    # Filter successful responses
    df = resp_df[resp_df["success"] != "FALSE"].copy()

    def extract_trust(raw):
        raw = str(raw)
        match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        raw_json = match.group(1) if match else raw
        try:
            data = json.loads(raw_json)
            if isinstance(data, dict) and "trust_scores" in data and isinstance(data["trust_scores"], dict):
                return data["trust_scores"]
        except:
            pass
        return None

    df["trust_scores"] = df["raw_response"].apply(extract_trust)

    trust_data = []
    for _, row in df.dropna(subset=["trust_scores"]).iterrows():
        for target_power, score in row["trust_scores"].items():
            try:
                trust_data.append({
                    "experiment": row["experiment"],
                    "game": row["game"],
                    "composite_game": row["composite_game"],
                    "phase": row["phase"],
                    "evaluating_power": row["power"],
                    "target_power": target_power.upper(),
                    "score": float(score),
                    "model": short_model_name(row["model"]),
                })
            except:
                pass

    trust_df = pd.DataFrame(trust_data)
    print(f"  Extracted {len(trust_df)} trust observations from {trust_df['composite_game'].nunique()} games")
    return trust_df


def plot_trust_progression(trust_df, model_df):
    """Plot trust progression by model across all experiments."""
    print("Generating: trust_progression_all.png")

    # Merge model info to get model for each evaluating power
    trust_with_model = trust_df.merge(
        model_df[["composite_game", "power", "model"]],
        left_on=["composite_game", "evaluating_power"],
        right_on=["composite_game", "power"],
        how="left",
        suffixes=("_target", "_evaluator")
    )

    # Extract year from phase for grouping
    trust_with_model["year"] = trust_with_model["phase"].str.extract(r"(\d{4})").astype(int)

    # Average trust given per model per year
    avg_by_model_year = trust_with_model.groupby(["model_evaluator", "year"])["score"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=avg_by_model_year, x="year", y="score",
        hue="model_evaluator", marker="o", linewidth=2, ax=ax
    )
    ax.set_title("Trust Given Over Time by Model (All Experiments)")
    ax.set_xlabel("Game Year")
    ax.set_ylabel("Mean Trust Score (0-1)")
    ax.set_ylim(0, 1)
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.savefig(os.path.join(OUTPUT_DIR, "trust_progression_all.png"), **SAVE_KWARGS)
    plt.close()


def plot_trust_heatmap(trust_df, model_df):
    """Plot combined alliance/trust heatmap."""
    print("Generating: trust_heatmap_combined.png")

    # Merge to get model names for both evaluator and target
    trust_with_models = trust_df.merge(
        model_df[["composite_game", "power", "model"]].rename(
            columns={"power": "evaluating_power", "model": "evaluator_model"}
        ),
        on=["composite_game", "evaluating_power"],
        how="left"
    ).merge(
        model_df[["composite_game", "power", "model"]].rename(
            columns={"power": "target_power", "model": "target_model"}
        ),
        on=["composite_game", "target_power"],
        how="left"
    )

    # Average trust given by evaluator_model to target_model
    heatmap_data = trust_with_models.groupby(["evaluator_model", "target_model"])["score"].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index="evaluator_model", columns="target_model", values="score")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        heatmap_pivot, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=0, vmax=1, ax=ax, linewidths=0.5
    )
    ax.set_title("Average Trust Given: Model-to-Model (All Experiments)")
    ax.set_xlabel("Target Model (Trusted)")
    ax.set_ylabel("Evaluating Model (Trusting)")

    plt.savefig(os.path.join(OUTPUT_DIR, "trust_heatmap_combined.png"), **SAVE_KWARGS)
    plt.close()


def plot_trust_asymmetry(trust_df, model_df=None):
    """Plot trust asymmetry example - dynamic shifts between two powers."""
    print("Generating: trust_asymmetry.png")

    # Find a game with interesting asymmetry: look for max divergence between mutual trust
    # Use exp1_game3 (England vs Russia) as in original notebook, or find best example
    best_game = None
    best_divergence = 0

    for composite_game in trust_df["composite_game"].unique():
        game_trust = trust_df[trust_df["composite_game"] == composite_game]
        powers = game_trust["evaluating_power"].unique()
        for i, p1 in enumerate(powers):
            for p2 in powers[i+1:]:
                p1_to_p2 = game_trust[
                    (game_trust["evaluating_power"] == p1) & (game_trust["target_power"] == p2)
                ]["score"]
                p2_to_p1 = game_trust[
                    (game_trust["evaluating_power"] == p2) & (game_trust["target_power"] == p1)
                ]["score"]
                if len(p1_to_p2) > 3 and len(p2_to_p1) > 3:
                    # Measure divergence as max absolute difference
                    min_len = min(len(p1_to_p2), len(p2_to_p1))
                    div = abs(p1_to_p2.values[:min_len] - p2_to_p1.values[:min_len]).max()
                    if div > best_divergence:
                        best_divergence = div
                        best_game = (composite_game, p1, p2)

    if best_game is None:
        print("  WARNING: Could not find suitable asymmetry example")
        return

    composite_game, power_a, power_b = best_game

    # Resolve model names for the two powers
    model_a = power_a
    model_b = power_b
    if model_df is not None:
        match_a = model_df[(model_df["composite_game"] == composite_game) & (model_df["power"] == power_a)]
        match_b = model_df[(model_df["composite_game"] == composite_game) & (model_df["power"] == power_b)]
        if not match_a.empty:
            model_a = match_a.iloc[0]["model"]
        if not match_b.empty:
            model_b = match_b.iloc[0]["model"]

    plot_data = trust_df[
        (trust_df["composite_game"] == composite_game) &
        (
            ((trust_df["evaluating_power"] == power_a) & (trust_df["target_power"] == power_b)) |
            ((trust_df["evaluating_power"] == power_b) & (trust_df["target_power"] == power_a))
        )
    ].copy()

    # Use model names in direction labels
    plot_data["Direction"] = plot_data.apply(
        lambda r: f"{model_a} trusting {model_b}" if r["evaluating_power"] == power_a else f"{model_b} trusting {model_a}",
        axis=1
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    directions = plot_data["Direction"].unique()
    styles = [
        {"color": "0.2", "marker": "o", "linestyle": "-"},
        {"color": "0.6", "marker": "X", "linestyle": "--"},
    ]
    for i, direction in enumerate(directions):
        subset = plot_data[plot_data["Direction"] == direction]
        style = styles[i % len(styles)]
        ax.plot(
            subset["phase"], subset["score"],
            label=direction, linewidth=2.5, markersize=7, **style
        )
    ax.set_title(f"Trust Asymmetry: {model_a} vs {model_b}")
    ax.set_ylabel("Trust Score (0-1)")
    ax.set_xlabel("Phase")
    ax.set_ylim(-0.05, 1.05)
    plt.xticks(rotation=45, ha="right")
    ax.legend(loc="upper right")

    plt.savefig(os.path.join(OUTPUT_DIR, "trust_asymmetry.png"), **SAVE_KWARGS)
    plt.close()
    print(f"  Best asymmetry example: {best_game} (divergence={best_divergence:.2f})")


# ============================================================================
# Deception Analysis
# ============================================================================

def prepare_deception_data(dec_df, all_game_data, model_df):
    """Prepare deception data with proper game/model links."""
    if dec_df.empty:
        print("  WARNING: No deception data available")
        return pd.DataFrame()

    # Build game_id -> composite_game mapping
    id_to_composite = {}
    for composite_key, gdata in all_game_data.items():
        game_id = gdata.get("id")
        if game_id:
            id_to_composite[str(game_id)] = composite_key

    # Map game_id to composite_game
    dec_df = dec_df.copy()
    dec_df["composite_game"] = dec_df["game_id"].astype(str).map(id_to_composite)
    dec_df = dec_df.dropna(subset=["composite_game"])

    # Merge with model info
    dec_df = dec_df.merge(
        model_df[["composite_game", "power", "model"]],
        on=["composite_game", "power"],
        how="left"
    )

    print(f"  Deception data with model info: {len(dec_df)} records, "
          f"{dec_df['composite_game'].nunique()} games")
    return dec_df


def plot_deception_distribution(dec_df):
    """Plot deception score distribution across all experiments."""
    print("Generating: deception_distribution.png")

    if dec_df.empty:
        print("  SKIPPED: No deception data")
        return

    valid = dec_df[dec_df["deception_score"] >= 0].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    axes[0].hist(valid["deception_score"], bins=20, edgecolor="black", color="coral", alpha=0.8)
    axes[0].axvline(valid["deception_score"].mean(), color="red", linestyle="--",
                    label=f"Mean: {valid['deception_score'].mean():.2f}")
    axes[0].axvline(valid["deception_score"].median(), color="darkred", linestyle=":",
                    label=f"Median: {valid['deception_score'].median():.2f}")
    axes[0].set_xlabel("Deception Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Deception Scores (All Experiments)")
    axes[0].legend()

    # By experiment
    exp_order = ["exp1", "exp2", "exp3", "exp4"]
    valid_with_exp = valid[valid["experiment"].isin(exp_order)]
    sns.boxplot(data=valid_with_exp, x="experiment", y="deception_score",
                hue="experiment", order=exp_order, palette="Set2",
                legend=False, ax=axes[1])
    axes[1].set_xlabel("Experiment")
    axes[1].set_ylabel("Deception Score")
    axes[1].set_title("Deception by Experiment")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "deception_distribution.png"), **SAVE_KWARGS)
    plt.close()


def plot_deception_by_model(dec_df):
    """Plot average deception score per model."""
    print("Generating: deception_by_model.png")

    if dec_df.empty:
        print("  SKIPPED: No deception data")
        return

    valid = dec_df[(dec_df["deception_score"] >= 0) & dec_df["model"].notna()].copy()

    model_stats = valid.groupby("model").agg(
        mean_score=("deception_score", "mean"),
        std_score=("deception_score", "std"),
        count=("deception_score", "count")
    ).reset_index().sort_values("mean_score", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        model_stats["model"], model_stats["mean_score"],
        yerr=model_stats["std_score"], capsize=4,
        color=sns.color_palette("viridis", len(model_stats)),
        edgecolor="black", alpha=0.85
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Deception Score")
    ax.set_title("Average Deception Score by Model (All Experiments)")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=35, ha="right")

    # Add count annotations
    for bar, count in zip(bars, model_stats["count"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"n={count}", ha="center", va="bottom", fontsize=9)

    plt.savefig(os.path.join(OUTPUT_DIR, "deception_by_model.png"), **SAVE_KWARGS)
    plt.close()


# ============================================================================
# Error Analysis
# ============================================================================

def classify_response_success(success_str):
    """Classify a response success string into categories."""
    s = str(success_str).strip()
    if s.startswith("Failure:") or s.startswith("FALSE"):
        if "Invalid LLM Moves" in s:
            return "invalid_moves"
        elif "AuthenticationError" in s or "TimeoutError" in s:
            return "api_error"
        elif "No moves extracted" in s:
            return "no_moves"
        elif "Initialized" in s:
            return "initialization_failure"
        else:
            return "other_failure"
    else:
        return "success"


def plot_error_rates(resp_df, model_df):
    """Plot order validity / error rates per model."""
    print("Generating: error_rates_by_model.png")

    resp = resp_df.copy()
    resp["model_short"] = resp["model"].apply(short_model_name)
    resp["outcome"] = resp["success"].apply(classify_response_success)
    resp["is_failure"] = resp["outcome"] != "success"

    # Count by model
    error_stats = resp.groupby("model_short").agg(
        total=("is_failure", "count"),
        failures=("is_failure", "sum"),
    ).reset_index()
    error_stats["error_rate"] = error_stats["failures"] / error_stats["total"]
    error_stats = error_stats.sort_values("error_rate", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: error rate by model
    ax = axes[0]
    colors = sns.color_palette("Reds_r", len(error_stats))
    bars = ax.bar(
        error_stats["model_short"], error_stats["error_rate"],
        color=colors, edgecolor="black", alpha=0.85
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Error Rate")
    ax.set_title("LLM Response Error Rate by Model")
    ax.set_ylim(0, max(0.15, error_stats["error_rate"].max() * 1.3))
    plt.sca(ax)
    plt.xticks(rotation=35, ha="right")
    for bar, (_, row) in zip(bars, error_stats.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{int(row['failures'])}/{int(row['total'])}", ha="center", va="bottom", fontsize=8)

    # Right: error type breakdown
    ax = axes[1]
    failure_types = resp[resp["is_failure"]].groupby(["model_short", "outcome"]).size().unstack(fill_value=0)
    if not failure_types.empty:
        failure_types.plot(kind="bar", stacked=True, ax=ax, colormap="Set2", edgecolor="black")
        ax.set_xlabel("Model")
        ax.set_ylabel("Error Count")
        ax.set_title("Error Type Breakdown by Model")
        ax.legend(title="Error Type", fontsize=8)
        plt.sca(ax)
        plt.xticks(rotation=35, ha="right")
    else:
        ax.text(0.5, 0.5, "No errors detected", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
        ax.set_title("Error Type Breakdown by Model")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "error_rates_by_model.png"), **SAVE_KWARGS)
    plt.close()


# ============================================================================
# Playstyle / Aggression Analysis
# ============================================================================

def classify_order(order_str):
    """Classify an order into tactical categories."""
    o = str(order_str).upper().strip()
    if ' S ' in o:
        return 'support'
    elif ' C ' in o:
        return 'convoy'
    elif o.endswith(' H') or o.endswith(' HOLD'):
        return 'hold'
    elif ' - ' in o:
        return 'attack'
    else:
        return 'other'


def extract_orders(all_game_data):
    """Extract all orders from game data."""
    all_orders = []
    for composite_game, gdata in all_game_data.items():
        parts = composite_game.split("_", 1)
        experiment = parts[0]

        for phase in gdata.get("phases", []):
            phase_name = phase["name"]
            if not phase_name.endswith("M"):
                continue
            year_match = re.search(r"(\d{4})", phase_name)
            if not year_match:
                continue
            year = int(year_match.group(1))

            for power, orders in phase.get("orders", {}).items():
                if orders is None:
                    continue
                for o in orders:
                    all_orders.append({
                        "experiment": experiment,
                        "composite_game": composite_game,
                        "phase": phase_name,
                        "year": year,
                        "power": power,
                        "order": o,
                        "order_type": classify_order(o),
                    })

    return pd.DataFrame(all_orders)


def plot_aggression_by_model(orders_df, model_df):
    """Plot aggression index per model."""
    print("Generating: aggression_by_model.png")

    # Merge with model info
    orders_with_model = orders_df.merge(
        model_df[["composite_game", "power", "model"]],
        on=["composite_game", "power"],
        how="left"
    )

    # Calculate aggression index per model
    # aggression = attacks / (attacks + supports + holds)
    relevant = orders_with_model[orders_with_model["order_type"].isin(["attack", "support", "hold"])]
    model_aggression = relevant.groupby("model").apply(
        lambda g: pd.Series({
            "attacks": (g["order_type"] == "attack").sum(),
            "total": len(g),
            "aggression_index": (g["order_type"] == "attack").sum() / len(g) if len(g) > 0 else 0
        })
    ).reset_index()

    model_aggression = model_aggression.sort_values("aggression_index", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("YlOrRd", len(model_aggression))
    bars = ax.bar(
        model_aggression["model"], model_aggression["aggression_index"],
        color=colors, edgecolor="black", alpha=0.85
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Aggression Index\n(attacks / total tactical orders)")
    ax.set_title("Aggression Index by Model (All Experiments)")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Balanced (0.5)")
    plt.xticks(rotation=35, ha="right")
    ax.legend()

    for bar, n in zip(bars, model_aggression["total"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"n={int(n)}", ha="center", va="bottom", fontsize=9)

    plt.savefig(os.path.join(OUTPUT_DIR, "aggression_by_model.png"), **SAVE_KWARGS)
    plt.close()


# ============================================================================
# Supply Center Trajectories
# ============================================================================

def plot_supply_center_trajectories(all_game_data, model_df):
    """Plot supply center trajectories for representative games."""
    print("Generating: supply_center_trajectories.png")

    # Pick one representative game from each experiment
    representative_games = []
    for exp_id, (folder, games) in EXPERIMENTS.items():
        # Pick game1 from each
        composite = f"{exp_id}_game1"
        if composite in all_game_data:
            representative_games.append((exp_id, composite))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    for idx, (exp_id, composite_game) in enumerate(representative_games[:4]):
        ax = axes_flat[idx]
        gdata = all_game_data[composite_game]

        # Extract SC counts over time
        sc_timeline = defaultdict(list)
        years = []

        for phase in gdata.get("phases", []):
            phase_name = phase["name"]
            if not phase_name.endswith("M"):
                continue
            year_match = re.search(r"(\d{4})", phase_name)
            if not year_match:
                continue
            year = int(year_match.group(1))

            # Only take spring phases to avoid duplicates
            if not phase_name.startswith("S"):
                continue

            state = phase.get("state", {})
            centers = state.get("centers", {})

            if not centers:
                continue

            years.append(year)
            for power, sc_list in centers.items():
                sc_timeline[power].append(len(sc_list))

        if not years:
            continue

        for power, counts in sc_timeline.items():
            # Pad if needed
            while len(counts) < len(years):
                counts.append(counts[-1] if counts else 0)
            ax.plot(years[:len(counts)], counts[:len(counts)], marker=".", label=power, linewidth=1.5)

        ax.set_title(f"{exp_id.upper()} - Game 1")
        ax.set_xlabel("Year")
        ax.set_ylabel("Supply Centers")
        ax.axhline(18, color="red", linestyle="--", alpha=0.3, label="Victory (18)")
        ax.set_ylim(0, 20)
        ax.legend(fontsize=8, ncol=2, loc="upper left")

    plt.suptitle("Supply Center Trajectories (Game 1 from Each Experiment)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "supply_center_trajectories.png"), **SAVE_KWARGS)
    plt.close()


# ============================================================================
# Key Statistics
# ============================================================================

def print_key_statistics(trust_df, dec_df, orders_df, model_df, sc_df, resp_df):
    """Print key statistics for the report."""
    print("\n" + "=" * 60)
    print("KEY STATISTICS FOR REPORT")
    print("=" * 60)

    # Trust statistics
    print("\n--- TRUST ---")
    print(f"  Total trust observations: {len(trust_df)}")
    print(f"  Mean trust score (all): {trust_df['score'].mean():.3f}")
    print(f"  Std trust score: {trust_df['score'].std():.3f}")

    if not trust_df.empty:
        # Trust by experiment
        print("\n  Trust by experiment:")
        for exp in sorted(trust_df["experiment"].unique()):
            exp_trust = trust_df[trust_df["experiment"] == exp]["score"]
            print(f"    {exp}: mean={exp_trust.mean():.3f}, median={exp_trust.median():.3f}, n={len(exp_trust)}")

        # Trust by model (as evaluator)
        trust_with_model = trust_df.merge(
            model_df[["composite_game", "power", "model"]],
            left_on=["composite_game", "evaluating_power"],
            right_on=["composite_game", "power"],
            how="left",
            suffixes=("", "_eval")
        )
        print("\n  Average trust GIVEN by model:")
        model_trust = trust_with_model.groupby("model")["score"].mean().sort_values()
        for model, score in model_trust.items():
            print(f"    {model}: {score:.3f}")

    # Deception statistics
    print("\n--- DECEPTION ---")
    if not dec_df.empty:
        valid = dec_df[dec_df["deception_score"] >= 0]
        print(f"  Total records: {len(valid)}")
        print(f"  Mean deception: {valid['deception_score'].mean():.3f}")
        print(f"  Median deception: {valid['deception_score'].median():.3f}")
        print(f"  Std: {valid['deception_score'].std():.3f}")

        if "model" in valid.columns:
            print("\n  Deception by model:")
            model_dec = valid.groupby("model")["deception_score"].agg(["mean", "count"]).sort_values("mean", ascending=False)
            for model, row in model_dec.iterrows():
                print(f"    {model}: mean={row['mean']:.3f} (n={int(row['count'])})")

        # By memory strategy
        print("\n  Deception by memory strategy:")
        for group_name, exp_list in MEMORY_GROUPS.items():
            group_data = valid[valid["experiment"].isin(exp_list)]
            if not group_data.empty:
                print(f"    {group_name}: mean={group_data['deception_score'].mean():.3f} (n={len(group_data)})")
    else:
        print("  No deception data available")

    # Error/reliability statistics
    print("\n--- ERROR RATES ---")
    resp = resp_df.copy()
    resp["model_short"] = resp["model"].apply(short_model_name)
    resp["outcome"] = resp["success"].apply(classify_response_success)
    resp["is_failure"] = resp["outcome"] != "success"
    total = len(resp)
    failures = resp["is_failure"].sum()
    print(f"  Total LLM calls: {total}")
    print(f"  Failures: {failures} ({100*failures/total:.1f}%)")

    # Error type breakdown
    print("\n  Error type breakdown:")
    type_counts = resp[resp["is_failure"]]["outcome"].value_counts()
    for etype, count in type_counts.items():
        print(f"    {etype}: {count}")

    print("\n  Error rate by model:")
    model_errors = resp.groupby("model_short").agg(
        total=("is_failure", "count"),
        errors=("is_failure", "sum"),
    ).reset_index()
    model_errors["rate"] = model_errors["errors"] / model_errors["total"]
    model_errors = model_errors.sort_values("rate", ascending=False)
    for _, row in model_errors.iterrows():
        print(f"    {row['model_short']}: {row['rate']*100:.1f}% ({int(row['errors'])}/{int(row['total'])})")

    # Aggression statistics
    print("\n--- AGGRESSION ---")
    if not orders_df.empty:
        relevant = orders_df[orders_df["order_type"].isin(["attack", "support", "hold"])]
        orders_with_model = relevant.merge(
            model_df[["composite_game", "power", "model"]],
            on=["composite_game", "power"],
            how="left"
        )
        print(f"  Total tactical orders: {len(relevant)}")
        overall_aggression = (relevant["order_type"] == "attack").sum() / len(relevant)
        print(f"  Overall aggression index: {overall_aggression:.3f}")

        print("\n  Aggression by model:")
        model_agg = orders_with_model.groupby("model").apply(
            lambda g: (g["order_type"] == "attack").sum() / len(g)
        ).sort_values(ascending=False)
        for model, agg in model_agg.items():
            print(f"    {model}: {agg:.3f}")

    # Supply center outcomes
    print("\n--- SUPPLY CENTER OUTCOMES ---")
    print(f"  Total power-game entries: {len(sc_df)}")
    if not sc_df.empty:
        # Best performers by model
        sc_with_model = sc_df.merge(
            model_df[["composite_game", "power", "model"]],
            on=["composite_game", "power"],
            how="left"
        )
        print("\n  Average final SC count by model:")
        model_sc = sc_with_model.groupby("model")["final_sc"].agg(["mean", "count"]).sort_values("mean", ascending=False)
        for model, row in model_sc.iterrows():
            print(f"    {model}: {row['mean']:.1f} SC (n={int(row['count'])})")

        # Elimination rate
        print("\n  Elimination rate by model:")
        elim_rate = sc_with_model.groupby("model")["eliminated"].mean().sort_values(ascending=False)
        for model, rate in elim_rate.items():
            print(f"    {model}: {rate*100:.0f}%")

    # Trust-success correlation
    print("\n--- TRUST vs SUCCESS CORRELATION ---")
    if not trust_df.empty and not sc_df.empty:
        avg_trust_received = trust_df.groupby(["composite_game", "target_power"])["score"].mean().reset_index()
        avg_trust_received.rename(columns={"score": "avg_received_trust", "target_power": "power"}, inplace=True)
        merged = avg_trust_received.merge(sc_df, on=["composite_game", "power"], how="inner")
        if len(merged) > 5:
            corr = merged["avg_received_trust"].corr(merged["final_sc"])
            print(f"  Correlation (received trust vs final SC): {corr:.3f}")
            print(f"  N pairs: {len(merged)}")


# ============================================================================
# Main
# ============================================================================

def main():
    # Load all data
    resp_df, dec_df, all_game_data, model_df, sc_df, all_summaries = load_all_data()

    # Extract trust scores
    trust_df = extract_trust_data(resp_df)

    # Prepare deception data
    dec_prepared = prepare_deception_data(dec_df, all_game_data, model_df)

    # Extract orders
    orders_df = extract_orders(all_game_data)
    print(f"Extracted {len(orders_df)} orders from {orders_df['composite_game'].nunique()} games")

    # Generate figures
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60 + "\n")

    plot_trust_progression(trust_df, model_df)
    plot_trust_heatmap(trust_df, model_df)
    plot_trust_asymmetry(trust_df, model_df)
    plot_deception_distribution(dec_prepared)
    plot_deception_by_model(dec_prepared)
    plot_error_rates(resp_df, model_df)
    plot_aggression_by_model(orders_df, model_df)
    plot_supply_center_trajectories(all_game_data, model_df)

    # Print statistics
    print_key_statistics(trust_df, dec_prepared, orders_df, model_df, sc_df, resp_df)

    print(f"\n{'=' * 60}")
    print(f"ALL FIGURES SAVED TO: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
