#!/usr/bin/env python3
"""
Evaluation and visualization script for AI Diplomacy experiments.

Generates publication-quality plots and metrics from experiment results produced
by experiment_runner.py and (optionally) deception_analyzer.py.

Usage:
    python evaluate.py results/my_experiment/
    python evaluate.py results/my_experiment/ --output_dir results/my_experiment/evaluation
    python evaluate.py results/single_game/            # single game directory
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

# Silence matplotlib / seaborn style warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Academic style -- fall back gracefully if the exact name doesn't exist
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass  # keep default

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

ALL_POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

POWER_COLORS = {
    "AUSTRIA": "#C41E3A",
    "ENGLAND": "#003399",
    "FRANCE": "#0055A4",
    "GERMANY": "#555555",
    "ITALY": "#009246",
    "RUSSIA": "#7B2D8E",
    "TURKEY": "#E6A817",
}

RELATIONSHIP_ORDER = {"Enemy": 0, "Unfriendly": 1, "Neutral": 2, "Friendly": 3, "Ally": 4}

_PHASE_RE_STR = r"^[SWF](\d{4})[MRA]$"
import re
_PHASE_RE = re.compile(_PHASE_RE_STR)


def _phase_year(phase_name: str) -> Optional[int]:
    m = _PHASE_RE.match(phase_name)
    return int(m.group(1)) if m else None


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_experiment(exp_dir: str) -> dict:
    """Load experiment_summary.json, experiment_config.json, and discover game dirs."""
    result: dict = {
        "exp_dir": exp_dir,
        "summary": None,
        "config": None,
        "game_dirs": [],
    }

    summary_path = os.path.join(exp_dir, "experiment_summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path, "r") as f:
            result["summary"] = json.load(f)
        logger.info("Loaded experiment_summary.json with %d results.",
                     len(result["summary"].get("results", {})))

    config_path = os.path.join(exp_dir, "experiment_config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            result["config"] = json.load(f)
        logger.info("Loaded experiment_config.json.")

    # Discover game directories (those containing lmvsgame.json)
    if os.path.isfile(os.path.join(exp_dir, "lmvsgame.json")):
        # Input is a single game directory
        result["game_dirs"] = [exp_dir]
    else:
        for entry in sorted(os.listdir(exp_dir)):
            subdir = os.path.join(exp_dir, entry)
            if os.path.isdir(subdir) and os.path.isfile(os.path.join(subdir, "lmvsgame.json")):
                result["game_dirs"].append(subdir)

    logger.info("Found %d game directories.", len(result["game_dirs"]))
    return result


def load_game_phases(game_dir: str) -> list[dict]:
    """Load lmvsgame.json and return the phases list."""
    path = os.path.join(game_dir, "lmvsgame.json")
    if not os.path.isfile(path):
        logger.warning("lmvsgame.json not found in %s", game_dir)
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("phases", [])


def extract_sc_trajectories(phases: list[dict]) -> pd.DataFrame:
    """Extract supply-centre counts per power per phase."""
    rows: list[dict] = []
    for idx, phase in enumerate(phases):
        name = phase.get("name", "")
        year = _phase_year(name)
        centers = phase.get("state", {}).get("centers", {})
        for power in ALL_POWERS:
            sc_list = centers.get(power, [])
            rows.append({
                "phase": name,
                "phase_idx": idx,
                "year": year,
                "power": power,
                "sc_count": len(sc_list) if isinstance(sc_list, list) else 0,
            })
    return pd.DataFrame(rows)


def extract_trust_trajectories(phases: list[dict]) -> pd.DataFrame:
    """Extract trust scores per power-pair per phase."""
    rows: list[dict] = []
    for idx, phase in enumerate(phases):
        name = phase.get("name", "")
        state_agents = phase.get("state_agents")
        if not state_agents:
            continue
        for observer, agent_data in state_agents.items():
            ts = agent_data.get("trust_scores", {})
            for target, score in ts.items():
                if isinstance(score, (int, float)):
                    rows.append({
                        "phase": name,
                        "phase_idx": idx,
                        "observer": observer,
                        "target": target,
                        "trust_score": float(score),
                    })
    return pd.DataFrame(rows)


def extract_relationship_trajectories(phases: list[dict]) -> pd.DataFrame:
    """Extract relationship labels per power-pair per phase."""
    rows: list[dict] = []
    for idx, phase in enumerate(phases):
        name = phase.get("name", "")
        state_agents = phase.get("state_agents")
        if not state_agents:
            continue
        for observer, agent_data in state_agents.items():
            rels = agent_data.get("relationships", {})
            for target, rel in rels.items():
                rows.append({
                    "phase": name,
                    "phase_idx": idx,
                    "observer": observer,
                    "target": target,
                    "relationship": rel,
                })
    return pd.DataFrame(rows)


def extract_order_success(phases: list[dict]) -> pd.DataFrame:
    """Extract order success rates from order_results."""
    rows: list[dict] = []
    for phase in phases:
        name = phase.get("name", "")
        order_results = phase.get("order_results", {})
        if not order_results:
            continue
        for power, type_map in order_results.items():
            total = 0
            successful = 0
            failed = 0
            if not isinstance(type_map, dict):
                continue
            for _otype, order_list in type_map.items():
                if not isinstance(order_list, list):
                    continue
                for entry in order_list:
                    total += 1
                    res = entry.get("result", "").lower() if isinstance(entry, dict) else ""
                    if res in ("success", "succeeds", ""):
                        successful += 1
                    else:
                        failed += 1
            rows.append({
                "phase": name,
                "power": power,
                "total_orders": total,
                "successful": successful,
                "failed": failed,
            })
    return pd.DataFrame(rows)


def load_deception_data(exp_dir: str) -> Optional[pd.DataFrame]:
    """Load all deception_scores.csv files found in subdirectories."""
    csv_files: list[str] = []
    for root, _dirs, files in os.walk(exp_dir):
        for fname in files:
            if fname == "deception_scores.csv":
                csv_files.append(os.path.join(root, fname))

    if not csv_files:
        return None

    frames: list[pd.DataFrame] = []
    for path in csv_files:
        try:
            df = pd.read_csv(path)
            frames.append(df)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", path, exc)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Loaded deception data: %d rows from %d files.", len(combined), len(frames))
    return combined


# ── Metric Computation ───────────────────────────────────────────────────────


def _classify_games(config: Optional[dict]) -> dict[str, str]:
    """Return {game_id: 'memory' | 'baseline'} from experiment config."""
    mapping: dict[str, str] = {}
    if config is None:
        return mapping
    for game_spec in config.get("games", []):
        gid = game_spec.get("id", "")
        mapping[gid] = "memory" if game_spec.get("memory", False) else "baseline"
    return mapping


def _final_sc_for_game(game_result: dict) -> dict[str, int]:
    """Extract {power: sc_count} from a single game result."""
    sc = game_result.get("supply_centers", {})
    return {p: info.get("count", 0) if isinstance(info, dict) else 0 for p, info in sc.items()}


def compute_improvement_delta(exp_summary: dict, config: dict) -> dict:
    """Compare average final SC counts between memory and baseline game groups."""
    classification = _classify_games(config)
    results = exp_summary.get("results", {})

    memory_totals: list[float] = []
    baseline_totals: list[float] = []

    for gid, result in results.items():
        if "error" in result:
            continue
        final_sc = _final_sc_for_game(result)
        mean_sc = np.mean(list(final_sc.values())) if final_sc else 0.0
        game_type = classification.get(gid, "unknown")
        if game_type == "memory":
            memory_totals.append(mean_sc)
        elif game_type == "baseline":
            baseline_totals.append(mean_sc)

    mean_mem = float(np.mean(memory_totals)) if memory_totals else 0.0
    mean_base = float(np.mean(baseline_totals)) if baseline_totals else 0.0

    return {
        "mean_sc_memory": round(mean_mem, 3),
        "mean_sc_baseline": round(mean_base, 3),
        "delta": round(mean_mem - mean_base, 3),
    }


def compute_memory_success_rates(exp_summary: dict, config: dict) -> dict:
    """Win/survival rates for memory vs baseline games."""
    classification = _classify_games(config)
    results = exp_summary.get("results", {})

    counts = {"memory": {"wins": 0, "survivals": 0, "total_powers": 0, "games": 0},
              "baseline": {"wins": 0, "survivals": 0, "total_powers": 0, "games": 0}}

    for gid, result in results.items():
        if "error" in result:
            continue
        game_type = classification.get(gid, "unknown")
        if game_type not in counts:
            continue
        counts[game_type]["games"] += 1
        if result.get("winner"):
            counts[game_type]["wins"] += 1
        sc = result.get("supply_centers", {})
        for _p, info in sc.items():
            counts[game_type]["total_powers"] += 1
            if isinstance(info, dict) and not info.get("eliminated", False):
                counts[game_type]["survivals"] += 1

    def _rate(numer: int, denom: int) -> float:
        return round(numer / denom, 3) if denom > 0 else 0.0

    return {
        "memory_win_rate": _rate(counts["memory"]["wins"], counts["memory"]["games"]),
        "baseline_win_rate": _rate(counts["baseline"]["wins"], counts["baseline"]["games"]),
        "memory_survival_rate": _rate(counts["memory"]["survivals"], counts["memory"]["total_powers"]),
        "baseline_survival_rate": _rate(counts["baseline"]["survivals"], counts["baseline"]["total_powers"]),
    }


# ── Plot Functions ───────────────────────────────────────────────────────────


def _save_fig(fig: plt.Figure, path_stem: str) -> None:
    """Save figure as both PNG and PDF."""
    for ext in (".png", ".pdf"):
        fig.savefig(path_stem + ext, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s.{png,pdf}", path_stem)


def plot_sc_trajectories(sc_df: pd.DataFrame, output_dir: str, game_ids: list[str]) -> None:
    """Fig 1: SC trajectory per game, one subplot per game."""
    if sc_df.empty:
        logger.warning("No SC data to plot.")
        return

    n_games = len(game_ids)
    if n_games == 0:
        return

    ncols = min(n_games, 3)
    nrows = (n_games + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for i, gid in enumerate(game_ids):
        ax = axes[i // ncols][i % ncols]
        game_sc = sc_df[sc_df["game_id"] == gid] if "game_id" in sc_df.columns else sc_df
        for power in ALL_POWERS:
            pdata = game_sc[game_sc["power"] == power].sort_values("phase_idx")
            if pdata.empty:
                continue
            ax.plot(pdata["phase_idx"], pdata["sc_count"],
                    label=power, color=POWER_COLORS.get(power, None), linewidth=1.5)
        ax.set_title(gid, fontsize=10)
        ax.set_xlabel("Phase index")
        ax.set_ylabel("Supply centres")
        ax.legend(fontsize=6, ncol=2)

    # Hide unused subplots
    for j in range(n_games, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("Supply Centre Trajectories", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, os.path.join(output_dir, "fig1_sc_trajectories"))


def plot_memory_vs_baseline(exp_summary: dict, config: dict, output_dir: str) -> None:
    """Fig 2: Grouped bar chart -- memory vs baseline mean final SC."""
    classification = _classify_games(config)
    results = exp_summary.get("results", {})

    groups: dict[str, list[float]] = {"memory": [], "baseline": []}
    for gid, result in results.items():
        if "error" in result:
            continue
        game_type = classification.get(gid, "unknown")
        if game_type not in groups:
            continue
        final_sc = _final_sc_for_game(result)
        # Use the max SC (winner's perspective)
        groups[game_type].append(max(final_sc.values()) if final_sc else 0)

    if not any(groups.values()):
        logger.warning("Not enough data for memory vs baseline plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(groups.keys())
    means = [float(np.mean(groups[k])) if groups[k] else 0 for k in labels]
    stds = [float(np.std(groups[k])) if groups[k] else 0 for k in labels]

    bars = ax.bar(labels, means, yerr=stds, capsize=5,
                  color=["#4C72B0", "#DD8452"], edgecolor="black", linewidth=0.8)

    delta = compute_improvement_delta(exp_summary, config)
    ax.annotate(f"Δ = {delta['delta']:+.2f}",
                xy=(0.5, max(means) * 1.05), fontsize=11, ha="center",
                fontweight="bold", color="#333333")

    ax.set_ylabel("Mean max final SC")
    ax.set_title("Memory vs Baseline Performance")
    fig.tight_layout()
    _save_fig(fig, os.path.join(output_dir, "fig2_memory_vs_baseline"))


def plot_win_survival_heatmap(exp_summary: dict, output_dir: str) -> None:
    """Fig 3: Heatmap -- rows=models, columns=powers, cells=final SC count."""
    results = exp_summary.get("results", {})

    # Collect (model, power) -> list of SC counts
    model_power_sc: dict[str, dict[str, list[int]]] = {}
    for gid, result in results.items():
        if "error" in result:
            continue
        pmap = result.get("power_model_map", {})
        sc_info = result.get("supply_centers", {})
        for power, model in pmap.items():
            sc_count = sc_info.get(power, {}).get("count", 0) if isinstance(sc_info.get(power), dict) else 0
            model_power_sc.setdefault(model, {}).setdefault(power, []).append(sc_count)

    if not model_power_sc:
        logger.warning("No model-power data for heatmap.")
        return

    models = sorted(model_power_sc.keys())
    powers = [p for p in ALL_POWERS if any(p in model_power_sc.get(m, {}) for m in models)]

    data = np.zeros((len(models), len(powers)))
    for mi, model in enumerate(models):
        for pi, power in enumerate(powers):
            vals = model_power_sc.get(model, {}).get(power, [])
            data[mi, pi] = np.mean(vals) if vals else 0

    fig, ax = plt.subplots(figsize=(max(8, len(powers) * 1.2), max(3, len(models) * 0.8)))
    # Shorten long model names for display
    display_models = [m.split("/")[-1] if "/" in m else m for m in models]
    sns.heatmap(data, annot=True, fmt=".1f", xticklabels=powers, yticklabels=display_models,
                cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("Mean Final SC by Model × Power Position")
    ax.set_xlabel("Power")
    ax.set_ylabel("Model")
    fig.tight_layout()
    _save_fig(fig, os.path.join(output_dir, "fig3_win_survival_heatmap"))


def plot_trust_evolution(trust_df: pd.DataFrame, output_dir: str, game_id: str) -> None:
    """Fig 4: Mean trust given by each power over time."""
    if trust_df.empty:
        logger.warning("No trust data to plot for %s.", game_id)
        return

    # Aggregate: mean trust given by each observer per phase
    agg = trust_df.groupby(["phase_idx", "observer"])["trust_score"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    for power in ALL_POWERS:
        pdata = agg[agg["observer"] == power].sort_values("phase_idx")
        if pdata.empty:
            continue
        ax.plot(pdata["phase_idx"], pdata["trust_score"],
                label=power, color=POWER_COLORS.get(power), linewidth=1.5)

    ax.set_xlabel("Phase index")
    ax.set_ylabel("Mean trust score (given)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(f"Trust Evolution — {game_id}")
    fig.tight_layout()
    _save_fig(fig, os.path.join(output_dir, f"fig4_trust_evolution_{game_id}"))


def plot_trust_matrix(trust_df: pd.DataFrame, output_dir: str, game_id: str) -> None:
    """Fig 5: 7x7 heatmap of final-phase trust scores."""
    if trust_df.empty:
        logger.warning("No trust data for matrix in %s.", game_id)
        return

    max_idx = trust_df["phase_idx"].max()
    final = trust_df[trust_df["phase_idx"] == max_idx]
    if final.empty:
        return

    matrix = pd.pivot_table(final, values="trust_score",
                            index="observer", columns="target", aggfunc="mean")

    # Reindex to consistent ALL_POWERS order
    present = [p for p in ALL_POWERS if p in matrix.index or p in matrix.columns]
    matrix = matrix.reindex(index=present, columns=present)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdYlGn", center=0.5,
                vmin=0, vmax=1, linewidths=0.5, ax=ax, square=True)
    ax.set_title(f"Final Trust Matrix — {game_id}")
    fig.tight_layout()
    _save_fig(fig, os.path.join(output_dir, f"fig5_trust_matrix_{game_id}"))


def plot_deception_distribution(deception_df: pd.DataFrame, config: Optional[dict],
                                output_dir: str) -> None:
    """Fig 6: Box plot of deception scores, grouped by memory/baseline if available."""
    if deception_df.empty:
        logger.warning("No deception data for distribution plot.")
        return

    classification = _classify_games(config) if config else {}

    df = deception_df.copy()
    # Filter out invalid scores
    df = df[df["deception_score"].apply(lambda x: isinstance(x, (int, float)) and x >= 0)]
    if df.empty:
        return

    if classification:
        df["game_type"] = df["game_id"].map(classification).fillna("unknown")
        hue_col = "game_type" if df["game_type"].nunique() > 1 else None
    else:
        hue_col = None

    fig, ax = plt.subplots(figsize=(7, 5))
    if hue_col:
        sns.boxplot(data=df, x="game_type", y="deception_score", ax=ax, palette="Set2")
        ax.set_xlabel("Game type")
    else:
        sns.boxplot(data=df, y="deception_score", ax=ax, color="#4C72B0")
        ax.set_xlabel("")
    ax.set_ylabel("Deception score")
    ax.set_title("Deception Score Distribution")
    fig.tight_layout()
    _save_fig(fig, os.path.join(output_dir, "fig6_deception_distribution"))


def plot_deception_vs_sc(deception_df: pd.DataFrame, all_sc_df: pd.DataFrame,
                         output_dir: str) -> None:
    """Fig 7: Scatter -- mean deception vs final SC per power per game."""
    if deception_df.empty or all_sc_df.empty:
        logger.warning("Insufficient data for deception vs SC plot.")
        return

    # Filter valid deception scores
    dec = deception_df[deception_df["deception_score"].apply(
        lambda x: isinstance(x, (int, float)) and x >= 0)].copy()
    if dec.empty:
        return

    dec_agg = dec.groupby(["game_id", "power"])["deception_score"].mean().reset_index()

    # Get final SC per power per game
    if "game_id" in all_sc_df.columns:
        final_idx = all_sc_df.groupby(["game_id", "power"])["phase_idx"].max().reset_index()
        final_sc = final_idx.merge(all_sc_df, on=["game_id", "power", "phase_idx"], how="left")
    else:
        final_sc = all_sc_df.loc[all_sc_df.groupby("power")["phase_idx"].idxmax()]
        final_sc = final_sc[["power", "sc_count"]].copy()
        final_sc["game_id"] = "single"

    merged = dec_agg.merge(final_sc[["game_id", "power", "sc_count"]],
                           on=["game_id", "power"], how="inner")
    if merged.empty:
        logger.warning("No matching data for deception-vs-SC scatter.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for power in ALL_POWERS:
        sub = merged[merged["power"] == power]
        if sub.empty:
            continue
        ax.scatter(sub["deception_score"], sub["sc_count"],
                   c=POWER_COLORS.get(power), label=power, s=60, alpha=0.8, edgecolors="black", linewidth=0.4)

    # Regression line
    if len(merged) > 2:
        z = np.polyfit(merged["deception_score"], merged["sc_count"], 1)
        xline = np.linspace(merged["deception_score"].min(), merged["deception_score"].max(), 50)
        ax.plot(xline, np.polyval(z, xline), "--", color="gray", linewidth=1.5, label="OLS fit")

    ax.set_xlabel("Mean deception score")
    ax.set_ylabel("Final supply centres")
    ax.set_title("Deception vs Supply Centre Count")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _save_fig(fig, os.path.join(output_dir, "fig7_deception_vs_sc"))


def plot_relationship_evolution(rel_df: pd.DataFrame, output_dir: str, game_id: str) -> None:
    """Fig 8: Stacked area chart of relationship type proportions over time."""
    if rel_df.empty:
        logger.warning("No relationship data for %s.", game_id)
        return

    # Map relationship labels to numeric for ordering
    df = rel_df.copy()
    df["rel_val"] = df["relationship"].map(RELATIONSHIP_ORDER)
    df = df.dropna(subset=["rel_val"])
    if df.empty:
        return

    # Count proportions per phase per relationship type
    phase_counts = df.groupby(["phase_idx", "relationship"]).size().unstack(fill_value=0)
    # Reindex columns to consistent order
    rel_labels = [r for r in RELATIONSHIP_ORDER.keys() if r in phase_counts.columns]
    phase_counts = phase_counts.reindex(columns=rel_labels, fill_value=0)
    # Normalise to proportions
    phase_totals = phase_counts.sum(axis=1)
    phase_props = phase_counts.div(phase_totals, axis=0)

    rel_colors = {
        "Enemy": "#d62728",
        "Unfriendly": "#ff7f0e",
        "Neutral": "#aec7e8",
        "Friendly": "#2ca02c",
        "Ally": "#1f77b4",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(phase_props))
    x = phase_props.index.values

    for rel in rel_labels:
        vals = phase_props[rel].values
        ax.fill_between(x, bottom, bottom + vals,
                        label=rel, color=rel_colors.get(rel, "#999999"), alpha=0.85)
        bottom = bottom + vals

    ax.set_xlabel("Phase index")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Relationship Evolution — {game_id}")
    fig.tight_layout()
    _save_fig(fig, os.path.join(output_dir, f"fig8_relationship_evolution_{game_id}"))


# ── Metrics Summary ──────────────────────────────────────────────────────────


def build_metrics_summary(
    exp_summary: Optional[dict],
    config: Optional[dict],
    all_order_success: dict[str, pd.DataFrame],
    deception_df: Optional[pd.DataFrame],
) -> dict:
    """Assemble the metrics_summary.json dict."""
    metrics: dict = {
        "improvement_delta": {},
        "memory_success_rate": {},
        "per_game": {},
    }

    if exp_summary and config:
        metrics["improvement_delta"] = compute_improvement_delta(exp_summary, config)
        metrics["memory_success_rate"] = compute_memory_success_rates(exp_summary, config)

    results = exp_summary.get("results", {}) if exp_summary else {}
    for gid, result in results.items():
        if "error" in result:
            metrics["per_game"][gid] = {"error": result["error"]}
            continue

        final_sc = _final_sc_for_game(result)
        order_df = all_order_success.get(gid, pd.DataFrame())
        mean_success = 0.0
        if not order_df.empty and order_df["total_orders"].sum() > 0:
            mean_success = float(order_df["successful"].sum() / order_df["total_orders"].sum())

        mean_deception = None
        if deception_df is not None and "game_id" in deception_df.columns:
            game_dec = deception_df[deception_df["game_id"] == gid]
            valid = game_dec["deception_score"].apply(lambda x: isinstance(x, (int, float)) and x >= 0)
            if valid.any():
                mean_deception = float(game_dec.loc[valid, "deception_score"].mean())

        entry: dict[str, Any] = {
            "winner": result.get("winner"),
            "final_sc": final_sc,
            "elapsed": result.get("elapsed_seconds", 0),
            "mean_order_success_rate": round(mean_success, 3),
        }
        if mean_deception is not None:
            entry["mean_deception_score"] = round(mean_deception, 3)

        metrics["per_game"][gid] = entry

    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize AI Diplomacy experiment results."
    )
    parser.add_argument(
        "experiment_dir",
        help="Path to experiment directory (with experiment_summary.json) or single game directory.",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Output directory for plots and metrics. Defaults to <experiment_dir>/evaluation.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    exp_dir = args.experiment_dir.rstrip("/")
    if not os.path.isdir(exp_dir):
        logger.error("Directory not found: %s", exp_dir)
        sys.exit(1)

    output_dir = args.output_dir if args.output_dir else os.path.join(exp_dir, "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    exp = load_experiment(exp_dir)
    exp_summary = exp["summary"]
    config = exp["config"]
    game_dirs = exp["game_dirs"]

    if not game_dirs:
        logger.error("No game directories found. Exiting.")
        sys.exit(1)

    # Per-game data extraction
    all_sc_frames: list[pd.DataFrame] = []
    all_order_success: dict[str, pd.DataFrame] = {}
    game_ids: list[str] = []

    for gdir in game_dirs:
        gid = os.path.basename(gdir)
        game_ids.append(gid)
        phases = load_game_phases(gdir)
        if not phases:
            continue

        sc_df = extract_sc_trajectories(phases)
        sc_df["game_id"] = gid
        all_sc_frames.append(sc_df)

        order_df = extract_order_success(phases)
        all_order_success[gid] = order_df

        # Per-game trust / relationship plots
        trust_df = extract_trust_trajectories(phases)
        rel_df = extract_relationship_trajectories(phases)

        if not trust_df.empty:
            plot_trust_evolution(trust_df, output_dir, gid)
            plot_trust_matrix(trust_df, output_dir, gid)
        if not rel_df.empty:
            plot_relationship_evolution(rel_df, output_dir, gid)

    all_sc_df = pd.concat(all_sc_frames, ignore_index=True) if all_sc_frames else pd.DataFrame()

    # Deception data
    deception_df = load_deception_data(exp_dir)

    # ── Plots ────────────────────────────────────────────────────────────

    # Fig 1: SC trajectories
    if not all_sc_df.empty:
        plot_sc_trajectories(all_sc_df, output_dir, game_ids)

    # Fig 2: Memory vs baseline (requires experiment with config + multiple games)
    if exp_summary and config and len(game_ids) > 1:
        try:
            plot_memory_vs_baseline(exp_summary, config, output_dir)
        except Exception as exc:
            logger.warning("Skipping memory vs baseline plot: %s", exc)

    # Fig 3: Win/survival heatmap
    if exp_summary:
        try:
            plot_win_survival_heatmap(exp_summary, output_dir)
        except Exception as exc:
            logger.warning("Skipping win/survival heatmap: %s", exc)

    # Figs 6-7: Deception plots (optional)
    if deception_df is not None and not deception_df.empty:
        try:
            plot_deception_distribution(deception_df, config, output_dir)
        except Exception as exc:
            logger.warning("Skipping deception distribution plot: %s", exc)
        try:
            plot_deception_vs_sc(deception_df, all_sc_df, output_dir)
        except Exception as exc:
            logger.warning("Skipping deception vs SC plot: %s", exc)
    else:
        logger.info("No deception_scores.csv found — skipping deception plots (figs 6-7).")

    # ── Metrics summary ──────────────────────────────────────────────────
    metrics = build_metrics_summary(exp_summary, config, all_order_success, deception_df)
    metrics_path = os.path.join(output_dir, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote %s", metrics_path)

    # ── Print summary to stdout ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Experiment dir:  {exp_dir}")
    print(f"Output dir:      {output_dir}")
    print(f"Games evaluated: {len(game_ids)}")

    if metrics.get("improvement_delta"):
        d = metrics["improvement_delta"]
        print(f"\nImprovement Δ:   {d.get('delta', 'N/A'):+.3f}")
        print(f"  Memory mean:   {d.get('mean_sc_memory', 'N/A'):.3f}")
        print(f"  Baseline mean: {d.get('mean_sc_baseline', 'N/A'):.3f}")

    if metrics.get("memory_success_rate"):
        s = metrics["memory_success_rate"]
        print(f"\nWin rates:       memory={s.get('memory_win_rate', 'N/A'):.3f}  "
              f"baseline={s.get('baseline_win_rate', 'N/A'):.3f}")
        print(f"Survival rates:  memory={s.get('memory_survival_rate', 'N/A'):.3f}  "
              f"baseline={s.get('baseline_survival_rate', 'N/A'):.3f}")

    print(f"\nPer-game results:")
    for gid, gm in metrics.get("per_game", {}).items():
        if "error" in gm:
            print(f"  {gid}: ERROR - {gm['error'][:80]}")
            continue
        winner = gm.get("winner") or "draw/timeout"
        sc_vals = list(gm.get("final_sc", {}).values())
        max_sc = max(sc_vals) if sc_vals else 0
        dec_str = f"  dec={gm['mean_deception_score']:.2f}" if "mean_deception_score" in gm else ""
        print(f"  {gid}: winner={winner}  max_sc={max_sc}  "
              f"order_success={gm.get('mean_order_success_rate', 0):.2f}{dec_str}")

    print("=" * 60)
    print(f"Plots and metrics saved to: {output_dir}/")


if __name__ == "__main__":
    main()
