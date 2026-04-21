#!/usr/bin/env python3
"""
Experiment runner for multi-game Diplomacy experiments.

Orchestrates sequential games with:
  - Power rotation (circular shift of model assignments)
  - Cross-game memory chaining (memory from game N feeds game N+1)
  - Naive baseline mode (no memory injection)
  - Resume from crash (skips completed games)

Usage:
    python experiment_runner.py experiment_configs/memory_vs_baseline.py
    python experiment_runner.py experiment_configs/memory_vs_baseline.py --output_dir results/my_experiment
"""

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

import dotenv

# Suppress gRPC noise before any imports
os.environ["GRPC_PYTHON_LOG_LEVEL"] = "40"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["ABSL_MIN_LOG_LEVEL"] = "2"
os.environ["GRPC_POLL_STRATEGY"] = "poll"

dotenv.load_dotenv()

from ai_diplomacy.game_runner import run_single_game  # noqa: E402

logger = logging.getLogger(__name__)

ALL_POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]


def load_experiment_config(config_path: str) -> dict:
    """Load an experiment config from a Python file."""
    spec = importlib.util.spec_from_file_location("experiment_config", config_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Cannot load config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "EXPERIMENT"):
        raise ValueError(f"Config file {config_path} must define an EXPERIMENT dict")

    return module.EXPERIMENT


def rotate_models(models: List[str], shift: int) -> List[str]:
    """Circular-shift the model list by `shift` positions."""
    if not models or shift == 0:
        return list(models)
    n = len(models)
    shift = shift % n
    return models[shift:] + models[:shift]


def load_experiment_state(output_dir: str) -> dict:
    """Load experiment progress state for resume support."""
    state_path = os.path.join(output_dir, "experiment_state.json")
    if os.path.isfile(state_path):
        with open(state_path, "r") as f:
            return json.load(f)
    return {"completed_games": [], "results": {}, "last_memory_dir": ""}


def save_experiment_state(output_dir: str, state: dict):
    """Persist experiment progress for crash recovery."""
    state_path = os.path.join(output_dir, "experiment_state.json")
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


async def run_experiment(config: dict, output_dir: str):
    """Execute a full experiment: multiple sequential games."""

    experiment_name = config.get("name", "unnamed_experiment")
    models = config.get("models", [])
    games = config.get("games", [])
    rotate_powers = config.get("rotate_powers", False)
    rotation_offset = config.get("rotation_offset", 0)
    initial_memory_dir = config.get("initial_memory_dir", "")
    max_year = config["max_year"]

    # Shared game settings
    shared_kwargs = {
        "max_year": max_year,
        "num_negotiation_rounds": config.get("num_negotiation_rounds", 0),
        "simple_prompts": config.get("simple_prompts", True),
        "planning_phase_enabled": config.get("planning_phase", False),
        "max_tokens": config.get("max_tokens", 16000),
        "memory_cap_words": config.get("memory_cap_words", 500),
        "generate_phase_summaries": config.get("generate_phase_summaries", False),
    }

    os.makedirs(output_dir, exist_ok=True)

    # Save config copy
    config_copy_path = os.path.join(output_dir, "experiment_config.json")
    if not os.path.exists(config_copy_path):
        # Make a JSON-safe copy (convert any non-serializable items)
        safe_config = {k: v for k, v in config.items()}
        with open(config_copy_path, "w") as f:
            json.dump(safe_config, f, indent=2)

    # Load or initialize state
    state = load_experiment_state(output_dir)
    completed = set(state.get("completed_games", []))
    last_memory_dir = state.get("last_memory_dir", "")

    # Seed initial memory from config (e.g. prior local game uploaded to B2)
    if not last_memory_dir and initial_memory_dir and os.path.isdir(initial_memory_dir):
        last_memory_dir = initial_memory_dir
        logger.info(f"Seeding initial memory from: {initial_memory_dir}")

    logger.info(f"=== Experiment: {experiment_name} ===")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Games: {len(games)} total, {len(completed)} already completed")
    logger.info(f"Models: {models}")
    logger.info(f"Power rotation: {rotate_powers} (offset={rotation_offset})")

    all_results = state.get("results", {})

    for game_idx, game_spec in enumerate(games):
        game_id = game_spec.get("id", f"game_{game_idx}")

        if game_id in completed:
            logger.info(f"[{game_id}] Already completed, skipping.")
            # Restore last_memory_dir from completed game
            game_run_dir = os.path.join(output_dir, game_id)
            memories_dir = os.path.join(game_run_dir, "agent_memories")
            if os.path.isdir(memories_dir):
                last_memory_dir = memories_dir
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"[{game_id}] Starting game {game_idx + 1}/{len(games)}")
        logger.info(f"{'='*60}")

        # Apply power rotation
        if rotate_powers and len(models) == 7:
            rotated = rotate_models(models, game_idx + rotation_offset)
        else:
            rotated = list(models)

        models_str = ",".join(rotated) if len(rotated) > 1 else rotated[0] if rotated else ""

        # Log model-power assignment
        if len(rotated) == 7:
            for power, model in zip(ALL_POWERS, rotated):
                logger.info(f"  {power}: {model}")

        # Determine memory injection
        use_memory = game_spec.get("memory", False)
        prior_memory = ""
        if use_memory and last_memory_dir and os.path.isdir(last_memory_dir):
            prior_memory = last_memory_dir
            logger.info(f"[{game_id}] Injecting prior memory from: {prior_memory}")
        elif use_memory:
            logger.warning(f"[{game_id}] memory=True but no prior memory available (first game?). Running without.")

        game_run_dir = os.path.join(output_dir, game_id)

        game_start = time.time()
        try:
            result = await run_single_game(
                models=models_str,
                run_dir=game_run_dir,
                prior_memory_dir=prior_memory,
                **shared_kwargs,
            )
            game_elapsed = time.time() - game_start

            logger.info(f"[{game_id}] Completed in {game_elapsed:.1f}s")
            logger.info(f"[{game_id}] Winner: {result.get('winner', 'None (draw/timeout)')}")
            for pn, info in result.get("supply_centers", {}).items():
                logger.info(f"  {pn}: {info['count']} SCs")

            all_results[game_id] = result

            # Update memory chain
            memories_dir = os.path.join(game_run_dir, "agent_memories")
            if os.path.isdir(memories_dir):
                last_memory_dir = memories_dir

        except Exception as e:
            logger.error(f"[{game_id}] FAILED: {e}", exc_info=True)
            all_results[game_id] = {"error": str(e)}

        # Mark as completed and save state (crash recovery)
        completed.add(game_id)
        state = {
            "completed_games": list(completed),
            "results": all_results,
            "last_memory_dir": last_memory_dir,
        }
        save_experiment_state(output_dir, state)

    # Write final summary
    summary_path = os.path.join(output_dir, "experiment_summary.json")
    summary = {
        "experiment_name": experiment_name,
        "total_games": len(games),
        "completed_games": len(completed),
        "results": all_results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info(f"Experiment complete: {len(completed)}/{len(games)} games")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"{'='*60}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run a multi-game Diplomacy experiment."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to experiment config Python file (must define EXPERIMENT dict).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Override output directory (default: results/<experiment_name>).",
    )
    parser.add_argument(
        "--initial_memory_dir",
        type=str,
        default="",
        help="Seed cross-game memory from a prior game's agent_memories/ directory.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    config = load_experiment_config(args.config)

    # CLI overrides for config values
    if args.initial_memory_dir:
        config["initial_memory_dir"] = args.initial_memory_dir

    output_dir = args.output_dir
    if not output_dir:
        name = config.get("name", "experiment")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/{name}_{timestamp}"

    asyncio.run(run_experiment(config, output_dir))


if __name__ == "__main__":
    main()
