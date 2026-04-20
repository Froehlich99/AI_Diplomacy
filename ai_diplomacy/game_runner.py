"""
Core game runner extracted from lm_game.py for programmatic use.

Provides run_single_game() which can be called by:
  - lm_game.py (CLI wrapper)
  - experiment_runner.py (batch orchestration)
"""

import logging
import time
import os
import json
import asyncio
from collections import defaultdict
from argparse import Namespace
from typing import Dict, Any, Optional
import shutil
import sys

from diplomacy import Game

from .utils import get_valid_orders, gather_possible_orders, parse_prompts_dir_arg
from .negotiations import conduct_negotiations
from .planning import planning_phase
from .game_history import GameHistory
from .agent import DiplomacyAgent
from .game_logic import (
    save_game_state,
    load_game_state,
    initialize_new_game,
    export_agent_memories,
)
from .diary_logic import run_diary_consolidation
from config import config

logger = logging.getLogger(__name__)


def _detect_victory(game: Game, threshold: int = 18) -> bool:
    """True iff any power already owns >= `threshold` supply centres."""
    return any(len(p.centers) >= threshold for p in game.powers.values())


def _build_game_result(game: Game, agents: Dict[str, DiplomacyAgent], run_dir: str, elapsed: float) -> dict:
    """Build a summary dict of the game outcome."""
    supply_centers = {}
    for power_name, power in game.powers.items():
        supply_centers[power_name] = {
            "centers": list(power.centers),
            "count": len(power.centers),
            "eliminated": power.is_eliminated(),
        }

    winner = None
    for pn, info in supply_centers.items():
        if info["count"] >= 18:
            winner = pn

    return {
        "run_dir": run_dir,
        "elapsed_seconds": round(elapsed, 2),
        "final_phase": game.get_current_phase(),
        "winner": winner,
        "supply_centers": supply_centers,
        "power_model_map": getattr(game, "power_model_map", {}),
        "agent_memories_dir": os.path.join(run_dir, "agent_memories"),
    }


async def run_single_game(
    models: str,
    max_year: int,
    run_dir: str,
    prior_memory_dir: str = "",
    memory_cap_words: int = 500,
    num_negotiation_rounds: int = 0,
    simple_prompts: bool = True,
    planning_phase_enabled: bool = False,
    max_tokens: int = 16000,
    max_tokens_per_model: str = "",
    prompts_dir: Optional[str] = None,
    generate_phase_summaries: bool = False,
    use_unformatted_prompts: bool = False,
    country_specific_prompts: bool = False,
    end_at_phase: str = "",
    resume_from_phase: str = "",
    critical_state_analysis_dir: str = "",
    seed_base: int = 42,
) -> dict:
    """
    Run a single Diplomacy game programmatically.

    This is the core game loop extracted from lm_game.py.
    Returns a result dict with game outcome summary.
    """
    start_whole = time.time()

    # Build an args Namespace matching what the old main() expected
    args = Namespace(
        models=models,
        max_year=max_year,
        run_dir=run_dir,
        prior_memory_dir=prior_memory_dir,
        memory_cap_words=memory_cap_words,
        num_negotiation_rounds=num_negotiation_rounds,
        simple_prompts=simple_prompts,
        planning_phase=planning_phase_enabled,
        max_tokens=max_tokens,
        max_tokens_per_model=max_tokens_per_model,
        prompts_dir=prompts_dir,
        generate_phase_summaries=generate_phase_summaries,
        use_unformatted_prompts=use_unformatted_prompts,
        country_specific_prompts=country_specific_prompts,
        end_at_phase=end_at_phase,
        resume_from_phase=resume_from_phase,
        critical_state_analysis_dir=critical_state_analysis_dir,
        seed_base=seed_base,
    )

    # --- Config overrides ---
    if args.simple_prompts:
        config.SIMPLE_PROMPTS = True
        if args.prompts_dir is None:
            pkg_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ai_diplomacy")
            args.prompts_dir = os.path.join(pkg_root, "prompts_simple")
    else:
        config.SIMPLE_PROMPTS = False

    try:
        args.prompts_dir_map = parse_prompts_dir_arg(args.prompts_dir)
    except Exception as exc:
        raise RuntimeError(f"Prompt dir error: {exc}") from exc

    if args.generate_phase_summaries:
        import ai_diplomacy.narrative  # noqa: F401

    config.USE_UNFORMATTED_PROMPTS = bool(args.use_unformatted_prompts)
    config.COUNTRY_SPECIFIC_PROMPTS = bool(args.country_specific_prompts)

    if args.max_year is None:
        if args.end_at_phase:
            args.max_year = int(args.end_at_phase[1:5])
        else:
            raise ValueError("max_year is required")

    # --- Run directory ---
    run_dir = args.run_dir
    is_resuming = False
    if run_dir and os.path.exists(run_dir) and not args.critical_state_analysis_dir:
        is_resuming = True

    if args.critical_state_analysis_dir:
        if not run_dir:
            raise ValueError("run_dir must be given when using critical_state_analysis_dir")
        original_run_dir = run_dir
        run_dir = args.critical_state_analysis_dir
        os.makedirs(run_dir, exist_ok=True)
        src = os.path.join(original_run_dir, "lmvsgame.json")
        dst = os.path.join(run_dir, "lmvsgame.json")
        if not os.path.exists(src):
            raise FileNotFoundError(f"No saved game found at {src}")
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        is_resuming = True

    if not run_dir:
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        run_dir = f"./results/{timestamp_str}"

    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Using result directory: {run_dir}")

    # --- Logging ---
    general_log_file_path = os.path.join(run_dir, "general_game.log")
    file_handler = logging.FileHandler(general_log_file_path, mode="a")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - [%(funcName)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(file_handler)

    game_file_name = "lmvsgame.json"
    game_file_path = os.path.join(run_dir, game_file_name)
    llm_log_file_path = os.path.join(run_dir, "llm_responses.csv")
    model_error_stats = defaultdict(lambda: {"conversation_errors": 0, "order_decoding_errors": 0})

    run_config: Namespace = args

    # --- Initialize or load game state ---
    if is_resuming:
        try:
            game, agents, game_history, _ = load_game_state(
                run_dir, game_file_name, run_config, args.resume_from_phase
            )
            logger.info(f"Successfully resumed game from phase: {game.get_current_phase()}.")
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Could not resume game: {e}. Starting a new game instead.")
            is_resuming = False

    if not is_resuming:
        game = Game()
        game_history = GameHistory()
        if not hasattr(game, "phase_summaries"):
            game.phase_summaries = {}
        agents = await initialize_new_game(
            run_config, game, game_history, llm_log_file_path,
            prior_memory_dir=run_config.prior_memory_dir or None,
        )

    if _detect_victory(game):
        game.is_game_done = True
        logger.info(
            "Game already complete on load – a power has >=18 centres "
            f"(current phase {game.get_current_phase()})."
        )

    # --- Main game loop ---
    while not game.is_game_done:
        phase_start = time.time()
        current_phase = game.get_current_phase()
        current_short_phase = game.current_short_phase

        year_int = int(current_phase[1:5])
        if year_int > run_config.max_year:
            logger.info(f"Reached max year {run_config.max_year}, stopping simulation.")
            break
        if run_config.end_at_phase and current_phase == run_config.end_at_phase:
            logger.info(f"Reached end phase {run_config.end_at_phase}, stopping simulation.")
            break

        logger.info(f"PHASE: {current_phase} (time so far: {time.time() - start_whole:.2f}s)")
        game_history.add_phase(current_phase)

        # Pre-order steps (movement phases)
        if current_short_phase.endswith("M"):
            if run_config.num_negotiation_rounds > 0:
                game_history = await conduct_negotiations(
                    game, agents, game_history, model_error_stats,
                    max_rounds=run_config.num_negotiation_rounds,
                    log_file_path=llm_log_file_path,
                )
            if run_config.planning_phase:
                await planning_phase(
                    game, agents, game_history, model_error_stats,
                    log_file_path=llm_log_file_path,
                )

            neg_diary_tasks = [
                agent.generate_negotiation_diary_entry(game, game_history, llm_log_file_path)
                for agent in agents.values()
                if not game.powers[agent.power_name].is_eliminated()
            ]
            if neg_diary_tasks:
                await asyncio.gather(*neg_diary_tasks, return_exceptions=True)

        # Parallel order generation + diary consolidation
        consolidation_future = None
        if current_short_phase.startswith("S") and current_short_phase.endswith("M"):
            consolidation_tasks = [
                run_diary_consolidation(agent, game, llm_log_file_path, prompts_dir=agent.prompts_dir)
                for agent in agents.values()
                if not game.powers[agent.power_name].is_eliminated()
            ]
            if consolidation_tasks:
                consolidation_future = asyncio.gather(*consolidation_tasks, return_exceptions=True)

        logger.info("Getting orders from agents...")
        board_state = game.get_state()
        order_tasks = []
        for power_name, agent in agents.items():
            if not game.powers[power_name].is_eliminated():
                possible_orders = gather_possible_orders(game, power_name)
                if not possible_orders:
                    game.set_orders(power_name, [])
                    continue
                order_tasks.append(
                    get_valid_orders(
                        game, agent.client, board_state, power_name, possible_orders,
                        game_history, model_error_stats,
                        agent_goals=agent.goals,
                        agent_relationships=agent.relationships,
                        agent_trust_scores=agent.trust_scores,
                        agent_private_diary_str=agent.get_latest_phase_diary_entries(),
                        log_file_path=llm_log_file_path,
                        phase=current_phase,
                    )
                )

        order_results = await asyncio.gather(*order_tasks, return_exceptions=True)

        if consolidation_future:
            await consolidation_future

        active_powers = [p for p, a in agents.items() if not game.powers[p].is_eliminated()]
        order_power_names = [p for p in active_powers if gather_possible_orders(game, p)]
        submitted_orders_this_phase = defaultdict(list)

        for i, result in enumerate(order_results):
            p_name = order_power_names[i]
            if isinstance(result, Exception):
                logger.error("Error getting orders for %s: %s", p_name, result, exc_info=result)
                valid, invalid = [], []
            else:
                valid = result.get("valid", [])
                invalid = result.get("invalid", [])
            game.set_orders(p_name, valid)
            submitted_orders_this_phase[p_name] = valid + invalid

        # Process phase
        completed_phase = current_phase
        game.process()
        logger.info(f"Results for {current_phase}:")
        for power_name, power in game.powers.items():
            logger.info(f"{power_name}: {power.centers}")

        # Post-processing
        phase_history_from_game = game.get_phase_history()
        if phase_history_from_game:
            last_phase_from_game = phase_history_from_game[-1]
            if last_phase_from_game.name == completed_phase:
                phase_obj_in_my_history = game_history._get_phase(completed_phase)
                if phase_obj_in_my_history:
                    phase_obj_in_my_history.submitted_orders_by_power = submitted_orders_this_phase
                    phase_obj_in_my_history.orders_by_power = last_phase_from_game.orders
                    converted_results = defaultdict(list)
                    if last_phase_from_game.results:
                        for pwr, res_list in last_phase_from_game.results.items():
                            converted_results[pwr] = [[res] for res in res_list]
                    phase_obj_in_my_history.results_by_power = converted_results

        phase_summary = game.phase_summaries.get(current_phase, "(Summary not generated)")
        all_orders_this_phase = game.order_history.get(current_short_phase, {})

        if current_short_phase.endswith("M"):
            phase_result_diary_tasks = [
                agent.generate_phase_result_diary_entry(
                    game, game_history, phase_summary, all_orders_this_phase,
                    llm_log_file_path, current_short_phase,
                )
                for agent in agents.values()
                if not game.powers[agent.power_name].is_eliminated()
            ]
            if phase_result_diary_tasks:
                await asyncio.gather(*phase_result_diary_tasks, return_exceptions=True)

        if current_short_phase.endswith("M") and run_config.num_negotiation_rounds == 0:
            current_board_state = game.get_state()
            state_update_tasks = [
                agent.analyze_phase_and_update_state(
                    game, current_board_state, phase_summary, game_history, llm_log_file_path,
                )
                for agent in agents.values()
                if not game.powers[agent.power_name].is_eliminated()
            ]
            if state_update_tasks:
                await asyncio.gather(*state_update_tasks, return_exceptions=True)

        await save_game_state(game, agents, game_history, game_file_path, run_config, completed_phase)
        logger.info(f"Phase {current_phase} took {time.time() - phase_start:.2f}s")

    # --- Game end ---
    total_time = time.time() - start_whole
    logger.info(f"Game ended after {total_time:.2f}s. Final state saved in {run_dir}")

    # Export cross-game memory
    try:
        memory_cap = getattr(run_config, "memory_cap_words", 500)
        exported_memories = export_agent_memories(game, agents, run_dir, memory_cap_words=memory_cap)
        logger.info(f"Exported cross-game memories for {len(exported_memories)} agents to {run_dir}/agent_memories/")
    except Exception as e:
        logger.error(f"Failed to export agent memories: {e}", exc_info=True)

    # Save overview
    overview_file_path = os.path.join(run_dir, "overview.jsonl")
    with open(overview_file_path, "w") as overview_file:
        cfg = vars(run_config).copy()
        if "prompts_dir_map" in cfg and isinstance(cfg["prompts_dir_map"], dict):
            cfg["prompts_dir_map"] = {p: str(path) for p, path in cfg["prompts_dir_map"].items()}
        overview_file.write(json.dumps(model_error_stats) + "\n")
        overview_file.write(json.dumps(getattr(game, "power_model_map", {})) + "\n")
        overview_file.write(json.dumps(cfg) + "\n")

    # Remove file handler to avoid accumulating handlers across games
    logging.getLogger().removeHandler(file_handler)
    file_handler.close()

    logger.info("Done.")
    return _build_game_result(game, agents, run_dir, total_time)
