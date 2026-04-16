import argparse
import logging
import os
import asyncio
import sys

import dotenv

# Suppress Gemini/PaLM gRPC warnings
os.environ["GRPC_PYTHON_LOG_LEVEL"] = "40"  # ERROR level only
os.environ["GRPC_VERBOSITY"] = "ERROR"  # Additional gRPC verbosity control
os.environ["ABSL_MIN_LOG_LEVEL"] = "2"  # Suppress abseil warnings
# Disable gRPC forking warnings
os.environ["GRPC_POLL_STRATEGY"] = "poll"  # Use 'poll' for macOS compatibility

dotenv.load_dotenv()

from ai_diplomacy.game_runner import run_single_game  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)


def _str2bool(v: str) -> bool:
    v = str(v).lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a Diplomacy game simulation with configurable parameters."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="Directory for results. If it exists, the game resumes. If not, it's created. Defaults to a new timestamped directory.",
    )
    parser.add_argument(
        "--output",            # alias for back compatibility
        dest="run_dir",        # write to the same variable as --run_dir
        type=str,
        help=argparse.SUPPRESS # hides it from `--help`
    )
    parser.add_argument(
        "--critical_state_analysis_dir",
        type=str,
        default="",
        help="Resumes from the game state in --run_dir, but saves new results to a separate dir, leaving the original run_dir intact.",
    )
    parser.add_argument(
        "--resume_from_phase",
        type=str,
        default="",
        help="Phase to resume from (e.g., 'S1902M'). Requires --run_dir. IMPORTANT: This option clears any existing phase results ahead of & including the specified resume phase.",
    )
    parser.add_argument(
        "--end_at_phase",
        type=str,
        default="",
        help="Phase to end the simulation after (e.g., 'F1905M').",
    )
    parser.add_argument(
        "--max_year",
        type=int,
        help="Maximum year to simulate. The game will stop once this year is reached.",
    )
    parser.add_argument(
        "--num_negotiation_rounds",
        type=int,
        default=0,
        help="Number of negotiation rounds per phase.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help=(
            "Comma-separated list of model names to assign to powers in order. "
            "The order is: AUSTRIA, ENGLAND, FRANCE, GERMANY, ITALY, RUSSIA, TURKEY."
        ),
    )
    parser.add_argument(
        "--planning_phase", 
        action="store_true",
        help="Enable the planning phase for each power to set strategic directives.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=16000,
        help="Maximum number of new tokens to generate per LLM call (default: 16000)."
    )
    parser.add_argument(
        "--seed_base",
        type=int,
        default=42,
        help="RNG seed placeholder for compatibility with experiment_runner. Currently unused."
    )
    parser.add_argument(
        "--max_tokens_per_model",
        type=str,
        default="",
        help="Comma-separated list of 7 token limits (in order: AUSTRIA, ENGLAND, FRANCE, GERMANY, ITALY, RUSSIA, TURKEY). Overrides --max_tokens."
    )
    parser.add_argument(
        "--prompts_dir",
        type=str,
        default=None,
        help="Path to the directory containing prompt files. Defaults to the packaged prompts directory."
    )
    parser.add_argument(
        "--simple_prompts",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
        help=(
            "When true (1 / true / yes) the engine switches to simpler prompts which low-midrange models handle better."
        ),
    )
    parser.add_argument(
        "--generate_phase_summaries",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "When true (1 / true / yes / default) generates narrative phase summaries. "
            "Set to false (0 / false / no) to skip phase summary generation."
        ),
    )
    parser.add_argument(
        "--use_unformatted_prompts",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "When true (1 / true / yes / default) uses two-step approach: unformatted prompts + Gemini Flash formatting. "
            "Set to false (0 / false / no) to use original single-step formatted prompts."
        ),
    )
    parser.add_argument(
        "--country_specific_prompts",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "When true (1 / true / yes) enables country-specific order and conversation prompts. "
            "Each power will use their own custom prompts if available (e.g., order_instructions_movement_phase_france.txt). "
            "Falls back to generic prompts if country-specific not found."
        ),
    )
    parser.add_argument(
        "--prior_memory_dir",
        type=str,
        default="",
        help=(
            "Path to a directory containing prior-game agent memory files "
            "(e.g., results/<prev_game>/agent_memories/). "
            "When set, agents start with experience from the previous game injected into their system prompt."
        ),
    )
    parser.add_argument(
        "--memory_cap_words",
        type=int,
        default=500,
        help="Maximum number of words to retain in the exported cross-game diary memory (default: 500).",
    )

    return parser.parse_args()


async def main():
    args = parse_arguments()

    if args.max_year is None:
        if args.end_at_phase:
            args.max_year = int(args.end_at_phase[1:5])
        else:
            print("ERROR: --max_year is required.", file=sys.stderr)
            sys.exit(1)

    result = await run_single_game(
        models=args.models,
        max_year=args.max_year,
        run_dir=args.run_dir,
        prior_memory_dir=args.prior_memory_dir,
        memory_cap_words=args.memory_cap_words,
        num_negotiation_rounds=args.num_negotiation_rounds,
        simple_prompts=args.simple_prompts,
        planning_phase_enabled=args.planning_phase,
        max_tokens=args.max_tokens,
        max_tokens_per_model=args.max_tokens_per_model,
        prompts_dir=args.prompts_dir,
        generate_phase_summaries=args.generate_phase_summaries,
        use_unformatted_prompts=args.use_unformatted_prompts,
        country_specific_prompts=args.country_specific_prompts,
        end_at_phase=args.end_at_phase,
        resume_from_phase=args.resume_from_phase,
        critical_state_analysis_dir=args.critical_state_analysis_dir,
        seed_base=args.seed_base,
    )

    logger.info(f"Game result: winner={result.get('winner')}, "
                f"phase={result.get('final_phase')}, "
                f"elapsed={result.get('elapsed_seconds')}s")


if __name__ == "__main__":
    asyncio.run(main())
