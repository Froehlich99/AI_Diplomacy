"""
Example experiment: 3 games with memory chaining + 3 baseline games.

Models are rotated across powers each game to reduce positional bias.
Memory games form a chain: game1 -> game2 -> game3.
Baseline games have no memory injection at all.

Usage:
    python experiment_runner.py experiment_configs/memory_vs_baseline.py
"""

EXPERIMENT = {
    "name": "memory_vs_baseline",
    "description": (
        "Compare LLM agent performance with cross-game memory vs. naive baseline. "
        "3 sequential memory games (each inheriting the previous game's experience) "
        "followed by 3 independent baseline games without any memory."
    ),

    # Max game length
    "max_year": 1910,

    # Game settings
    "num_negotiation_rounds": 1,
    "simple_prompts": True,
    "planning_phase": False,
    "max_tokens": 16000,
    "memory_cap_words": 500,

    # 7 models, one per power (AUSTRIA, ENGLAND, FRANCE, GERMANY, ITALY, RUSSIA, TURKEY)
    "models": [
        "openrouter:x-ai/grok-4.1-fast",
        "openrouter:x-ai/grok-4.1-fast",
        "openrouter:x-ai/grok-4.1-fast",
        "openrouter:x-ai/grok-4.1-fast",
        "openrouter:x-ai/grok-4.1-fast",
        "openrouter:x-ai/grok-4.1-fast",
        "openrouter:x-ai/grok-4.1-fast",
    ],

    # Rotate model-power assignments each game (circular shift)
    "rotate_powers": True,

    # Game schedule
    "games": [
        # Memory series: each game feeds memory to the next
        {"id": "mem_game1", "memory": False},   # first game has no prior memory
        {"id": "mem_game2", "memory": True},     # inherits from mem_game1
        {"id": "mem_game3", "memory": True},     # inherits from mem_game2

        # Baseline series: no memory injection at all
        {"id": "baseline1", "memory": False},
        {"id": "baseline2", "memory": False},
        {"id": "baseline3", "memory": False},
    ],
}
