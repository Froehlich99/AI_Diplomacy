"""
Main experiment: 7 different LLMs playing Diplomacy with cross-game memory vs baseline.

Memory series (3 games): Game 1 has no memory, Game 2 inherits Game 1's memory,
Game 3 inherits Game 2's memory. Tests self-improvement across sequential games.

Baseline series (3 games): No memory injection at all. Control group.

Power rotation enabled: models shift positions each game to reduce positional bias.

Models (one per power, rotated):
  - x-ai/grok-4.1-fast
  - google/gemini-2.5-flash-lite
  - deepseek/deepseek-v3.2
  - z-ai/glm-4.7-flash
  - xiaomi/mimo-v2-flash
  - qwen/qwen-2.5-7b-instruct
  - mistralai/mistral-small-3.2-24b-instruct

Usage:
    python experiment_runner.py experiment_configs/memory_vs_baseline_7models.py
"""

EXPERIMENT = {
    "name": "memory_vs_baseline_7models",
    "description": (
        "6-game experiment comparing LLM agent performance with cross-game memory "
        "vs naive baseline. 7 different models, power rotation, 1 negotiation round. "
        "Games run to 1910 max year."
    ),

    # Game settings
    "max_year": 1910,
    "num_negotiation_rounds": 0,
    "simple_prompts": True,
    "planning_phase": False,
    "max_tokens": 16000,
    "memory_cap_words": 500,

    # 7 models, one per power (AUSTRIA, ENGLAND, FRANCE, GERMANY, ITALY, RUSSIA, TURKEY)
    "models": [
        "openrouter:x-ai/grok-4.1-fast",
        "openrouter:google/gemini-2.5-flash-lite",
        "openrouter:deepseek/deepseek-v3.2",
        "openrouter:z-ai/glm-4.7-flash",
        "openrouter:xiaomi/mimo-v2-flash",
        "openrouter:qwen/qwen-2.5-7b-instruct",
        "openrouter:mistralai/mistral-small-3.2-24b-instruct",
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
