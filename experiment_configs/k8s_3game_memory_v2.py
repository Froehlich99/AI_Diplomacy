"""
K8s 3-game memory experiment v2: Full 3-game run with model-centric memory fix.

Same 7 models as the original experiment, with power rotation and memory chaining.
All 3 games run in a single k8s job (no separate local Game 1).

Power rotation (shift per game):
  Game 1 (shift=0): grok | gemma | gemini-flash | qwen3.5 | qwen3.6 | gpt-oss | haiku
  Game 2 (shift=1): gemma | gemini-flash | qwen3.5 | qwen3.6 | gpt-oss | haiku | grok
  Game 3 (shift=2): gemini-flash | qwen3.5 | qwen3.6 | gpt-oss | haiku | grok | gemma

Usage:
    python experiment_runner.py experiment_configs/k8s_3game_memory_v2.py \\
        --output_dir /app/results/k8s_3game_memory_v2
"""

EXPERIMENT = {
    "name": "k8s_3game_memory_v2",
    "description": (
        "Full 3-game memory experiment with model-centric memory fix. "
        "Tests whether LLM agents improve across sequential games with "
        "cross-game memory and power rotation."
    ),

    # Game settings
    "max_year": 1910,
    "num_negotiation_rounds": 2,
    "simple_prompts": True,
    "planning_phase": False,
    "max_tokens": 16000,
    "memory_cap_words": 500,

    # Same 7 models as v1 (order: AUSTRIA, ENGLAND, FRANCE, GERMANY, ITALY, RUSSIA, TURKEY)
    "models": [
        "openrouter:x-ai/grok-4.1-fast",
        "openrouter:google/gemma-4-31b-it",
        "openrouter:google/gemini-2.5-flash-lite",
        "openrouter:qwen/qwen3.5-27b",
        "openrouter:qwen/qwen3.6-plus",
        "openrouter:openai/gpt-oss-120b",
        "openrouter:anthropic/claude-haiku-4.5",
    ],

    # Full rotation starting from shift=0
    "rotate_powers": True,
    "rotation_offset": 0,

    # All 3 games in one run
    "games": [
        {"id": "game1", "memory": False},
        {"id": "game2", "memory": True},
        {"id": "game3", "memory": True},
    ],
}
