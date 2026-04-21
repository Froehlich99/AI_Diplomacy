"""
K8s 3-game memory experiment: Games 2 and 3 with cross-game memory.

Game 1 was run locally (results/20260421_144558). This config runs Games 2 and 3
with power rotation and memory chaining from Game 1.

Power rotation (same 7 models as Game 1, shifted):
  Game 1 (shift=0, LOCAL):  grok | gemma | gemini-flash | qwen3.5 | qwen3.6 | gpt-oss | haiku
  Game 2 (shift=1, THIS):   gemma | gemini-flash | qwen3.5 | qwen3.6 | gpt-oss | haiku | grok
  Game 3 (shift=2, THIS):   gemini-flash | qwen3.5 | qwen3.6 | gpt-oss | haiku | grok | gemma

Usage:
    python experiment_runner.py experiment_configs/k8s_3game_memory.py \\
        --initial_memory_dir /app/prior_memories \\
        --output_dir /app/results/k8s_3game_memory
"""

EXPERIMENT = {
    "name": "k8s_3game_memory",
    "description": (
        "Games 2-3 of a 3-game memory experiment. Game 1 ran locally. "
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

    # Same 7 models as Game 1 (order: AUSTRIA, ENGLAND, FRANCE, GERMANY, ITALY, RUSSIA, TURKEY)
    "models": [
        "openrouter:x-ai/grok-4.1-fast",
        "openrouter:google/gemma-4-31b-it",
        "openrouter:google/gemini-2.5-flash-lite",
        "openrouter:qwen/qwen3.5-27b",
        "openrouter:qwen/qwen3.6-plus",
        "openrouter:openai/gpt-oss-120b",
        "openrouter:anthropic/claude-haiku-4.5",
    ],

    # Start rotation at shift=1 since Game 1 was shift=0
    "rotate_powers": True,
    "rotation_offset": 1,

    # Game schedule — both inherit memory from previous
    "games": [
        {"id": "game2", "memory": True},
        {"id": "game3", "memory": True},
    ],
}
