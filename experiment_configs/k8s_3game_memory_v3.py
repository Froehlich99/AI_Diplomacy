"""
K8s 2-game memory experiment v3: Same as v2 but with Claude Opus 4.6 replacing Haiku 4.5.

Same 7-model setup with power rotation and memory chaining.
Reduced to 2 games to fit within OpenRouter budget (~$40).
Opus thinking budget set to 2048 tokens.

Power rotation (shift per game):
  Game 1 (shift=0): grok | gemma | gemini-flash | qwen3.5 | qwen3.6 | gpt-oss | opus
  Game 2 (shift=1): gemma | gemini-flash | qwen3.5 | qwen3.6 | gpt-oss | opus | grok

Usage:
    python experiment_runner.py experiment_configs/k8s_3game_memory_v3.py \
        --output_dir /app/results/k8s_3game_memory_v3
"""

EXPERIMENT = {
    "name": "k8s_3game_memory_v3",
    "description": (
        "2-game memory experiment with Claude Opus 4.6 replacing Haiku 4.5. "
        "Tests whether a stronger reasoning model improves cross-game learning "
        "with memory and power rotation. Reduced to 2 games for budget."
    ),

    # Game settings
    "max_year": 1910,
    "num_negotiation_rounds": 2,
    "simple_prompts": True,
    "planning_phase": False,
    "max_tokens": 16000,
    "memory_cap_words": 500,

    # Same 7 models as v2 but with opus replacing haiku (order: AUSTRIA, ENGLAND, FRANCE, GERMANY, ITALY, RUSSIA, TURKEY)
    "models": [
        "openrouter:x-ai/grok-4.1-fast",
        "openrouter:google/gemma-4-31b-it",
        "openrouter:google/gemini-2.5-flash-lite",
        "openrouter:qwen/qwen3.5-27b",
        "openrouter:qwen/qwen3.6-plus",
        "openrouter:openai/gpt-oss-120b",
        "openrouter:anthropic/claude-opus-4.6",
    ],

    # Full rotation starting from shift=0
    "rotate_powers": True,
    "rotation_offset": 0,

    # All 2 games in one run
    "games": [
        {"id": "game1", "memory": False},
        {"id": "game2", "memory": True},
    ],
}
