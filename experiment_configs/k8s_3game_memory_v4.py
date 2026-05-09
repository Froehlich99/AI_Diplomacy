"""
K8s 3-game memory experiment v4: GPT-5.4 and DeepSeek V4 Pro replace GPT-OSS-120B and Gemini Flash.

Same 7-model setup with power rotation and memory chaining.
3 games with full rotation. Claude Opus 4.6 retained from v3.

Power rotation (shift per game):
  Game 1 (shift=0): grok | gemma | deepseek-v4 | qwen3.5 | qwen3.6 | gpt-5.4 | opus
  Game 2 (shift=1): gemma | deepseek-v4 | qwen3.5 | qwen3.6 | gpt-5.4 | opus | grok
  Game 3 (shift=2): deepseek-v4 | qwen3.5 | qwen3.6 | gpt-5.4 | opus | grok | gemma

Usage:
    python experiment_runner.py experiment_configs/k8s_3game_memory_v4.py \
        --output_dir /app/results/k8s_3game_memory_v4
"""

EXPERIMENT = {
    "name": "k8s_3game_memory_v4",
    "description": (
        "3-game memory experiment with GPT-5.4 replacing GPT-OSS-120B and "
        "DeepSeek V4 Pro replacing Gemini 2.5 Flash Lite. "
        "Tests cross-game learning with memory and power rotation."
    ),

    # Game settings
    "max_year": 1910,
    "num_negotiation_rounds": 2,
    "simple_prompts": True,
    "planning_phase": False,
    "max_tokens": 16000,
    "memory_cap_words": 500,

    # 7 models (order: AUSTRIA, ENGLAND, FRANCE, GERMANY, ITALY, RUSSIA, TURKEY)
    "models": [
        "openrouter:x-ai/grok-4.1-fast",
        "openrouter:google/gemma-4-31b-it",
        "openrouter:deepseek/deepseek-v4-pro",
        "openrouter:qwen/qwen3.5-27b",
        "openrouter:qwen/qwen3.6-plus",
        "openrouter:openai/gpt-5.4",
        "openrouter:anthropic/claude-opus-4.6",
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
