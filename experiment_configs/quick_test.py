"""Quick 2-game test config: 1 baseline + 1 memory game, ending after S1901M."""

EXPERIMENT = {
    "name": "quick_test",
    "description": "Quick 2-game test to verify experiment runner works",
    "max_year": 1901,
    "num_negotiation_rounds": 0,
    "simple_prompts": True,
    "models": [
        "openrouter:x-ai/grok-4.1-fast",
        "openrouter:x-ai/grok-4.1-fast",
        "openrouter:x-ai/grok-4.1-fast",
        "openrouter:x-ai/grok-4.1-fast",
        "openrouter:x-ai/grok-4.1-fast",
        "openrouter:x-ai/grok-4.1-fast",
        "openrouter:x-ai/grok-4.1-fast",
    ],
    "rotate_powers": True,
    "games": [
        {"id": "game1_baseline", "memory": False},
        {"id": "game2_memory", "memory": True},
    ],
}
