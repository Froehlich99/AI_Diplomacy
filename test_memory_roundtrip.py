#!/usr/bin/env python3
"""
Test that cross-game memory export/load works correctly with model-centric keying.

Verifies:
1. Memory files are named by model ID (not power)
2. After power rotation, each model loads its OWN prior memory
3. Trust scores and relationships use canonical model names (no provider prefix)
4. format_prior_experience produces a prompt with correct model references

Run:
    python test_memory_roundtrip.py
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(__file__))

from ai_diplomacy.game_logic import (
    export_agent_memories,
    format_prior_experience,
    load_agent_memory,
    sanitize_model_id,
    strip_model_prefix,
)

ALL_POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

# Same models as the experiment config (canonical names, no prefix)
MODELS = [
    "x-ai/grok-4.1-fast",
    "google/gemma-4-31b-it",
    "google/gemini-2.5-flash-lite",
    "qwen/qwen3.5-27b",
    "qwen/qwen3.6-plus",
    "openai/gpt-oss-120b",
    "anthropic/claude-haiku-4.5",
]

# Config-style model IDs (with openrouter: prefix, as they appear in experiment configs)
CONFIG_MODELS = [f"openrouter:{m}" for m in MODELS]


def make_mock_agent(power_name, model_name):
    """Create a mock agent with the essential attributes for memory export."""
    agent = MagicMock()
    agent.power_name = power_name
    agent.client.model_name = model_name
    agent.private_diary = [f"Diary entry for {model_name} playing {power_name}. This model was strategic."]
    agent.full_private_diary = agent.private_diary
    agent.trust_scores = {}
    agent.relationships = {}
    agent.goals = [f"Goal for {model_name}"]

    # Set trust/relationships toward other models (keyed by power, as the agent stores them during game)
    for other_power in ALL_POWERS:
        if other_power != power_name:
            agent.trust_scores[other_power] = 0.5
            agent.relationships[other_power] = "Neutral"

    return agent


def make_mock_game(power_model_map_config):
    """Create a mock game object."""
    game = MagicMock()
    game.power_model_map = power_model_map_config
    game.is_game_done = True

    # Mock powers
    powers = {}
    for power_name in ALL_POWERS:
        power_obj = MagicMock()
        power_obj.is_eliminated.return_value = False
        power_obj.centers = ["CTR1", "CTR2", "CTR3"]
        powers[power_name] = power_obj
    game.powers = powers

    # Mock phase history
    phase = MagicMock()
    phase.name = "W1910A"
    game.get_phase_history.return_value = [phase]

    return game


def test_strip_model_prefix():
    """Test that strip_model_prefix handles all cases."""
    assert strip_model_prefix("openrouter:x-ai/grok-4.1-fast") == "x-ai/grok-4.1-fast"
    assert strip_model_prefix("anthropic:claude-haiku-4.5") == "claude-haiku-4.5"
    assert strip_model_prefix("gpt-4o") == "gpt-4o"
    assert strip_model_prefix("openai:o4-mini") == "o4-mini"
    print("  PASS: strip_model_prefix")


def test_sanitize_model_id():
    """Test filename sanitization."""
    assert sanitize_model_id("x-ai/grok-4.1-fast") == "x-ai_grok-4.1-fast"
    assert sanitize_model_id("google/gemma-4-31b-it") == "google_gemma-4-31b-it"
    assert sanitize_model_id("gpt-4o") == "gpt-4o"
    print("  PASS: sanitize_model_id")


def test_export_creates_model_keyed_files():
    """Test that export creates files named by model, not power."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Game 1: models assigned in order
        game1_map = dict(zip(ALL_POWERS, CONFIG_MODELS))
        game = make_mock_game(game1_map)
        agents = {
            power: make_mock_agent(power, model)
            for power, model in zip(ALL_POWERS, MODELS)
        }

        exported = export_agent_memories(game, agents, tmpdir)

        memories_dir = os.path.join(tmpdir, "agent_memories")
        files = sorted(os.listdir(memories_dir))

        # Should have 7 files, one per model
        assert len(files) == 7, f"Expected 7 files, got {len(files)}: {files}"

        # Files should be named by model ID, not power
        for model in MODELS:
            expected_file = f"{sanitize_model_id(model)}_memory.json"
            assert expected_file in files, f"Missing {expected_file}, got {files}"

        # No power-named files should exist
        for power in ALL_POWERS:
            assert f"{power}_memory.json" not in files, f"Found power-named file {power}_memory.json"

        print("  PASS: export creates model-keyed files")


def test_export_trust_scores_are_model_keyed():
    """Test that exported trust scores use model names, not power names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        game1_map = dict(zip(ALL_POWERS, CONFIG_MODELS))
        game = make_mock_game(game1_map)
        agents = {
            power: make_mock_agent(power, model)
            for power, model in zip(ALL_POWERS, MODELS)
        }

        export_agent_memories(game, agents, tmpdir)

        # Load one memory file and check trust score keys
        memories_dir = os.path.join(tmpdir, "agent_memories")
        grok_file = os.path.join(memories_dir, f"{sanitize_model_id(MODELS[0])}_memory.json")
        with open(grok_file) as f:
            memory = json.load(f)

        trust_keys = set(memory["final_trust_scores"].keys())
        rel_keys = set(memory["final_relationships"].keys())

        # Keys should be model names (e.g., "google/gemma-4-31b-it"), not power names
        for key in trust_keys:
            assert key in MODELS, f"Trust score key '{key}' is not a model name. Keys: {trust_keys}"
        for key in rel_keys:
            assert key in MODELS, f"Relationship key '{key}' is not a model name. Keys: {rel_keys}"

        # Should NOT contain power names
        for power in ALL_POWERS:
            assert power not in trust_keys, f"Trust scores contain power name '{power}'"
            assert power not in rel_keys, f"Relationships contain power name '{power}'"

        print("  PASS: trust scores and relationships are model-keyed")


def test_load_after_rotation():
    """
    Core test: export Game 1 memories, rotate powers, verify each model loads its OWN memory.

    Game 1: AUSTRIA=grok, ENGLAND=gemma, FRANCE=gemini-flash, ...
    Game 2: AUSTRIA=gemma, ENGLAND=gemini-flash, FRANCE=qwen3.5, ... (shift by 1)

    After rotation, grok (now TURKEY) should still load grok's memory from Game 1.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # === GAME 1: Export ===
        game1_map = dict(zip(ALL_POWERS, CONFIG_MODELS))
        game = make_mock_game(game1_map)
        agents = {
            power: make_mock_agent(power, model)
            for power, model in zip(ALL_POWERS, MODELS)
        }
        # Give grok a distinctive diary
        agents["AUSTRIA"].private_diary = ["I am grok and I played Austria in game 1"]

        export_agent_memories(game, agents, tmpdir)
        memories_dir = os.path.join(tmpdir, "agent_memories")

        # === GAME 2: Load with rotation (shift by 1) ===
        rotated_models = MODELS[1:] + MODELS[:1]  # gemma, gemini-flash, qwen3.5, ..., grok
        game2_config_models = [f"openrouter:{m}" for m in rotated_models]
        game2_map = dict(zip(ALL_POWERS, game2_config_models))

        # Simulate what initialize_new_game does: for each power, load memory by client.model_name
        for power, config_model_id in game2_map.items():
            canonical_model = strip_model_prefix(config_model_id)
            memory_data = load_agent_memory(memories_dir, canonical_model)

            if canonical_model == "x-ai/grok-4.1-fast":
                # Grok is now TURKEY (last position after rotation)
                assert power == "TURKEY", f"Expected grok at TURKEY after rotation, got {power}"
                assert memory_data is not None, "Grok should have found its memory file"
                assert "grok" in memory_data["consolidated_diary"].lower(), (
                    f"Grok should get its own diary, got: {memory_data['consolidated_diary'][:80]}"
                )
                assert memory_data["power_name"] == "AUSTRIA", (
                    "Memory should record that grok played AUSTRIA in game 1"
                )
            else:
                assert memory_data is not None, f"Model {canonical_model} should have found its memory"
                assert memory_data["model_id"] == canonical_model, (
                    f"Memory model_id mismatch: expected {canonical_model}, got {memory_data['model_id']}"
                )

        print("  PASS: after rotation, each model loads its own memory")


def test_format_prior_experience_uses_canonical_names():
    """Test that the formatted prompt uses canonical model names matching trust score keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export
        game1_map = dict(zip(ALL_POWERS, CONFIG_MODELS))
        game = make_mock_game(game1_map)
        agents = {
            power: make_mock_agent(power, model)
            for power, model in zip(ALL_POWERS, MODELS)
        }
        export_agent_memories(game, agents, tmpdir)

        # Load and format with a rotated config map
        memories_dir = os.path.join(tmpdir, "agent_memories")
        memory_data = load_agent_memory(memories_dir, MODELS[0])  # grok

        # Simulate the canonical_map construction from initialize_new_game
        rotated_config = dict(zip(ALL_POWERS, CONFIG_MODELS[1:] + CONFIG_MODELS[:1]))
        canonical_map = {p: strip_model_prefix(m) for p, m in rotated_config.items()}

        result = format_prior_experience(memory_data, current_power_model_map=canonical_map)

        # The prompt should contain canonical model names (no "openrouter:" prefix)
        assert "openrouter:" not in result, f"Prompt contains prefixed model ID:\n{result[:500]}"

        # Should reference actual model names
        assert "google/gemma-4-31b-it" in result or "google/gemini-2.5-flash-lite" in result, (
            f"Prompt should reference model names in the current map:\n{result[:500]}"
        )

        # Should contain the "CURRENT game" assignments section
        assert "CURRENT game" in result, f"Prompt missing current game section:\n{result[:500]}"

        print("  PASS: format_prior_experience uses canonical model names")


def test_no_memory_file_returns_none():
    """Test that loading a nonexistent model returns None gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = load_agent_memory(tmpdir, "nonexistent/model")
        assert result is None, f"Expected None for missing model, got {result}"
        print("  PASS: missing memory returns None")


def test_full_roundtrip_3_games():
    """
    Simulate the full 3-game experiment flow:
    Game 1 → export → Game 2 (rotated) → export → Game 3 (rotated again) → load

    Verify memory chains correctly across all 3 games.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        game1_dir = os.path.join(tmpdir, "game1")
        game2_dir = os.path.join(tmpdir, "game2")
        game3_dir = os.path.join(tmpdir, "game3")

        # --- Game 1 (shift=0) ---
        game1_map = dict(zip(ALL_POWERS, CONFIG_MODELS))
        game1 = make_mock_game(game1_map)
        agents1 = {
            power: make_mock_agent(power, model)
            for power, model in zip(ALL_POWERS, MODELS)
        }
        agents1["AUSTRIA"].private_diary = ["Game1: grok dominated the Balkans"]
        export_agent_memories(game1, agents1, game1_dir)

        # --- Game 2 (shift=1): load from game1, then export ---
        rotated1 = MODELS[1:] + MODELS[:1]
        game2_config = dict(zip(ALL_POWERS, [f"openrouter:{m}" for m in rotated1]))
        game2 = make_mock_game(game2_config)
        agents2 = {
            power: make_mock_agent(power, model)
            for power, model in zip(ALL_POWERS, rotated1)
        }

        # Verify load works for game 2
        game1_memories = os.path.join(game1_dir, "agent_memories")
        for power, model in zip(ALL_POWERS, rotated1):
            mem = load_agent_memory(game1_memories, model)
            assert mem is not None, f"Game2: {model} ({power}) couldn't load memory from game1"

        agents2["TURKEY"].private_diary = ["Game2: grok defended Turkey well"]
        export_agent_memories(game2, agents2, game2_dir)

        # --- Game 3 (shift=2): load from game2 ---
        rotated2 = MODELS[2:] + MODELS[:2]
        game2_memories = os.path.join(game2_dir, "agent_memories")

        for power, model in zip(ALL_POWERS, rotated2):
            mem = load_agent_memory(game2_memories, model)
            assert mem is not None, f"Game3: {model} ({power}) couldn't load memory from game2"

        # Grok should get its game2 memory (where it played Turkey)
        grok_mem = load_agent_memory(game2_memories, "x-ai/grok-4.1-fast")
        assert "grok defended Turkey" in grok_mem["consolidated_diary"], (
            f"Grok should have game2 diary, got: {grok_mem['consolidated_diary']}"
        )
        assert grok_mem["power_name"] == "TURKEY", "Grok played Turkey in game 2"

        print("  PASS: full 3-game roundtrip works correctly")


def main():
    print("\nRunning memory roundtrip tests...\n")

    test_strip_model_prefix()
    test_sanitize_model_id()
    test_export_creates_model_keyed_files()
    test_export_trust_scores_are_model_keyed()
    test_load_after_rotation()
    test_format_prior_experience_uses_canonical_names()
    test_no_memory_file_returns_none()
    test_full_roundtrip_3_games()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
