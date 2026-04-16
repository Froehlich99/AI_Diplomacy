#!/usr/bin/env python3
"""
Generate dummy cross-game memory files for testing the memory import flow.

Usage:
    python create_test_memory.py [--output_dir OUTPUT_DIR]

This creates agent_memories/{POWER}_memory.json files that can be passed
to lm_game.py via --prior_memory_dir to verify the injection works.

Example workflow:
    # 1. Generate test memories
    python create_test_memory.py --output_dir ./test_memories

    # 2. Run a game with those memories injected
    python lm_game.py --max_year 1902 --models <model> --simple_prompts True \
        --prior_memory_dir ./test_memories/agent_memories/

    # 3. Check the logs to verify "PRIOR GAME EXPERIENCE" appears in prompts
    grep "Prior game experience" ./results/<timestamp>/general_game.log
"""

import argparse
import json
import os

ALL_POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

# Dummy diary content for each power
DUMMY_DIARIES = {
    "AUSTRIA": (
        "[CONSOLIDATED HISTORY] In the previous game, Austria focused on securing the Balkans "
        "early. A strong alliance with Italy helped defend against a Turkish assault in 1902. "
        "Russia proved untrustworthy, breaking a non-aggression pact in Fall 1903 by moving "
        "into Galicia. Germany remained neutral throughout. Key lesson: never trust verbal "
        "agreements with Russia without verifiable action. The Lepanto opening with Italy "
        "was highly effective against Turkey."
    ),
    "ENGLAND": (
        "[CONSOLIDATED HISTORY] England played a naval-focused game, securing the North Sea "
        "and Norwegian Sea early. An alliance with France against Germany worked well in "
        "1901-1902 but France betrayed the alliance in 1903 by moving into the English Channel. "
        "Key lesson: always keep a fleet in the English Channel as insurance. Germany can be a "
        "useful ally against France if approached early."
    ),
    "FRANCE": (
        "[CONSOLIDATED HISTORY] France attempted to balance between England and Germany. "
        "The initial pact with England worked until mid-game when territorial disputes arose. "
        "Italy was a reliable ally throughout. Key lesson: prioritize securing Iberia early "
        "and maintain a defensive posture toward England until the mid-game."
    ),
    "GERMANY": (
        "[CONSOLIDATED HISTORY] Germany was caught in a two-front war between France and Russia. "
        "An early alliance with England provided some relief but England was unreliable. "
        "Key lesson: Germany must secure either France or Russia as an ally immediately "
        "and cannot afford to fight on two fronts."
    ),
    "ITALY": (
        "[CONSOLIDATED HISTORY] Italy executed a Lepanto opening with Austria against Turkey. "
        "This was highly successful, taking Turkey out by 1905. However, Austria then turned "
        "on Italy. Key lesson: after defeating Turkey, immediately pivot to defensive positioning "
        "against your former ally."
    ),
    "RUSSIA": (
        "[CONSOLIDATED HISTORY] Russia expanded aggressively southward, taking the Black Sea "
        "and pressuring Turkey. Relations with Austria deteriorated when both competed for "
        "the Balkans. Key lesson: the Juggernaut alliance (Russia+Turkey) is powerful but "
        "difficult to maintain. Northern defense against England must not be neglected."
    ),
    "TURKEY": (
        "[CONSOLIDATED HISTORY] Turkey was eliminated early due to a coordinated attack from "
        "Austria, Italy, and Russia. The Lepanto opening was devastating. Key lesson: Turkey "
        "must break up the Austria-Italy alliance early through diplomacy, or secure a strong "
        "alliance with Russia to counterbalance."
    ),
}

DUMMY_RELATIONSHIPS = {
    "AUSTRIA": {"ENGLAND": "Neutral", "FRANCE": "Neutral", "GERMANY": "Friendly", "ITALY": "Ally", "RUSSIA": "Enemy", "TURKEY": "Unfriendly"},
    "ENGLAND": {"AUSTRIA": "Neutral", "FRANCE": "Enemy", "GERMANY": "Friendly", "ITALY": "Neutral", "RUSSIA": "Neutral", "TURKEY": "Neutral"},
    "FRANCE": {"AUSTRIA": "Neutral", "ENGLAND": "Unfriendly", "GERMANY": "Enemy", "ITALY": "Ally", "RUSSIA": "Neutral", "TURKEY": "Neutral"},
    "GERMANY": {"AUSTRIA": "Friendly", "ENGLAND": "Unfriendly", "FRANCE": "Enemy", "ITALY": "Neutral", "RUSSIA": "Enemy", "TURKEY": "Neutral"},
    "ITALY": {"AUSTRIA": "Unfriendly", "ENGLAND": "Neutral", "FRANCE": "Friendly", "GERMANY": "Neutral", "RUSSIA": "Neutral", "TURKEY": "Enemy"},
    "RUSSIA": {"AUSTRIA": "Unfriendly", "ENGLAND": "Neutral", "FRANCE": "Neutral", "GERMANY": "Enemy", "ITALY": "Neutral", "TURKEY": "Friendly"},
    "TURKEY": {"AUSTRIA": "Enemy", "ENGLAND": "Neutral", "FRANCE": "Neutral", "GERMANY": "Neutral", "ITALY": "Enemy", "RUSSIA": "Friendly"},
}

DUMMY_SC_COUNTS = {
    "AUSTRIA": 6, "ENGLAND": 5, "FRANCE": 7, "GERMANY": 4,
    "ITALY": 5, "RUSSIA": 6, "TURKEY": 0,
}


def main():
    parser = argparse.ArgumentParser(description="Generate test cross-game memory files.")
    parser.add_argument("--output_dir", type=str, default="./test_memories",
                        help="Output directory (agent_memories/ will be created inside).")
    args = parser.parse_args()

    memories_dir = os.path.join(args.output_dir, "agent_memories")
    os.makedirs(memories_dir, exist_ok=True)

    for power in ALL_POWERS:
        sc_count = DUMMY_SC_COUNTS[power]
        survived = sc_count > 0
        memory = {
            "power_name": power,
            "model_id": "test-model",
            "game_outcome": {
                "survived": survived,
                "won": False,
                "draw": survived,  # all survivors drew
                "final_supply_center_count": sc_count,
                "final_year": "1908",
            },
            "consolidated_diary": DUMMY_DIARIES[power],
            "final_relationships": DUMMY_RELATIONSHIPS[power],
            "final_goals": [f"Defend home centers", f"Expand influence in neighboring regions"],
        }

        file_path = os.path.join(memories_dir, f"{power}_memory.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
        print(f"  Created {file_path}")

    print(f"\nDone. Test memories written to {memories_dir}/")
    print(f"\nTo use them:")
    print(f"  python lm_game.py --max_year 1902 --models <model> \\")
    print(f"    --prior_memory_dir {memories_dir}/")


if __name__ == "__main__":
    main()
