#!/usr/bin/env python3
"""
Persona Analysis: Extract qualitative LLM diplomatic personality characterizations
from AI Diplomacy game data.

Analyzes communication style, strategic behavior, and distinctive patterns
for each model across all experiments.
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

# Seed for reproducible quote selection
random.seed(42)

# Base path
RESULTS_DIR = Path("/Users/I568850/repositories/uni/AI_Diplomacy/results")

# Experiments to analyze
EXPERIMENTS = [
    "3game_experiment",
    "3game_experiment_v2",
    "3game_experiment_v3",
    "3game_experiment_v4",
]

# Model name normalization mapping
MODEL_NAMES = {
    "openrouter:x-ai/grok-4.1-fast": "Grok 4.1 Fast (xAI)",
    "openrouter:google/gemma-4-31b-it": "Gemma 4 31B (Google)",
    "openrouter:qwen/qwen3.5-27b": "Qwen 3.5 27B (Alibaba)",
    "openrouter:qwen/qwen3.6-plus": "Qwen 3.6 Plus (Alibaba)",
    "openrouter:openai/gpt-oss-120b": "GPT-OSS 120B (OpenAI)",
    "openrouter:openai/gpt-5.4": "GPT-5.4 (OpenAI)",
    "openrouter:anthropic/claude-haiku-4.5": "Claude Haiku 4.5 (Anthropic)",
    "openrouter:anthropic/claude-opus-4.6": "Claude Opus 4.6 (Anthropic)",
    "openrouter:google/gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite (Google)",
    "openrouter:deepseek/deepseek-v4-pro": "DeepSeek V4 Pro (DeepSeek)",
}

# Group models that represent the same "slot" across experiments
MODEL_GROUPS = {
    "Grok 4.1 Fast (xAI)": ["openrouter:x-ai/grok-4.1-fast"],
    "Gemma 4 31B (Google)": ["openrouter:google/gemma-4-31b-it"],
    "Qwen 3.5 27B (Alibaba)": ["openrouter:qwen/qwen3.5-27b"],
    "Qwen 3.6 Plus (Alibaba)": ["openrouter:qwen/qwen3.6-plus"],
    "GPT-OSS/GPT-5.4 (OpenAI)": ["openrouter:openai/gpt-oss-120b", "openrouter:openai/gpt-5.4"],
    "Claude Haiku 4.5 / Opus 4.6 (Anthropic)": ["openrouter:anthropic/claude-haiku-4.5", "openrouter:anthropic/claude-opus-4.6"],
    "Gemini 2.5 Flash Lite / DeepSeek V4 Pro": ["openrouter:google/gemini-2.5-flash-lite", "openrouter:deepseek/deepseek-v4-pro"],
}


def load_experiment_data():
    """Load all game data and power→model mappings."""
    all_games = []
    
    for exp_name in EXPERIMENTS:
        exp_dir = RESULTS_DIR / exp_name
        summary_path = exp_dir / "experiment_summary.json"
        
        if not summary_path.exists():
            print(f"WARNING: {summary_path} not found, skipping")
            continue
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        for game_key, game_info in summary["results"].items():
            power_model_map = game_info.get("power_model_map", {})
            if not power_model_map:
                continue  # Skip games without model mapping (e.g., v4/game2)
            
            game_dir = exp_dir / game_key
            game_file = game_dir / "lmvsgame.json"
            
            if not game_file.exists():
                print(f"WARNING: {game_file} not found, skipping")
                continue
            
            with open(game_file) as f:
                game_data = json.load(f)
            
            all_games.append({
                "experiment": exp_name,
                "game": game_key,
                "power_model_map": power_model_map,
                "data": game_data,
            })
    
    return all_games


def extract_messages_by_model(all_games):
    """Extract all messages grouped by model."""
    model_messages = defaultdict(list)
    
    for game in all_games:
        power_model_map = game["power_model_map"]
        
        for phase in game["data"]["phases"]:
            for msg in phase.get("messages", []):
                sender = msg["sender"]
                model_id = power_model_map.get(sender)
                if model_id:
                    model_messages[model_id].append({
                        "message": msg["message"],
                        "recipient": msg["recipient"],
                        "phase": msg["phase"],
                        "sender_power": sender,
                        "experiment": game["experiment"],
                        "game": game["game"],
                    })
    
    return model_messages


def extract_diaries_by_model(all_games):
    """Extract diary entries grouped by model."""
    model_diaries = defaultdict(list)
    
    for game in all_games:
        power_model_map = game["power_model_map"]
        
        for phase in game["data"]["phases"]:
            state_agents = phase.get("state_agents", {})
            for power, agent_data in state_agents.items():
                model_id = power_model_map.get(power)
                if model_id and isinstance(agent_data, dict):
                    diary = agent_data.get("full_private_diary", [])
                    if diary:
                        for entry in diary:
                            model_diaries[model_id].append({
                                "entry": entry,
                                "phase": phase.get("name", ""),
                                "power": power,
                                "experiment": game["experiment"],
                                "game": game["game"],
                            })
    
    return model_diaries


def compute_stats(messages):
    """Compute basic statistics for a model's messages."""
    if not messages:
        return {"total": 0, "avg_length": 0, "private_ratio": 0, "global_ratio": 0}
    
    total = len(messages)
    lengths = [len(m["message"]) for m in messages]
    avg_length = sum(lengths) / total
    
    global_msgs = sum(1 for m in messages if m["recipient"] == "GLOBAL")
    private_msgs = total - global_msgs
    
    # Word counts
    word_counts = [len(m["message"].split()) for m in messages]
    avg_words = sum(word_counts) / total
    
    return {
        "total": total,
        "avg_length_chars": round(avg_length, 1),
        "avg_length_words": round(avg_words, 1),
        "private_count": private_msgs,
        "global_count": global_msgs,
        "private_ratio": round(private_msgs / total, 3),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "median_length": sorted(lengths)[len(lengths) // 2],
    }


def find_representative_quotes(messages, n=5):
    """
    Find representative/interesting quotes that characterize a model's style.
    
    Strategy:
    - Filter out very short or very generic messages
    - Look for messages with distinctive vocabulary or structure
    - Sample from different phases and games for variety
    - Prefer messages that show personality (threats, proposals, reasoning)
    """
    if not messages:
        return []
    
    # Score messages by "interestingness"
    scored = []
    generic_phrases = {
        "let's work together", "i propose", "looking forward",
        "sounds good", "agreed", "thank you", "thanks",
    }
    
    for msg in messages:
        text = msg["message"]
        score = 0
        
        # Length bonus (prefer substantive messages, not too short or too long)
        words = len(text.split())
        if 20 <= words <= 150:
            score += 2
        elif words < 10:
            score -= 3  # Too short, likely generic
        elif words > 200:
            score += 1  # Long but might be interesting
        
        # Penalty for very generic messages
        text_lower = text.lower()
        for phrase in generic_phrases:
            if phrase in text_lower and words < 20:
                score -= 2
        
        # Bonus for specific order notation (shows strategic specificity)
        if any(notation in text for notation in ["->", "- >", "A ", "F ", "BUD", "VIE", "WAR", "MOS"]):
            score += 1
        
        # Bonus for emotional/personality markers
        personality_markers = [
            "betray", "trust", "honest", "warning", "threat", "danger",
            "promise", "guarantee", "suspicious", "concerned", "disappointed",
            "unacceptable", "demand", "insist", "suggest", "propose",
            "unfortunately", "regret", "apologize", "confident",
            "mutual benefit", "long-term", "stability", "peace",
            "aggression", "hostile", "defensive", "offensive",
            "skeptic", "doubt", "question", "verify",
        ]
        for marker in personality_markers:
            if marker in text_lower:
                score += 1
        
        # Bonus for rhetorical questions or conditional language
        if "?" in text:
            score += 1
        if "if " in text_lower or "would you" in text_lower:
            score += 1
        
        # Bonus for unique structural features
        if text.count("\n") > 2:
            score += 1  # Multi-paragraph
        if "1." in text or "- " in text:
            score += 1  # Bullet points/lists
        
        scored.append((score, msg))
    
    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Take top candidates, ensuring variety across games/phases
    selected = []
    seen_phases = set()
    seen_games = set()
    
    for score, msg in scored:
        game_key = f"{msg['experiment']}_{msg['game']}"
        phase_year = msg["phase"][:5] if msg["phase"] else ""
        
        # Prefer diversity
        diversity_bonus = 0
        if game_key not in seen_games:
            diversity_bonus += 1
        if phase_year not in seen_phases:
            diversity_bonus += 1
        
        if len(selected) < n * 3:  # Collect more candidates first
            selected.append((score + diversity_bonus, msg))
            seen_phases.add(phase_year)
            seen_games.add(game_key)
    
    # Final selection - take top n
    selected.sort(key=lambda x: x[0], reverse=True)
    return [msg for _, msg in selected[:n]]


def find_representative_diary_entries(diaries, n=3):
    """Find representative diary entries that show internal reasoning style."""
    if not diaries:
        return []
    
    # Deduplicate diary entries (same text can appear across accumulative phases)
    seen_texts = set()
    unique_diaries = []
    for d in diaries:
        entry = d["entry"]
        if not entry or len(entry) < 30:
            continue
        # Use first 200 chars as dedup key
        key = entry[:200]
        if key not in seen_texts:
            seen_texts.add(key)
            unique_diaries.append(d)
    
    # Score diary entries by interestingness
    scored = []
    for d in unique_diaries:
        entry = d["entry"]
        score = 0
        entry_lower = entry.lower()
        
        # Prefer entries with strategic reasoning
        reasoning_markers = [
            "because", "therefore", "however", "risk", "opportunity",
            "trust", "betray", "alliance", "threat", "plan",
            "priority", "objective", "strategy", "concern",
        ]
        for marker in reasoning_markers:
            if marker in entry_lower:
                score += 1
        
        # Prefer medium-length entries
        if 100 <= len(entry) <= 600:
            score += 2
        elif 50 <= len(entry) <= 100:
            score += 1
        
        # Bonus for entries that discuss multiple actors
        powers = ["england", "france", "germany", "austria", "italy", "russia", "turkey"]
        mentioned = sum(1 for p in powers if p in entry_lower)
        if mentioned >= 2:
            score += 1
        
        scored.append((score, d))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Ensure variety across games
    selected = []
    seen_games = set()
    for score, d in scored:
        game_key = d["experiment"] + "_" + d["game"]
        if game_key not in seen_games or len(selected) < n:
            selected.append(d)
            seen_games.add(game_key)
        if len(selected) >= n:
            break
    
    return selected


def analyze_communication_patterns(messages):
    """Analyze specific communication patterns."""
    if not messages:
        return {}
    
    patterns = {
        "uses_order_notation": 0,
        "asks_questions": 0,
        "uses_lists": 0,
        "makes_threats": 0,
        "proposes_alliances": 0,
        "expresses_suspicion": 0,
        "uses_formal_language": 0,
        "multi_paragraph": 0,
        "very_short_msgs": 0,  # < 20 words
        "very_long_msgs": 0,   # > 100 words
    }
    
    for msg in messages:
        text = msg["message"]
        text_lower = text.lower()
        words = len(text.split())
        
        # Order notation (move commands like A VIE-BUD)
        if any(x in text for x in [" -> ", "A ", "F ", "->"]):
            # More specific check
            import re
            if re.search(r'[AF]\s+[A-Z]{3}', text):
                patterns["uses_order_notation"] += 1
        
        if "?" in text:
            patterns["asks_questions"] += 1
        
        if "1." in text or "- " in text or "•" in text:
            patterns["uses_lists"] += 1
        
        threat_words = ["warn", "threat", "consequence", "force", "attack you", "retaliate"]
        if any(w in text_lower for w in threat_words):
            patterns["makes_threats"] += 1
        
        alliance_words = ["alliance", "ally", "partner", "together", "cooperat", "team up"]
        if any(w in text_lower for w in alliance_words):
            patterns["proposes_alliances"] += 1
        
        suspicion_words = ["suspicious", "skeptic", "doubt", "trust", "verify", "concern", "worried"]
        if any(w in text_lower for w in suspicion_words):
            patterns["expresses_suspicion"] += 1
        
        formal_words = ["propose", "shall", "hereby", "regarding", "with respect to", "furthermore"]
        if any(w in text_lower for w in formal_words):
            patterns["uses_formal_language"] += 1
        
        if text.count("\n") >= 2:
            patterns["multi_paragraph"] += 1
        
        if words < 20:
            patterns["very_short_msgs"] += 1
        if words > 100:
            patterns["very_long_msgs"] += 1
    
    # Normalize to percentages
    total = len(messages)
    patterns_pct = {k: round(v / total * 100, 1) for k, v in patterns.items()}
    patterns_pct["_raw"] = patterns
    
    return patterns_pct


def print_report(model_group_name, model_ids, model_messages, model_diaries):
    """Print formatted report for a model group."""
    # Aggregate messages across model IDs in this group
    all_msgs = []
    all_diary = []
    for mid in model_ids:
        all_msgs.extend(model_messages.get(mid, []))
        all_diary.extend(model_diaries.get(mid, []))
    
    if not all_msgs:
        print(f"\n{'='*80}")
        print(f"  {model_group_name}")
        print(f"{'='*80}")
        print("  NO DATA FOUND")
        return
    
    stats = compute_stats(all_msgs)
    patterns = analyze_communication_patterns(all_msgs)
    quotes = find_representative_quotes(all_msgs, n=5)
    diary_entries = find_representative_diary_entries(all_diary, n=3)
    
    print(f"\n{'='*80}")
    print(f"  {model_group_name}")
    print(f"{'='*80}")
    
    # Model IDs used
    used_ids = [mid for mid in model_ids if model_messages.get(mid)]
    print(f"\n  Model IDs: {', '.join(MODEL_NAMES.get(m, m) for m in used_ids)}")
    games_played = len(set(m["experiment"] + "_" + m["game"] for m in all_msgs))
    print(f"  Games played: {games_played}")
    
    # Basic stats
    print(f"\n  --- BASIC STATS ---")
    print(f"  Total messages sent: {stats['total']}")
    print(f"  Avg message length: {stats['avg_length_words']} words ({stats['avg_length_chars']} chars)")
    print(f"  Private messages: {stats['private_count']} ({stats['private_ratio']*100:.1f}%)")
    print(f"  Global messages: {stats['global_count']} ({(1-stats['private_ratio'])*100:.1f}%)")
    print(f"  Message length range: {stats['min_length']}-{stats['max_length']} chars (median: {stats['median_length']})")
    
    # Communication patterns
    print(f"\n  --- COMMUNICATION PATTERNS ---")
    print(f"  Uses specific order notation: {patterns.get('uses_order_notation', 0)}% of messages")
    print(f"  Asks questions: {patterns.get('asks_questions', 0)}% of messages")
    print(f"  Uses lists/structure: {patterns.get('uses_lists', 0)}% of messages")
    print(f"  Makes threats/warnings: {patterns.get('makes_threats', 0)}% of messages")
    print(f"  Proposes alliances: {patterns.get('proposes_alliances', 0)}% of messages")
    print(f"  Expresses suspicion/trust concerns: {patterns.get('expresses_suspicion', 0)}% of messages")
    print(f"  Formal language: {patterns.get('uses_formal_language', 0)}% of messages")
    print(f"  Multi-paragraph messages: {patterns.get('multi_paragraph', 0)}% of messages")
    print(f"  Very short (<20 words): {patterns.get('very_short_msgs', 0)}% | Very long (>100 words): {patterns.get('very_long_msgs', 0)}%")
    
    # Representative quotes
    print(f"\n  --- REPRESENTATIVE QUOTES ---")
    for i, q in enumerate(quotes, 1):
        text = q["message"]
        # Truncate very long quotes for readability
        if len(text) > 400:
            text = text[:397] + "..."
        # Clean up newlines for display
        text = text.replace("\n", " | ")
        print(f"\n  [{i}] Phase: {q['phase']}, To: {q['recipient']}, As: {q['sender_power']}")
        print(f"      \"{text}\"")
    
    # Diary entries (internal reasoning)
    if diary_entries:
        print(f"\n  --- INTERNAL REASONING (from diaries) ---")
        for i, d in enumerate(diary_entries, 1):
            entry = d["entry"]
            if len(entry) > 400:
                entry = entry[:397] + "..."
            entry = entry.replace("\n", " | ")
            print(f"\n  [{i}] Phase: {d['phase']}, As: {d['power']}")
            print(f"      \"{entry}\"")
    
    print()


def generate_characterization(model_group_name, model_ids, model_messages, model_diaries):
    """Generate a brief characterization based on data analysis."""
    all_msgs = []
    for mid in model_ids:
        all_msgs.extend(model_messages.get(mid, []))
    
    if not all_msgs:
        return "No data available."
    
    stats = compute_stats(all_msgs)
    patterns = analyze_communication_patterns(all_msgs)
    
    descriptors = []
    
    # Verbosity
    if stats["avg_length_words"] > 70:
        descriptors.append("highly verbose")
    elif stats["avg_length_words"] > 45:
        descriptors.append("moderately verbose")
    elif stats["avg_length_words"] < 25:
        descriptors.append("terse and concise")
    else:
        descriptors.append("moderate in message length")
    
    # Communication style
    if patterns.get("uses_order_notation", 0) > 30:
        descriptors.append("frequently shares specific order plans")
    if patterns.get("asks_questions", 0) > 40:
        descriptors.append("highly interrogative")
    if patterns.get("makes_threats", 0) > 10:
        descriptors.append("occasionally threatening")
    if patterns.get("proposes_alliances", 0) > 40:
        descriptors.append("strongly alliance-oriented")
    elif patterns.get("proposes_alliances", 0) > 25:
        descriptors.append("cooperative-minded")
    if patterns.get("expresses_suspicion", 0) > 15:
        descriptors.append("skeptical and trust-focused")
    if patterns.get("uses_formal_language", 0) > 30:
        descriptors.append("formal in tone")
    if patterns.get("multi_paragraph", 0) > 30:
        descriptors.append("writes structured multi-paragraph messages")
    if patterns.get("very_short_msgs", 0) > 40:
        descriptors.append("often sends brief tactical messages")
    
    # Private vs global
    if stats["private_ratio"] > 0.9:
        descriptors.append("almost exclusively uses private channels")
    elif stats["private_ratio"] < 0.7:
        descriptors.append("more willing to use global broadcasts")
    
    return "; ".join(descriptors) + "."


def main():
    print("=" * 80)
    print("  AI DIPLOMACY - LLM PERSONA ANALYSIS")
    print("  Extracting qualitative characterizations from game data")
    print("=" * 80)
    
    # Load data
    print("\nLoading game data...")
    all_games = load_experiment_data()
    print(f"  Loaded {len(all_games)} games across {len(EXPERIMENTS)} experiments")
    
    # Extract messages and diaries
    print("Extracting messages...")
    model_messages = extract_messages_by_model(all_games)
    print(f"  Found messages from {len(model_messages)} distinct model IDs")
    for mid, msgs in sorted(model_messages.items(), key=lambda x: -len(x[1])):
        print(f"    {MODEL_NAMES.get(mid, mid)}: {len(msgs)} messages")
    
    print("\nExtracting diary entries...")
    model_diaries = extract_diaries_by_model(all_games)
    for mid, entries in sorted(model_diaries.items(), key=lambda x: -len(x[1])):
        print(f"    {MODEL_NAMES.get(mid, mid)}: {len(entries)} diary entries")
    
    # Generate reports per model group
    print("\n\n" + "#" * 80)
    print("#" + " " * 26 + "MODEL PERSONA REPORTS" + " " * 31 + "#")
    print("#" * 80)
    
    for group_name, model_ids in MODEL_GROUPS.items():
        print_report(group_name, model_ids, model_messages, model_diaries)
        
        # Print characterization
        char = generate_characterization(group_name, model_ids, model_messages, model_diaries)
        print(f"  --- AUTO-CHARACTERIZATION ---")
        print(f"  {char}")
        print(f"\n{'─'*80}")
    
    # Summary comparison table
    print("\n\n" + "=" * 80)
    print("  COMPARATIVE SUMMARY")
    print("=" * 80)
    print(f"\n  {'Model':<45} {'Msgs':>5} {'AvgWords':>8} {'Priv%':>6} {'Alliance%':>9} {'Suspic%':>7}")
    print(f"  {'─'*45} {'─'*5} {'─'*8} {'─'*6} {'─'*9} {'─'*7}")
    
    for group_name, model_ids in MODEL_GROUPS.items():
        all_msgs = []
        for mid in model_ids:
            all_msgs.extend(model_messages.get(mid, []))
        
        if not all_msgs:
            continue
        
        stats = compute_stats(all_msgs)
        patterns = analyze_communication_patterns(all_msgs)
        
        print(f"  {group_name:<45} {stats['total']:>5} {stats['avg_length_words']:>8.1f} {stats['private_ratio']*100:>5.1f}% {patterns.get('proposes_alliances', 0):>8.1f}% {patterns.get('expresses_suspicion', 0):>6.1f}%")


if __name__ == "__main__":
    main()
