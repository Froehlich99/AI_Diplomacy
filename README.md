# AI Diplomacy

LLM agents playing the board game [Diplomacy](https://en.wikipedia.org/wiki/Diplomacy_(game)) against each other. Each of the 7 powers (Austria, England, France, Germany, Italy, Russia, Turkey) is controlled by an LLM that negotiates, plans, and submits orders every phase.

The project investigates whether LLM agents improve their performance across sequential games using **cross-game memory**, **numeric trust scores**, and **strategic adaptation**. A deception analysis pipeline compares agents' private reasoning against their public communication to measure honesty.

Forked from [GoodStartLabs/AI_Diplomacy](https://github.com/GoodStartLabs/AI_Diplomacy).

## Setup

```bash
# Install dependencies (requires Python 3.13+)
uv sync

# Create .env with your API key
echo "OPENROUTER_API_KEY=sk-or-v1-your-key-here" > .env
```

## Quick Start

```bash
# Run a single game (all 7 powers use the same model)
python lm_game.py --max_year 1910 --models "openrouter:x-ai/grok-4.1-fast"

# Run with 7 different models (order: Austria, England, France, Germany, Italy, Russia, Turkey)
python lm_game.py --max_year 1910 --models \
  "openrouter:x-ai/grok-4.1-fast,openrouter:google/gemini-2.5-flash-lite,openrouter:deepseek/deepseek-v3.2,openrouter:z-ai/glm-4.7-flash,openrouter:xiaomi/mimo-v2-flash,openrouter:qwen/qwen-2.5-7b-instruct,openrouter:mistralai/mistral-small-3.2-24b-instruct"

# Enable negotiations (1 round per movement phase)
python lm_game.py --max_year 1910 --models "openrouter:x-ai/grok-4.1-fast" \
  --num_negotiation_rounds 1
```

Results are saved to `results/<timestamp>/` including game state, logs, and agent memories.

## Cross-Game Memory

After each game, every agent's diary, relationships, trust scores, and goals are exported to `agent_memories/`. These can be injected into the next game's system prompt so agents learn from prior experience.

```bash
# Game 1: produces memory exports
python lm_game.py --max_year 1910 --models "openrouter:x-ai/grok-4.1-fast" \
  --run_dir results/game1

# Game 2: agents start with Game 1's experience
python lm_game.py --max_year 1910 --models "openrouter:x-ai/grok-4.1-fast" \
  --run_dir results/game2 \
  --prior_memory_dir results/game1/agent_memories/

# Game 3: chain further
python lm_game.py --max_year 1910 --models "openrouter:x-ai/grok-4.1-fast" \
  --run_dir results/game3 \
  --prior_memory_dir results/game2/agent_memories/
```

Memory files are JSON containing the consolidated diary (token-capped via `--memory_cap_words`, default 500), final relationships, numeric trust scores, goals, and game outcome.

For quick testing without running a full game, generate dummy memory files:

```bash
python create_test_memory.py --output_dir ./test_memories
python lm_game.py --max_year 1901 --models "openrouter:x-ai/grok-4.1-fast" \
  --prior_memory_dir ./test_memories/agent_memories/
```

## Experiment Runner

For multi-game experiments with memory chaining, power rotation, and baseline comparisons, use `experiment_runner.py` with a Python config file.

### Config Format

```python
# experiment_configs/my_experiment.py
EXPERIMENT = {
    "name": "my_experiment",
    "max_year": 1910,
    "num_negotiation_rounds": 1,
    "simple_prompts": True,
    "memory_cap_words": 500,

    # 7 models, one per power
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

    "games": [
        # Memory series: each game feeds memory to the next
        {"id": "mem_game1", "memory": False},   # no prior memory
        {"id": "mem_game2", "memory": True},     # inherits from mem_game1
        {"id": "mem_game3", "memory": True},     # inherits from mem_game2
        # Baseline series: no memory injection
        {"id": "baseline1", "memory": False},
        {"id": "baseline2", "memory": False},
        {"id": "baseline3", "memory": False},
    ],
}
```

### Running

```bash
# Run the experiment
python experiment_runner.py experiment_configs/my_experiment.py

# Custom output directory
python experiment_runner.py experiment_configs/my_experiment.py \
  --output_dir results/my_run
```

### Features

- **Power rotation**: Models shift positions each game (circular shift) to reduce positional bias.
- **Memory chaining**: When `memory: True`, the previous game's `agent_memories/` is automatically passed to the next game.
- **Crash recovery**: Progress is saved to `experiment_state.json` after each game. Re-running the same command skips completed games.
- **Baseline mode**: Games with `memory: False` run without memory injection, providing a control group.

### Output

```
results/<experiment_name>/
  experiment_config.json      # Copy of the config used
  experiment_state.json       # Progress tracker (for crash recovery)
  experiment_summary.json     # Final results: SC counts, winners, per game
  mem_game1/
    lmvsgame.json             # Full game state with per-phase snapshots
    general_game.log          # Game log
    llm_responses.csv         # All LLM calls (prompt + response)
    overview.jsonl            # Error stats, power-model map, config
    agent_memories/           # Exported memories for next game
  mem_game2/
    ...
```

## Deception Analysis

Post-game tool that compares each agent's **private diary** (internal reasoning) against their **public messages** (what they told other players) and **actual orders** (what they did). Uses an LLM to classify behavior on the Honorable-Deceitful axis.

```bash
# Analyze a single game
python deception_analyzer.py results/game1/lmvsgame.json \
  --model "openrouter:x-ai/grok-4.1-fast"

# Analyze all games in an experiment
python deception_analyzer.py results/my_experiment/ \
  --model "openrouter:x-ai/grok-4.1-fast"
```

Outputs `deception_analysis.json` (nested by game/phase/power) and `deception_scores.csv` with columns: `game_id, phase, power, model_id, deception_score, classification, evidence, broken_promises, kept_promises`.

Only movement phases where a power sent messages are analyzed (no messages = no opportunity for deception).

## Evaluation & Visualization

Generate quantitative metrics and publication-ready plots from experiment data.

```bash
python evaluate.py results/my_experiment/
```

### Metrics

| Metric | Description |
|--------|-------------|
| **Improvement Delta** | Performance shift between initial and later games (memory effect) |
| **Memory-Induced Success Rate** | Win/survival rate: memory games vs. baseline |
| **SC Trajectories** | Supply center count per power over time |
| **Order Success Rate** | Fraction of orders that succeeded per power |
| **Trust Dynamics** | Trust score evolution and final trust matrices |
| **Deception Scores** | Mean deception per model/game type (if deception analysis was run) |

### Plots (saved as PDF + PNG at 300 DPI)

1. SC trajectory line charts (per game)
2. Memory vs. baseline bar chart (Improvement Delta)
3. Win/survival heatmap (model x power)
4. Trust score evolution over phases
5. Trust matrix heatmap (7x7, per game)
6. Deception score distribution (requires deception analysis)
7. Deception vs. SC correlation scatter (requires deception analysis)
8. Relationship evolution stacked area charts

Output is saved to `results/<experiment>/evaluation/`.

## CLI Reference

### lm_game.py

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | `""` | Comma-separated model IDs (1 for all powers, or 7 for each) |
| `--max_year` | required | Maximum game year (e.g., 1910) |
| `--run_dir` | auto timestamp | Output directory (resumes if exists) |
| `--num_negotiation_rounds` | `0` | Negotiation rounds per movement phase |
| `--simple_prompts` | `True` | Use simplified prompts (recommended for most models) |
| `--planning_phase` | `False` | Enable strategic planning before orders |
| `--prior_memory_dir` | `""` | Path to prior game's `agent_memories/` directory |
| `--memory_cap_words` | `500` | Max words in exported diary memory |
| `--max_tokens` | `16000` | Max tokens per LLM call |
| `--end_at_phase` | `""` | Stop at specific phase (e.g., `S1905M`) |
| `--resume_from_phase` | `""` | Resume from phase (clears later data) |

### Supported LLM Providers

Models are specified as `<prefix>:<model_id>` or via heuristic matching:

| Provider | Format | Example |
|----------|--------|---------|
| OpenAI | `openai:` or bare | `gpt-4o`, `o4-mini` |
| Anthropic | `anthropic:` or bare | `claude-sonnet-4-20250514` |
| Google Gemini | `gemini:` | `gemini:gemini-2.5-flash` |
| DeepSeek | `deepseek:` | `deepseek:deepseek-chat` |
| OpenRouter | `openrouter:` | `openrouter:x-ai/grok-4.1-fast` |
| Together | `together:` | `together:meta-llama/Llama-3-70b` |

OpenRouter gives access to 200+ models via a single API key.

## Agent Architecture

Each power is controlled by a `DiplomacyAgent` that maintains:

- **Strategic goals**: Updated every phase via LLM
- **Relationships**: Categorical labels (Enemy / Unfriendly / Neutral / Friendly / Ally)
- **Trust scores**: Numeric 0.0-1.0 per opponent, updated every phase
- **Private diary**: Phase-prefixed reflections, consolidated every Spring to manage context

### Game Phase Pipeline

Each movement phase runs this pipeline for all active powers:

1. **Negotiation**: Round-robin message exchange (private + global)
2. **Diary update**: Summarize negotiations, update relationships/trust/goals
3. **Order generation**: LLM produces orders given full context (board state, diary, goals, trust)
4. **Phase resolution**: Engine resolves orders simultaneously
5. **Post-phase diary**: Analyze outcomes, detect betrayals
6. **Consolidation** (Spring only): LLM rewrites diary to keep context bounded

## Project Structure

```
lm_game.py                  CLI wrapper for single games
experiment_runner.py         Multi-game experiment orchestrator
deception_analyzer.py        Post-game deception analysis
evaluate.py                  Metrics computation + visualization
create_test_memory.py        Generate dummy memory files for testing
config.py                    Configuration (reads from .env)

ai_diplomacy/                LLM agent system
  game_runner.py             Core game loop (run_single_game)
  agent.py                   DiplomacyAgent with goals, relationships, trust, diary
  clients.py                 LLM provider clients (OpenAI, Anthropic, Gemini, etc.)
  negotiations.py            Multi-agent negotiation rounds
  planning.py                Strategic planning phase
  game_logic.py              Save/load/resume + memory export/import
  game_history.py            Phase-by-phase event tracking
  prompt_constructor.py      Prompt assembly from templates
  possible_order_context.py  Legal move context with BFS pathfinding
  initialization.py          Agent state initialization
  diary_logic.py             Diary consolidation
  utils.py                   Logging, order validation, prompt loading
  prompts_simple/            Prompt templates (simplified)
  prompts/                   Prompt templates (full)

experiment_configs/          Experiment configuration files
documentation/               Project outline and report (LaTeX)
diplomacy/                   Game engine (from diplomacy library)
```
