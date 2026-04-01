# AI Diplomacy

LLM agents playing the board game [Diplomacy](https://en.wikipedia.org/wiki/Diplomacy_(game)) against each other. Each of the 7 powers (Austria, England, France, Germany, Italy, Russia, Turkey) is controlled by an LLM that negotiates, plans, and submits orders every phase.

Forked from [GoodStartLabs/AI_Diplomacy](https://github.com/GoodStartLabs/AI_Diplomacy) and stripped down to the core game engine + agent system.

## Setup

```bash
# Install dependencies (requires Python 3.13+)
uv sync

# Create .env with your API keys
cat > .env << 'EOF'
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
OPENROUTER_API_KEY=sk-or-...
# DEEPSEEK_API_KEY=...
# TOGETHER_API_KEY=...
EOF
```

## Usage

```bash
# Run a game with default model assignments
python lm_game.py

# Specify models for each power (order: Austria, England, France, Germany, Italy, Russia, Turkey)
python lm_game.py --models "gpt-4o,claude-sonnet-4-20250514,gpt-4o,gpt-4o,gpt-4o,gpt-4o,gpt-4o"

# Enable negotiations (2 rounds per movement phase)
python lm_game.py --num_negotiation_rounds 2

# Enable strategic planning phase
python lm_game.py --planning_phase

# Stop at a specific year
python lm_game.py --max_year 1910

# Resume a previous game
python lm_game.py --run_dir results/20260401_120000
```

### Supported LLM providers

Models are specified as `<prefix>:<model_id>` or just the model name for common ones:

| Provider | Prefix | Example |
|----------|--------|---------|
| OpenAI | `openai:` | `gpt-4o`, `o3`, `o4-mini` |
| Anthropic | `anthropic:` | `claude-sonnet-4-20250514`, `claude-opus-4-20250514` |
| Google Gemini | `gemini:` | `gemini-2.5-flash-preview-05-20` |
| DeepSeek | `deepseek:` | `deepseek-chat` |
| OpenRouter | `openrouter-` | `openrouter-google/gemini-2.5-flash-preview-05-20` |
| Together | `together:` | `together:meta-llama/Llama-3-70b` |

## Project structure

```
lm_game.py          Main game loop
config.py           Configuration (reads from .env)
models.py           Power enum definitions

ai_diplomacy/       LLM agent system
  agent.py          DiplomacyAgent — stateful agent with goals, relationships, diary
  clients.py        LLM provider clients (OpenAI, Anthropic, Gemini, DeepSeek, OpenRouter, Together)
  negotiations.py   Multi-agent negotiation rounds
  planning.py       Strategic planning phase
  game_logic.py     Game state save/load/resume
  game_history.py   Phase-by-phase event tracking
  prompt_constructor.py   Prompt assembly from templates
  possible_order_context.py   Legal move context generation
  initialization.py Agent goal/relationship initialization
  diary_logic.py    Diary entry consolidation
  narrative.py      Phase summary generation
  formatter.py      Two-step LLM formatting (optional)
  utils.py          Logging, order validation, prompt loading
  prompts/          Prompt templates

diplomacy/          Game engine (from github.com/diplomacy/diplomacy)
  engine/           Core: game state, map, powers, messages, rendering
  utils/            Engine utilities
  maps/             Map definitions (standard + variants)
```
