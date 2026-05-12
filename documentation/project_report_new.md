# Multi-Agent Gaming with the Board Game Diplomacy

## Application area and goals

### Motivation

- Why Diplomacy etc.

### Brief Introduction to Diplomacy

## Profile of Data (new name!)

### how we generated the gameplay data

### what data we captured

| Aspect         | What to report                                                                                                                                                                   |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Structure      | The Imsgame. json schema: phases → istate, messages, orders, state_agents}. The nested structure of agent state (diary entries, trust scores, relationships, goals).             |
| Size           | Number of games, phases per game (~20 movement phases), messages per phase, orders per phase. Total token counts (you have token usage logs). File sizes.                        |
| Data types     | Board state (structured: units/centers dicts), messages (free-text, sender/recipient), orders (domain-specific syntax), diary ( free-text reflections), trust scores (float 0-1) |
| Volume metrics | e.g., "Across 12 games, we generated ~2,400 movement phases, ~8, 000 negotiation messages, ~16,800 orders, and ~4,800 diary entries"                                             |

## Approach


### Agent Architecture / Game Setup (name?)

- Based on the AIDiplomacy repo (source)
- Modified to fit our use case... (check git)
- Context of each agent
  - Diary
  - Current game state
  - ...

- Diary Compaction approach
  - Why use Compaction after each phase instead of "auto compaction" or trimming

- switch from memory-per-power to memory-per-model (completely done intentionally of course)
- learning: reducing the thinking budgets to reduce the duration of game runs

- Practical setup with K8s and B2 => maybe different section?

### Model selection

- orientation on food truck (https://foodtruckbench.com/#leaderboard) and one other that sounds plausible

### Negotiation structure

### Analysis Methodology
- LLM as judge deception
- trust score
- behavioral metrics
- maybe qualitative analysis??

## Evaluation

### Error Analysis

### Quantitative Results

- E.g. message volume, attacking percentage, betrayal moves, correlation to Supply centers
- If we use ipynb notebooks, rerun them first and make sure they use the updated data/latest games!

### Behavioral Analysis

- trust dynamics
- deception
- model personas

- (fetch information from the games; show direct quotes; search for interesting cases; validate with numbers where possible)
- e.g. GPT-OSS 120B very passive; only attacked in x% of its turns
- which other models?


### Problems that we faced

## Conclusion
