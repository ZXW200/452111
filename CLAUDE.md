# CLAUDE.md - AI Assistant Guide for Game Theory LLM Research

## Project Overview

This is a **Game Theory LLM Multi-Agent Research** platform for studying how large language models (LLMs) behave as agents in repeated game theory scenarios. The system compares LLM decision-making strategies against classical game theory strategies (e.g., Tit-for-Tat) across various game types.

**Primary Research Questions:**
- How do LLMs behave in iterated game theory games?
- How does memory/history window size affect cooperation?
- How do different LLM providers (DeepSeek, OpenAI, Claude) compare?
- What is the effect of "Cheap Talk" (language-based communication)?
- How do multi-agent group dynamics emerge?

## Quick Commands

```bash
# Install dependencies
pip install numpy matplotlib requests

# Configure API keys (interactive setup)
python game_theory/llm_api.py setup

# Or set environment variables
export DEEPSEEK_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# Test API connectivity
python testapi

# Run experiments
python run_research_experiment.py all                    # Run all experiments
python run_research_experiment.py pure_hybrid            # Pure vs Hybrid mode
python run_research_experiment.py baseline               # LLM vs classical strategies
python run_research_experiment.py multi_llm              # Provider comparison
python run_research_experiment.py window                 # Memory window comparison
python run_research_experiment.py cheap_talk             # Language communication
python run_research_experiment.py group                  # Multi-agent dynamics
python run_research_experiment.py group_multi            # Multi-provider groups

# With options
python run_research_experiment.py all --provider deepseek --repeats 5 --rounds 30
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                  run_research_experiment.py                      │
│            (Experiment Orchestration & Analysis)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      game_theory/                                │
├─────────────┬─────────────┬─────────────┬─────────────────────────┤
│  games.py   │strategies.py│  network.py │    simulation.py       │
│  (Games &   │(Classical   │(Interaction │   (Core Engine)        │
│  Payoffs)   │ Strategies) │ Topologies) │                        │
├─────────────┴─────────────┴─────────────┴─────────────────────────┤
│                      llm_strategy.py                             │
│         (LLM-based Strategy + Response Parsing)                  │
├─────────────────────────────────────────────────────────────────┤
│                        llm_api.py                                │
│        (Unified LLM Provider Interface)                          │
│    DeepSeek | OpenAI | Anthropic Claude | Ollama                 │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
/
├── game_theory/                  # Core simulation module
│   ├── __init__.py              # Public API exports
│   ├── games.py                 # Game definitions (PD, Snowdrift, Stag Hunt, Harmony)
│   ├── strategies.py            # Classical strategies (TitForTat, Pavlov, etc.)
│   ├── llm_api.py              # LLM provider abstraction layer
│   ├── llm_strategy.py         # LLM-based strategy implementation
│   ├── network.py              # Agent interaction topologies
│   ├── simulation.py           # Core simulation engine
│   └── prompts/
│       └── strategy_select.txt  # LLM prompt template
├── run_research_experiment.py   # Main experiment runner (CLI)
├── testapi                      # API connectivity test script
├── README.md                    # User documentation
├── CLAUDE.md                    # This file (AI assistant guide)
├── .gitignore                   # Git ignore patterns
└── results/                     # Experiment output (gitignored)
```

## Core Concepts

### Games (game_theory/games.py)

Four classic games with payoff matrices defined:

| Game | Key Property | Nash Equilibrium |
|------|--------------|------------------|
| **Prisoner's Dilemma** | T > R > P > S | Defect-Defect (suboptimal) |
| **Snowdrift** | T > R > S > P | Mixed strategy |
| **Stag Hunt** | R > T > P > S | Two pure NE (CC, DD) |
| **Harmony** | R > T, R > S | Cooperate (dominant) |

**Access via:** `GAME_REGISTRY["prisoners_dilemma"]` or `PRISONERS_DILEMMA` constant

### Actions (game_theory/games.py)

```python
from game_theory import Action
Action.COOPERATE  # "cooperate"
Action.DEFECT     # "defect"
```

### Strategies (game_theory/strategies.py)

Classical strategies available in `STRATEGY_REGISTRY`:
- `AlwaysCooperate`, `AlwaysDefect`, `RandomStrategy`
- `TitForTat`, `TitForTwoTats`, `SuspiciousTitForTat`
- `GrimTrigger`, `Pavlov`, `GradualStrategy`
- `ProbabilisticCooperator`

**Important:** `LLMStrategy` is in a separate module to avoid circular imports:
```python
from game_theory.llm_strategy import LLMStrategy
```

### LLM Strategy Modes (game_theory/llm_strategy.py)

- **Pure Mode:** LLM analyzes game state independently
- **Hybrid Mode:** Code pre-computes statistics, LLM makes final decision

Key configuration options:
- `provider`: "deepseek" | "openai" | "claude" | "ollama"
- `mode`: "pure" | "hybrid"
- `history_window`: Number of past rounds to include (5/10/20/None=all)
- `enable_cheap_talk`: Allow language-based negotiation messages
- `temperature`, `max_tokens`: LLM generation parameters

### Network Topologies (game_theory/network.py)

Available in `NETWORK_REGISTRY`:
- `FullyConnectedNetwork` - Complete graph (all pairs interact)
- `RingNetwork` - Circular arrangement
- `GridNetwork` - 2D grid
- `StarNetwork` - Central hub
- `SmallWorldNetwork` - High clustering, short paths
- `ScaleFreeNetwork` - Power-law degree distribution
- `RandomNetwork` - Random connectivity

### Response Parsing (game_theory/llm_strategy.py:ResponseParser)

The `ResponseParser` class handles LLM response parsing with:
- 20+ format pattern support
- Confidence scoring (0.0-1.0)
- Ambiguity detection
- Fallback to random action (eliminates bias on parse failure)

## Key Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `run_research_experiment.py` | ~1625 | Experiment orchestration, CLI, statistics, visualization |
| `game_theory/llm_strategy.py` | ~620 | LLMStrategy class, ResponseParser |
| `game_theory/simulation.py` | ~432 | GameSimulation engine, AgentState |
| `game_theory/network.py` | ~402 | Network topology implementations |
| `game_theory/strategies.py` | ~311 | Classical strategy implementations |
| `game_theory/llm_api.py` | ~269 | LLM provider abstraction |
| `game_theory/games.py` | ~158 | Game definitions, payoff matrices |

## Code Conventions

### Naming
- **Classes:** PascalCase (`TitForTat`, `LLMStrategy`, `GameSimulation`)
- **Functions:** snake_case (`choose_action`, `compute_statistics`)
- **Constants:** UPPER_CASE (`PRISONERS_DILEMMA`, `MAX_API_WORKERS`)
- **Private:** Leading underscore (`_build_prompt`, `_run_single_round`)

### Design Patterns Used
1. **Strategy Pattern:** Base `Strategy` class with concrete implementations
2. **Factory Pattern:** `create_strategy()`, `create_network()`
3. **Registry Pattern:** `STRATEGY_REGISTRY`, `GAME_REGISTRY`, `NETWORK_REGISTRY`
4. **Singleton:** Global `_shared_session` in llm_api.py for connection pooling

### Documentation Style
- Bilingual docstrings (English and Chinese)
- Module-level documentation blocks
- Type hints on function signatures

### Error Handling
- Try-except with fallback strategies
- Random selection on parse failures (unbiased)
- Graceful degradation (e.g., networkx optional)

## Performance Optimizations

The codebase includes several optimizations:
- **Connection Pooling:** `HTTPAdapter` with `pool_connections=100`
- **Parallel API Calls:** `ThreadPoolExecutor` with `MAX_API_WORKERS=100`
- **History Windowing:** Configurable to reduce LLM context size
- **Session Reuse:** Global session for all API calls

## Output Structure

Experiment results are saved to `results/` (gitignored):

```
results/
└── {timestamp}/
    ├── experiment_config.json   # Run configuration
    ├── summary.json             # Aggregated results
    ├── prisoners_dilemma/
    │   ├── pure_vs_hybrid.json
    │   ├── pure_vs_hybrid.png
    │   └── ...
    ├── snowdrift/
    └── stag_hunt/
```

## Common Tasks for AI Assistants

### Adding a New Game

1. Define in `game_theory/games.py`:
```python
NEW_GAME = GameConfig(
    name="New Game",
    payoff_matrix={
        (Action.COOPERATE, Action.COOPERATE): (3, 3),
        (Action.COOPERATE, Action.DEFECT): (0, 5),
        (Action.DEFECT, Action.COOPERATE): (5, 0),
        (Action.DEFECT, Action.DEFECT): (1, 1),
    },
    description="English description",
    description_cn="中文描述"
)
```
2. Add to `GAME_REGISTRY`
3. Export in `__init__.py`

### Adding a New Strategy

1. Inherit from `Strategy` in `game_theory/strategies.py`:
```python
class NewStrategy(Strategy):
    name = "new_strategy"
    description = "Description"

    def choose_action(self, my_history, opponent_history, opponent_name=None):
        # Return Action.COOPERATE or Action.DEFECT
        pass
```
2. Add to `STRATEGY_REGISTRY`
3. Export in `__init__.py`

### Adding a New LLM Provider

1. Add configuration in `game_theory/llm_api.py`:
   - Add to `DEFAULT_CONFIG`
   - Implement API call logic in `LLMClient.call()`
2. Update the setup wizard if needed

### Running a Custom Experiment

```python
from game_theory import PRISONERS_DILEMMA, GameSimulation, AgentState
from game_theory.llm_strategy import LLMStrategy

# Create agents
agent1 = AgentState(
    name="Agent1",
    strategy=LLMStrategy(provider="deepseek", mode="hybrid"),
    description="LLM Agent"
)
agent2 = AgentState(
    name="Agent2",
    strategy=TitForTat(),
    description="TitForTat"
)

# Run simulation
sim = GameSimulation(
    agents=[agent1, agent2],
    game=PRISONERS_DILEMMA,
    rounds=20
)
results = sim.run()
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `DEEPSEEK_API_KEY` | DeepSeek API authentication |
| `OPENAI_API_KEY` | OpenAI API authentication |
| `ANTHROPIC_API_KEY` | Anthropic Claude API authentication |

API keys can also be configured via `game_theory/llm_config.json` or the interactive setup wizard.

## Testing

- **API Test:** `python testapi` - validates connectivity to all providers
- **No formal test framework** - testing is done via experiment runs
- **Manual validation** through statistical result summaries

## Important Notes

1. **LLMStrategy Import:** Always import separately to avoid circular dependencies:
   ```python
   from game_theory.llm_strategy import LLMStrategy
   ```

2. **Response Parsing:** The `ResponseParser` uses random fallback on failures to avoid systematic bias

3. **API Rate Limits:** The system uses connection pooling and parallel execution - be mindful of provider rate limits

4. **Results Directory:** The `results/` directory is gitignored - results are timestamped for each run

5. **Bilingual Support:** All game descriptions and docstrings support English and Chinese

## Version

Current version: `0.1.0` (defined in `game_theory/__init__.py`)
Experiment script version: `v7` (defined in `run_research_experiment.py`)
