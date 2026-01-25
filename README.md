# Game Theory LLM Multi-Agent Research

## Requirements

```bash
pip install numpy matplotlib requests
```

## Configuration

Set API keys via environment variables:

```bash
export DEEPSEEK_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

Or run the setup wizard:

```bash
python game_theory/llm_api.py setup
```

## Run Experiments

```bash
# Show help and usage
python research.py -h
python research.py --help

# Run all experiments (with default settings)
python research.py all

# Run specific experiment
python research.py pure_hybrid
python research.py multi_llm
python research.py cheap_talk
python research.py baseline

# With options
python research.py all --provider deepseek --repeats 5 --rounds 30
python research.py group --n_agents 15 --rounds 30
python research.py baseline --games pd

# Cross-provider cheap talk (LLM vs LLM)
python research.py cheap_talk --provider1 openai --provider2 gemini
```

## Available Experiments

| Command | Description |
|---------|-------------|
| `pure_hybrid` | Pure vs Hybrid LLM mode comparison |
| `window` | Memory window size comparison |
| `multi_llm` | Multi-provider LLM comparison |
| `cheap_talk` | Language communication (LLM vs LLM, cross-provider) |
| `group` | Group dynamics (multi-provider) |
| `group_single` | Group dynamics (single provider) |
| `baseline` | LLM vs classical strategies |
| `all` | Run all experiments |

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--provider` | LLM provider (deepseek/openai/gemini) | deepseek |
| `--provider1` | Player1 provider for cheap_talk | --provider |
| `--provider2` | Player2 provider for cheap_talk | --provider |
| `--repeats` | Number of repetitions | 3 |
| `--rounds` | Rounds per game | 20 |
| `--games` | Game type (pd/snowdrift/stag_hunt/harmony/all) | all |
| `--n_agents` | Number of agents for group experiments | 10 |

## Output Structure

```
results/{timestamp}/
├── experiment_config.json
├── summary.json
├── details/
│   ├── {experiment}_{provider}_{trial}_{rounds}.json
│   └── {experiment}_{game}_{provider}_rounds.json
├── summary/
│   └── {experiment}.csv
├── prisoners_dilemma/
│   ├── pure_vs_hybrid.json
│   └── pure_vs_hybrid.png
├── snowdrift/
├── stag_hunt/
└── harmony/
```
