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
# Run all experiments
python run_research_experiment.py

# Run specific experiment
python run_research_experiment.py pure_hybrid
python run_research_experiment.py multi_llm
python run_research_experiment.py cheap_talk
python run_research_experiment.py baseline

# With options
python run_research_experiment.py all --provider deepseek --repeats 5 --rounds 30
```

Available experiments: `pure_hybrid`, `window`, `multi_llm`, `cheap_talk`, `group`, `group_single`, `baseline`, `all`
