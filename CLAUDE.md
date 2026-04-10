# Project Name

> One-line description of what this project does.

## Quick Start

```bash
source venv/bin/activate
python src/main.py
```

## Structure

- `src/` — Source code
- `tests/` — pytest tests
- `scripts/` — Entry points, utilities, demo scripts
- `data/` — Input data, outputs, caches
- `configs/` — YAML/JSON configuration files
- `docs/` — Documentation

## Stack

- Python 3.12+
- Dependencies: see `requirements.txt`
- API Keys: loaded from `~/.claude/.env` or `.env`

## Development

```bash
# Run tests
pytest tests/ -v

# Run main
python src/main.py

# Run specific script
python scripts/demo.py
```

## Conventions

- Type hints on all public functions
- One module per concern — keep files under 300 lines
- Tests mirror src/ structure: `src/agent.py` → `tests/test_agent.py`
- Configs in YAML, not hardcoded
- API keys from environment, never committed
