"""Main entry point — starts the FastAPI server."""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'src' is importable
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import uvicorn


def load_env():
    """Load API keys from ~/.claude/.env or local .env"""
    env_paths = [
        Path.home() / ".claude" / ".env",
        Path(__file__).parent.parent / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ.setdefault(key.strip(), value.strip())


def main():
    load_env()
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
