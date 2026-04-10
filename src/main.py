"""Main entry point."""

import os
from pathlib import Path


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
            break


def main():
    load_env()
    print("Project running. Edit src/main.py to get started.")


if __name__ == "__main__":
    main()
