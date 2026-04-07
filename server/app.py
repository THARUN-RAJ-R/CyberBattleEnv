"""
server/app.py — Root-level server entry point required by openenv validate.

openenv validate checks:
  1. server/app.py exists
  2. It has a callable main() function
  3. It has if __name__ == '__main__' guard

The actual FastAPI app logic lives in cyber_battle_env/server/app.py
This file is the entry point shim.
"""
import sys
import os

# Make sure the package is importable when called from this directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cyber_battle_env.server.app import app  # noqa: F401


def main() -> None:
    """Start the CyberBattleEnv FastAPI server via uvicorn."""
    import uvicorn
    uvicorn.run(
        "cyber_battle_env.server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        workers=int(os.getenv("WEB_CONCURRENCY", "1")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
