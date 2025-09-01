# Logging

This repository supports structured logging for scripts in `allegiance/`, `julien_data/`, and shared modules.

- Configure via `config/logging.yaml` (see `config/logging.example.yaml`).
- Override path via environment variable `NET_FLUIDITY_LOGGING`.
- Default fallback uses `logging.basicConfig(level=INFO)`.

Quick start:

1) Copy the example: `cp config/logging.example.yaml config/logging.yaml`
2) Run any script; logs emit to console, and to `logs/net_fluidity.log` if the file handler stays enabled.

Conventions:

- Use `logger = logging.getLogger(__name__)` in modules; do not configure handlers in libraries.
- Scripts may call the shared `setup_logging()` (as in diffs) before heavy work starts.
- Keep INFO-level for high-level progress; DEBUG for verbose internals.

