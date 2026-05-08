# Contributing to Langfuse OpenAI Proxy

Thank you for your interest in contributing!

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
3. Install with dev dependencies: `pip install -e ".[dev]"`
4. Copy `.env.example` to `.env` and configure your upstream URL

## Making Changes

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Run the linter: `ruff check . && ruff format .`
4. Run tests: `pytest -v`
5. Commit with a descriptive message

## Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Line length: 100 characters
- Python 3.11+ (use modern syntax: `X | Y` unions, etc.)
- Follow the existing layered architecture (API / Domain / Infrastructure)

## Pull Requests

- Keep PRs focused on a single concern
- Include tests for new functionality
- Update README if adding user-facing features
- Ensure CI passes (ruff lint + pytest)

## Reporting Issues

- Use [GitHub Issues](https://github.com/CraftaleAgency/Langfuse-Openai-Proxy/issues)
- Include Python version, proxy version, and steps to reproduce
- Include relevant log output (redact any API keys)

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
