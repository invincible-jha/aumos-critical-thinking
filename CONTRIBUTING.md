# Contributing to aumos-critical-thinking

## Development Setup

```bash
git clone <repo-url>
cd aumos-critical-thinking
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
make dev
```

## Running Tests

```bash
make test
```

## Code Style

- Python 3.11+ with type hints on all function signatures
- Ruff for linting and formatting: `make lint && make fmt`
- mypy strict mode: `make typecheck`
- Line length: 120 characters
- Named exports, functional patterns preferred over classes where practical

## Commit Convention

Conventional commits:
- `feat:` — new feature
- `fix:` — bug fix
- `refactor:` — code restructure without behaviour change
- `docs:` — documentation
- `test:` — tests only
- `chore:` — tooling, dependencies, CI

## Architecture

This service follows AumOS hexagonal architecture:

```
api/        — FastAPI router and Pydantic schemas (no business logic)
core/       — Domain models, service classes, Protocol interfaces
adapters/   — SQLAlchemy repositories and Kafka publisher
```

Services depend on interfaces (Protocols), never on concrete adapters.
All DB operations set the `app.current_tenant` RLS parameter via `get_db_session`.

## Pull Request Process

1. Branch from `main`: `feature/`, `fix/`, or `docs/` prefix
2. Write tests alongside implementation
3. Ensure `make test lint typecheck` all pass
4. Open PR with description of WHY (not what)
5. Squash-merge after review
