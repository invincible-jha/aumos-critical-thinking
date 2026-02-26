.PHONY: install dev test lint typecheck fmt clean docker-build docker-up docker-down

# Install production dependencies
install:
	pip install -e .

# Install all dependencies including dev extras
dev:
	pip install -e ".[dev]"

# Run tests with coverage
test:
	pytest

# Run linter
lint:
	ruff check src/ tests/

# Run type checker
typecheck:
	mypy src/

# Format code
fmt:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# Clean build artefacts
clean:
	rm -rf dist/ build/ *.egg-info .coverage htmlcov/ .mypy_cache/ .ruff_cache/ .pytest_cache/

# Build Docker image
docker-build:
	docker build -t aumos-critical-thinking:dev .

# Start development stack
docker-up:
	docker compose -f docker-compose.dev.yml up -d

# Stop development stack
docker-down:
	docker compose -f docker-compose.dev.yml down

# View service logs
logs:
	docker compose -f docker-compose.dev.yml logs -f critical-thinking

# Run service locally (requires .env)
run:
	uvicorn aumos_critical_thinking.main:app --host 0.0.0.0 --port 8000 --reload
