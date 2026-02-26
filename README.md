# aumos-critical-thinking

Anti-automation-bias frameworks, human judgment validation, skill atrophy monitoring, challenge scenario generation, and training recommendations for the AumOS Enterprise platform.

## Purpose

As AI systems make more decisions, human operators risk losing the critical thinking skills needed to evaluate, override, and supervise AI output effectively. This service provides the tooling to detect, measure, and remediate automation bias and skill atrophy across your AI-augmented workforce.

## Key Features

- **Bias Detection** — Scores every human-AI decision event on a 0.0–1.0 automation bias index
- **Judgment Validation** — Validates human decisions against ground truth, expert consensus, or outcome feedback
- **Atrophy Monitoring** — Tracks skill degradation over time with baseline comparison and severity classification
- **Challenge Generation** — AI-assisted scenario generation with embedded "AI traps" to test critical override
- **Training Recommendations** — Auto-generated training programs from assessment outcomes, with deliberate practice challenge assignments

## Quick Start

```bash
cp .env.example .env
make docker-up
# Service available at http://localhost:8000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/critical/bias/detect` | Detect automation bias in a decision event |
| GET | `/api/v1/critical/bias/reports` | List bias detection reports |
| POST | `/api/v1/critical/judgment/validate` | Validate human judgment |
| GET | `/api/v1/critical/judgment/history` | Judgment validation history |
| POST | `/api/v1/critical/atrophy/assess` | Perform skill atrophy assessment |
| GET | `/api/v1/critical/atrophy/metrics` | List atrophy metrics |
| POST | `/api/v1/critical/challenges/generate` | Generate challenge scenario |
| GET | `/api/v1/critical/challenges` | List challenge scenarios |
| GET | `/api/v1/critical/training/recommendations` | Training recommendations |

API docs: `http://localhost:8000/docs`

## Database Tables

| Table | Description |
|-------|-------------|
| `crt_bias_detections` | Automation bias detection records |
| `crt_judgment_validations` | Human judgment validation results |
| `crt_atrophy_assessments` | Skill atrophy assessment metrics |
| `crt_challenges` | Generated challenge scenarios |
| `crt_training_recommendations` | Recommended training programs |

## Architecture

Hexagonal architecture with strict layer separation:

```
src/aumos_critical_thinking/
├── api/           — FastAPI router + Pydantic schemas
├── core/          — Domain models, services, Protocol interfaces
└── adapters/      — SQLAlchemy repositories, Kafka publisher
```

## Development

```bash
make dev        # Install dev dependencies
make test       # Run tests with coverage
make lint       # Lint with ruff
make typecheck  # Type-check with mypy
make fmt        # Auto-format
```

## Environment Variables

See `.env.example` for all configuration options. Key variables:

- `AUMOS_CRITICAL_DATABASE_URL` — PostgreSQL connection string
- `AUMOS_CRITICAL_KAFKA_BOOTSTRAP_SERVERS` — Kafka broker address
- `AUMOS_CRITICAL_SEVERE_BIAS_THRESHOLD` — Alert threshold for severe bias (default: 0.75)
- `AUMOS_CRITICAL_LLM_MODEL_ID` — LLM model for challenge generation (default: claude-opus-4-6)
- `AUMOS_CRITICAL_ATROPHY_INTERVENTION_THRESHOLD` — Severity level requiring immediate action (default: high)

## License

Apache-2.0. See [LICENSE](LICENSE).
