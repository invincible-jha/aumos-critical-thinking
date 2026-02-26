# CLAUDE.md — aumos-critical-thinking

## Service Overview
Anti-automation-bias frameworks, human judgment validation, skill atrophy monitoring,
challenge scenario generation, and training recommendations.

- **Package**: `aumos_critical_thinking`
- **Table prefix**: `crt_`
- **Env prefix**: `AUMOS_CRITICAL_`
- **Port**: 8000

## Architecture
Hexagonal: `api/` (FastAPI routes + schemas) → `core/` (services + models + interfaces) → `adapters/` (repositories + Kafka)

Services depend only on Protocol interfaces. Concrete adapters live in `adapters/`.

## Key Services
| Service | Responsibility |
|---------|---------------|
| `BiasDetectorService` | Detect and score automation bias in human-AI decisions |
| `JudgmentValidatorService` | Validate human judgment against reference standards |
| `AtrophyMonitorService` | Track skill atrophy with baseline comparison |
| `ChallengeGeneratorService` | Generate challenge scenarios with AI traps |
| `TrainingRecommenderService` | Recommend training programs from assessment data |

## DB Tables
- `crt_bias_detections` — bias score + category per decision event
- `crt_judgment_validations` — accuracy score + divergence analysis
- `crt_atrophy_assessments` — atrophy rate + severity per user/domain
- `crt_challenges` — scenario + AI trap + expected reasoning
- `crt_training_recommendations` — program modules + challenge assignments

## API Routes
```
POST   /api/v1/critical/bias/detect
GET    /api/v1/critical/bias/reports
POST   /api/v1/critical/judgment/validate
GET    /api/v1/critical/judgment/history
POST   /api/v1/critical/atrophy/assess
GET    /api/v1/critical/atrophy/metrics
POST   /api/v1/critical/challenges/generate
GET    /api/v1/critical/challenges
GET    /api/v1/critical/training/recommendations
```

## Key Invariants
- Bias scores are always 0.0–1.0; severe alerts fire at threshold ≥ 0.75
- Atrophy rate: positive = degradation, negative = improvement
- Challenge `ai_trap` field is only set when `include_ai_trap=True`
- `recommend_from_assessment()` auto-generates training from AtrophyAssessment
- Tenant isolation: all DB queries scoped via `get_db_session(tenant_id)`

## Development Commands
```bash
make dev        # Install with dev extras
make test       # pytest with coverage
make lint       # ruff check
make typecheck  # mypy strict
make fmt        # ruff format + fix
make docker-up  # Start dev stack
```

## Dependencies
- `aumos-common` — database, events, auth, config, health, errors, observability
- `aumos-proto` — Protobuf event schemas
- `fastapi`, `pydantic`, `uvicorn`, `sqlalchemy`, `asyncpg`, `httpx`
