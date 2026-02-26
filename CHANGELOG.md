# Changelog

All notable changes to `aumos-critical-thinking` will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-26

### Added
- Initial release of the AumOS Critical Thinking service
- `BiasDetectorService` — automation bias detection with 0.0–1.0 scoring and categorical classification
- `JudgmentValidatorService` — human judgment validation against ground truth, expert consensus, and outcome feedback
- `AtrophyMonitorService` — skill atrophy assessment with baseline comparison and severity classification
- `ChallengeGeneratorService` — AI-assisted challenge scenario generation with embedded AI traps
- `TrainingRecommenderService` — training program recommendations from atrophy assessments with `recommend_from_assessment`
- REST API: 9 endpoints across bias, judgment, atrophy, challenges, and training domains
- DB models: `crt_bias_detections`, `crt_judgment_validations`, `crt_atrophy_assessments`, `crt_challenges`, `crt_training_recommendations`
- Hexagonal architecture: api/ + core/ + adapters/ layers with Protocol-based interfaces
- Kafka events for severe bias detection, low accuracy judgment, atrophy interventions, and training lifecycle
- Tenant RLS isolation via `get_db_session` in all repository operations
- Pydantic v2 request/response schemas with field validation
- Multi-stage Docker build with non-root user
- Dev stack via `docker-compose.dev.yml`
