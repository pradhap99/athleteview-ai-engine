# Contributing to AthleteView

## Setup
1. Fork the repository
2. Clone your fork
3. Copy `.env.example` to `.env`
4. Run `docker compose -f docker-compose.dev.yml up`

## Code Style
- Python: Ruff + Black (line length 120)
- TypeScript: ESLint + Prettier
- C: Linux kernel style (tabs, 80 cols)

## Pull Requests
- One feature per PR
- Include tests
- Update docs if API changes
- Squash commits before merge

## Architecture Decisions
Major changes require an ADR (Architecture Decision Record) in `docs/adr/`.
