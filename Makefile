.PHONY: up down dev test lint clean download-models

up:
	docker compose up -d

down:
	docker compose down

dev:
	docker compose -f docker-compose.dev.yml up

test:
	cd gateway && npm test
	cd ai-engine && python -m pytest tests/ -v
	cd streaming && python -m pytest tests/ -v
	cd biometrics && python -m pytest tests/ -v

lint:
	cd gateway && npm run lint
	cd ai-engine && ruff check src/
	cd streaming && ruff check src/
	cd biometrics && ruff check src/

download-models:
	python training/scripts/download_pretrained.py

clean:
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf weights/ models_cache/
