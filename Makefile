# Prosper Challenge — run everything from the repo root.

VENV := backend/.venv
PYTHON := $(VENV)/bin/python

.PHONY: help install run eval clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv and install dependencies
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r backend/requirements.txt

run: ## Run the voice agent + builder UI (then open http://localhost:7860/builder)
	$(PYTHON) backend/bot.py

eval: ## Simulated-caller eval: our agent vs full-catalog baselines (needs OPENAI_API_KEY)
	cd backend && .venv/bin/python -m sim.eval --model gpt-4.1-mini

clean: ## Remove the venv and Python caches
	rm -rf $(VENV)
	find backend -type d -name __pycache__ -prune -exec rm -rf {} +
