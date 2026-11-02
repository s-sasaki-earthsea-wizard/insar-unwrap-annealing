# insar-unwrap-annealing
# Run `make help` for the list of targets. All Python commands go through
# `uv run --locked` so that uv.lock is never rewritten by accident.

.DEFAULT_GOAL := help

UV      ?= uv
# Optional: put the project environment outside the checkout (recommended when
# the repository lives on a network filesystem such as CIFS/NFS):
#   make VENV=$$HOME/.cache/insar-unwrap-annealing/venv sync
VENV    ?=
ifneq ($(VENV),)
export UV_PROJECT_ENVIRONMENT := $(VENV)
endif
export UV_LINK_MODE ?= copy
RUN     := $(UV) run --locked
CONFIG  ?= configs/experiment.example.yaml
COMPOSE ?= docker compose

.PHONY: help
help: ## Show this help
	@echo "insar-unwrap-annealing -- available targets:"
	@echo ""
	@awk 'BEGIN{FS=":.*?## "} /^[a-zA-Z0-9_.-]+:.*?## / { printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""

##@ Environment
.PHONY: sync
sync: ## Install the locked environment including dev tools
	$(UV) sync --locked --group dev

##@ Quality
.PHONY: lint
lint: ## Run ruff check
	$(RUN) ruff check .

.PHONY: format
format: ## Format code with ruff
	$(RUN) ruff format .

.PHONY: format-check
format-check: ## Check formatting without changing files
	$(RUN) ruff format --check .

.PHONY: test
test: ## Run pytest
	$(RUN) pytest

.PHONY: check
check: lint format-check test ## Lint, format check and tests

##@ CLI
.PHONY: validate-config
validate-config: ## Validate $(CONFIG) (no data access)
	$(RUN) insar-unwrap validate-config --config $(CONFIG)

.PHONY: doctor
doctor: ## Report Python, packages, SA/SQA samplers and SNAPHU
	$(RUN) insar-unwrap doctor

##@ Data
.PHONY: datasets
datasets: ## List approved sample datasets (configs/datasets.yaml)
	$(RUN) insar-unwrap datasets

.PHONY: fetch-sample
fetch-sample: ## Download + verify an approved dataset: make fetch-sample DATASET=<name>
	@test -n "$(DATASET)" || { echo "usage: make fetch-sample DATASET=<name> (see make datasets)"; exit 2; }
	$(RUN) insar-unwrap fetch-sample $(DATASET)

.PHONY: inspect
inspect: ## List layers, grids and masks of a GUNW file: make inspect FILE=<path>
	@test -n "$(FILE)" || { echo "usage: make inspect FILE=<path to .h5>"; exit 2; }
	$(RUN) insar-unwrap inspect $(FILE)

.PHONY: crop-summary
crop-summary: ## Read the crop of $(CONFIG) and build its BQM, no solving (SIZE=<n> optional)
	$(RUN) insar-unwrap crop-summary --config $(CONFIG) $(if $(SIZE),--size $(SIZE))

##@ Docker
.PHONY: docker-build
docker-build: ## Build the image (UID/GID default to the current user)
	HOST_UID=$$(id -u) HOST_GID=$$(id -g) $(COMPOSE) build

.PHONY: docker-check
docker-check: ## Run validate-config, doctor and pytest inside the container
	$(COMPOSE) run --rm app insar-unwrap validate-config --config $(CONFIG)
	$(COMPOSE) run --rm app insar-unwrap doctor
	$(COMPOSE) run --rm app pytest
