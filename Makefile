SHELL := /bin/zsh

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
CURRENT_TASK_FILE := .current-task
CURRENT_TASK := $(strip $(shell [ -f "$(CURRENT_TASK_FILE)" ] && sed -n '1p' "$(CURRENT_TASK_FILE)"))
TASK ?= $(CURRENT_TASK)

.PHONY: help ensure-task setup download run submit update shell clean clean-venv active-task task-help

help:
	@echo "Root workflow for the whole repository"
	@echo ""
	@echo "Active task file: $(CURRENT_TASK_FILE)"
	@echo "Current active task: $(if $(TASK),$(TASK),<not set>)"
	@echo ""
	@echo "make setup      - install dependencies for the active task into the shared root .venv"
	@echo "make download   - download Kaggle data for the active task into its data folder"
	@echo "make run        - run the active task"
	@echo 'make submit     - submit the active task output to Kaggle, pass MESSAGE="..." if needed'
	@echo "make update     - refresh dependencies for the active task"
	@echo "make task-help  - show the active task's local Makefile help"
	@echo "make active-task - print the active task path"
	@echo "make shell      - open a shell with the root .venv activated"
	@echo "make clean      - clean generated files for the active task"
	@echo "make clean-venv - remove the shared root .venv"
	@echo ""
	@echo "Override the active task for one command:"
	@echo '  make run TASK=courses/01-machine-learning-with-python/tasks/task-house-prices'

ensure-task:
	@if [ -z "$(TASK)" ]; then \
		echo "No active task selected."; \
		echo "Create $(CURRENT_TASK_FILE) with a task path, for example:"; \
		echo "  courses/01-machine-learning-with-python/tasks/task-house-prices"; \
		echo 'Or override it per command with: make run TASK=...'; \
		exit 1; \
	fi
	@if [ ! -d "$(TASK)" ]; then \
		echo "Task directory does not exist: $(TASK)"; \
		exit 1; \
	fi
	@if [ ! -f "$(TASK)/Makefile" ]; then \
		echo "Task Makefile not found: $(TASK)/Makefile"; \
		exit 1; \
	fi

setup:
	@$(MAKE) ensure-task
	@$(MAKE) -C "$(TASK)" setup

download:
	@$(MAKE) ensure-task
	@$(MAKE) -C "$(TASK)" download

run:
	@$(MAKE) ensure-task
	@$(MAKE) -C "$(TASK)" run

submit:
	@$(MAKE) ensure-task
	@$(MAKE) -C "$(TASK)" submit MESSAGE="$(MESSAGE)"

update:
	@$(MAKE) ensure-task
	@$(MAKE) -C "$(TASK)" setup

task-help:
	@$(MAKE) ensure-task
	@$(MAKE) -C "$(TASK)" help

active-task:
	@$(MAKE) ensure-task
	@echo "$(TASK)"

shell:
	@/bin/zsh -lc 'if [ -f "$(CURDIR)/$(VENV)/bin/activate" ]; then source "$(CURDIR)/$(VENV)/bin/activate"; fi; exec $$SHELL -l'

clean:
	@$(MAKE) ensure-task
	@$(MAKE) -C "$(TASK)" clean

clean-venv:
	rm -rf "$(VENV)"
