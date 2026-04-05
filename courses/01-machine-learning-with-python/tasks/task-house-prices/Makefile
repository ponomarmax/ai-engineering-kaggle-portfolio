SHELL := /bin/zsh

ROOT := $(shell git rev-parse --show-toplevel)
VENV := $(ROOT)/.venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
KAGGLE := $(VENV)/bin/kaggle
KAGGLE_ENV := $(ROOT)/.env.kaggle
COMPETITION := home-data-for-ml-course
DATA_DIR := data
RAW_DATA_DIR := $(DATA_DIR)/raw
DOWNLOAD_ARCHIVE := $(DATA_DIR)/$(COMPETITION).zip
SUBMISSION_FILE := submissions/submission.csv
MESSAGE ?= simple baseline

.PHONY: help setup download run submit clean

help:
	@echo "make setup   - create root virtualenv and install dependencies"
	@echo "make download - download and extract Kaggle competition data into data/raw/"
	@echo "make run     - run main.py"
	@echo 'make submit  - submit submissions/submission.csv using root .env.kaggle; optional MESSAGE="..."'
	@echo "make clean   - remove generated submission"

setup:
	python3 -m venv "$(VENV)"
	$(PIP) install --upgrade pip
	$(PIP) install -r  requirements.txt

download:
	mkdir -p "$(DATA_DIR)" "$(RAW_DATA_DIR)"
	@if [ -f "$(KAGGLE_ENV)" ]; then source "$(KAGGLE_ENV)"; fi; \
	$(KAGGLE) competitions download -c $(COMPETITION) -p "$(DATA_DIR)"
	@if [ -f "$(DOWNLOAD_ARCHIVE)" ]; then \
		$(PYTHON) -m zipfile -e "$(DOWNLOAD_ARCHIVE)" "$(RAW_DATA_DIR)"; \
	fi

run:
	$(PYTHON) main.py

submit:
	@if [ -f "$(KAGGLE_ENV)" ]; then source "$(KAGGLE_ENV)"; fi; \
	$(KAGGLE) competitions submit -c $(COMPETITION) -f "$(SUBMISSION_FILE)" -m "$(MESSAGE)"

clean:
	rm -f "$(SUBMISSION_FILE)"
