# SignWriting Transcription

# Default configuration
CONFIG_FILE ?= signwriting_transcription/config.yaml

# Data paths
DATA_DIR ?= $(PWD)/data
POSES_DIR ?= $(PWD)/poses
OUTPUT_DIR ?= $(PWD)/results

DATA_CSV ?= $(DATA_DIR)/data.csv

ifeq ($(firstword $(MAKECMDGOALS)),overfit)
CONFIG_FILE := example/overfit_config.yaml
DATA_DIR := example/data
POSES_DIR := example/poses
OUTPUT_DIR := example/results
DATA_CSV := example/data.csv
endif

# Model settings
WANDB_PROJECT := mmh_transcription_local

# HuggingFace token (set via environment variable)
# export HUGGINGFACE_TOKEN=your_token_here

# Help
.PHONY: help

## Print this help
help:
	@awk '/^##/{c=substr($$0,3);next}c&&/^[[:alpha:]][[:alnum:]_-]+:/{print substr($$1,1,index($$1,":")),c}1{c=0}' $(MAKEFILE_LIST) | column -s: -t

# ----------------------------------------------------------

.PHONY: prepare train overfit

$(DATA_DIR):
	mkdir -p $(DATA_DIR)

## Downloads the dataset to DATA_DIR
$(DATA_CSV): $(DATA_DIR)
	test -f $@ || wget -O $@ "https://github.com/sign/data/raw/refs/heads/main/signwriting-transcription/data.csv"

## Augment dataset by adding segmented signs for single signs
$(DATA_DIR)/data_augmented.csv: $(DATA_DIR) $(DATA_CSV) signwriting_transcription/segment_signs.py
	@echo "Segmenting signs..."
	python signwriting_transcription/segment_signs.py \
		--input "$(DATA_CSV)" \
		--poses "$(POSES_DIR)" \
		--output $@

TRANSFORM_STAMP = $(DATA_DIR)/.transformed.stamp
TRANSFORMED_FILES = \
	"$(DATA_DIR)/data.train.tsv" \
	"$(DATA_DIR)/data.dev.tsv" \
	"$(DATA_DIR)/data.test.tsv" \
	"$(DATA_DIR)/data.tokens.txt"

# tell make they exist and don’t need building
$(TRANSFORMED_FILES): ;

## Prepare dataset for training format
$(TRANSFORM_STAMP): $(DATA_DIR)/data_augmented.csv signwriting_transcription/transform_dataset.py
	@echo "Transforming dataset to TSV format..."
	python signwriting_transcription/transform_dataset.py \
		--input "$(DATA_DIR)/data_augmented.csv" \
		--poses "$(POSES_DIR)" \
		--output "$(DATA_DIR)/data.tsv"
	touch $(TRANSFORM_STAMP)


$(OUTPUT_DIR)/config.yaml: $(CONFIG_FILE)
	mkdir -p "$(OUTPUT_DIR)"
	@echo "Updating config file paths..."
	sed \
	  -e 's|train_metadata_file: ".*"|train_metadata_file: "$(DATA_DIR)/data.train.tsv"|' \
	  -e 's|validation_metadata_file: ".*"|validation_metadata_file: "$(DATA_DIR)/data.dev.tsv"|' \
	  -e 's|test_metadata_file: ".*"|test_metadata_file: "$(DATA_DIR)/data.test.tsv"|' \
	  -e 's|new_vocabulary: ".*"|new_vocabulary: "$(DATA_DIR)/data.tokens.txt"|' \
	  -e 's|output_dir: ".*"|output_dir: "$(OUTPUT_DIR)"|' \
	  -e 's|logging_dir: ".*"|logging_dir: "$(OUTPUT_DIR)/logs"|' \
  		"$(CONFIG_FILE)" > "$@"

## Prepare model, tokenizer, processor using multimodalhugs
prepare: $(OUTPUT_DIR)/config.yaml $(TRANSFORM_STAMP)
	@echo "Setting up multimodalhugs..."
	multimodalhugs-setup --modality "pose2text" --config_path "$(OUTPUT_DIR)/config.yaml"

## Train the model
train:
	huggingface-cli login --token "$(HUGGINGFACE_TOKEN)"

	# Train the model
	@echo "Starting training..."
	multimodalhugs-train \
	  --task "translation" \
	  --config_path "$(OUTPUT_DIR)/config.yaml" \
	  --output_dir "$(OUTPUT_DIR)"

example/overfit_config.yaml: signwriting_transcription/config.yaml
	sed \
	  -e 's/auto_find_batch_size: True/auto_find_batch_size: False/' \
	  -e 's/per_device_train_batch_size: .*/per_device_train_batch_size: 4/' \
	  -e 's/per_device_eval_batch_size: .*/per_device_eval_batch_size: 4/' \
	  -e 's/dataloader_num_workers: .*/dataloader_num_workers: 4/' \
	  -e 's/gradient_accumulation_steps: .*/gradient_accumulation_steps: 1/' \
	  -e 's/multimodal_mapper_dropout: .*/multimodal_mapper_dropout: 0.0/' \
	  -e 's/weight_decay: .*/weight_decay: 0/' \
	  -e 's/lr_scheduler_type: .*/lr_scheduler_type: "constant"/' \
	  -e 's/warmup_steps: .*/warmup_steps: 0/' \
	  -e 's/max_steps: .*/max_steps: 256/' \
	  -e 's/eval_steps: .*/eval_steps: 256/' \
	  -e 's/save_steps: .*/save_steps: 256/' \
	  -e 's/num_train_epochs: .*/num_train_epochs: 100/' \
	  -e 's/do_eval: True/do_eval: False/' \
	  -e 's/eval_on_start: True/eval_on_start: False/' \
	  "signwriting_transcription/config.yaml" > "$@"

overfit: example/overfit_config.yaml
	wandb offline

# Docker commands (platform-agnostic)
.PHONY: docker-build docker-run docker-shell

DOCKER_IMAGE ?= signwriting-transcription
DOCKER_TAG ?= latest

## Build Docker image
docker-build: Dockerfile
	docker build --platform linux/amd64 -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

## Run command in Docker container
docker-run: docker-build
	docker run --rm -it \
		--gpus all \
		-v $(PWD):/workspace \
		-v $(DATA_DIR):/data \
		-e WANDB_PROJECT=$(WANDB_PROJECT) \
		-e HUGGINGFACE_TOKEN=$(HUGGINGFACE_TOKEN) \
		$(DOCKER_IMAGE):$(DOCKER_TAG) $(CMD)

## Open Docker shell
docker-shell:
	@$(MAKE) docker-run CMD="/bin/bash"

# Utilities (platform-agnostic)
.PHONY: clean setup lint test

## Clean temporary files
clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf example/data
	rm -rf example/results
	rm -f example/overfit_config.yaml
	rm -rf wandb

## Install package in development mode
setup:
	pip install -e ".[dev]"

## Run code linting
lint:
	ruff check signwriting_transcription/

## Run tests
test:
	pytest signwriting_transcription/