# ------------------------------------------------------------------
# File: Makefile
# Author: Adam Dzurilla <xdzuri00@stud.fit.vutbr.cz>
# Created: 9.3.2025
# Description: Build script
# ------------------------------------------------------------------

# Variables
INTERPRETER:=python3
CONFIG_FILE_TRAINING:=src/training/config_training.yaml
CONFIG_FILE_XAI:=src/xai/config_xai.yaml
CONFIG_FILE_TXTEXP:=src/sls/config_sls.yaml
TMP_DIR:=tmp
MODELS_DIR:=models
LOGS_DIR:=logs
TEST_RESULTS_DIR:=test_results

# Targets
.PHONY: all xai clean

model:
	$(INTERPRETER) -m src.training.main --config $(CONFIG_FILE_TRAINING)

xai:
	$(INTERPRETER) -m src.xai.main --config $(CONFIG_FILE_XAI)

sls:
	$(INTERPRETER) -m src.sls.main --config $(CONFIG_FILE_TXTEXP)

test-sls:
	$(INTERPRETER) -m src.sls.test_sls

clean:
	@echo "Cleaning directories \"$(LOGS_DIR) $(MODELS_DIR) $(TMP_DIR)\"..."
	@rm -rf $(LOGS_DIR)/*
	@rm -rf $(MODELS_DIR)/*
	@rm -rf $(TMP_DIR)/*
	@rm -rf $(TEST_RESULTS_DIR)
	@echo "Cleaning completed."
