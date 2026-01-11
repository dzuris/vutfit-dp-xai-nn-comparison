# ------------------------------------------------------------------
# File: Makefile
# Author: Adam Dzurilla <xdzuri00@stud.fit.vutbr.cz>
# Created: 9.3.2025
# Description: Build script
# ------------------------------------------------------------------

# Variables

INTERPRETER:=python3
CONFIG_FILE:=src/config.yaml
CONFIG_FILE_XAI:=src/config_xai.yaml
TMP_DIR:=tmp
MODELS_DIR:=models
LOGS_DIR:=logs

# Targets
.PHONY: all xai clean

model:
	$(INTERPRETER) -m src.main --config $(CONFIG_FILE)

xai:
	$(INTERPRETER) -m src.xai --config $(CONFIG_FILE_XAI)

txtexp:
	$(INTERPRETER) -m src.txt_explain_alg

clean:
	@echo "Cleaning directories \"$(LOGS_DIR) $(MODELS_DIR) $(TMP_DIR)\"..."
	@rm -rf $(LOGS_DIR)/*
	@rm -rf $(MODELS_DIR)/*
	@rm -rf $(TMP_DIR)/*
	@echo "Cleaning completed."
