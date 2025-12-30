# ------------------------------------------------------------------
# File: Makefile
# Author: Adam Dzurilla <xdzuri00@stud.fit.vutbr.cz>
# Created: 9.3.2025
# Description: Build script
# ------------------------------------------------------------------

# Variables

INTERPRETER:=python3
CONFIG_FILE:=src/config.yaml
TMP_DIR:=tmp
MODELS_DIR:=models
LOGS_DIR:=logs

# Targets
.PHONY: all xai clean

model: src/main.py
	$(INTERPRETER) $< --config $(CONFIG_FILE)

xai: src/xai.py
	$(INTERPRETER) $< --config $(CONFIG_FILE)

clean:
	@echo "Cleaning directories \"$(LOGS_DIR) $(MODELS_DIR) $(TMP_DIR)\"..."
	@rm -rf $(LOGS_DIR)/*
	@rm -rf $(MODELS_DIR)/*
	@rm -rf $(TMP_DIR)/*
	@echo "Cleaning completed."
