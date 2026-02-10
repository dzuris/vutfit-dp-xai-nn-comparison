# ------------------------------------------------------------------
# File: Makefile
# Author: Adam Dzurilla <xdzuri00@stud.fit.vutbr.cz>
# Created: 9.3.2025
# Description: Build script
# ------------------------------------------------------------------

# Variables
INTERPRETER:=python3

# Configurations
CONFIG_FILE_MODEL:=src/model/config_training.yaml
CONFIG_FILE_XAI:=src/xai/config_xai.yaml
CONFIG_FILE_SLS:=src/sls/config_sls.yaml

# TMP folders
TMP_DIR:=tmp
MODELS_DIR:=models
LOGS_DIR:=logs
TEST_RESULTS_DIR:=test_results

# Targets
.PHONY: all xai sls test test-sls doc clean-doc clean

model:
	$(INTERPRETER) -m src.model.main --config $(CONFIG_FILE_MODEL)

xai:
	$(INTERPRETER) -m src.xai.main --config $(CONFIG_FILE_XAI)

sls:
	$(INTERPRETER) -m src.sls.main --config $(CONFIG_FILE_SLS)

test:
	$(INTERPRETER) -m test_training_xai

test-sls:
	$(INTERPRETER) -m src.sls.test_sls

doc: clean-doc
	@$(INTERPRETER) -m pydoc -w $(shell find src -name '*.py' -print | sed 's|/|.|g;s|\.py$$||' | tr '\n' ' ') test_training_xai
	@mv *.html docs_pydoc

clean-doc:
	@find docs_pydoc -type f ! -name "index.html" -delete

clean: clean-doc
	@echo "Cleaning directories \"$(LOGS_DIR) $(MODELS_DIR) $(TMP_DIR)\"..."
	@rm -rf $(LOGS_DIR)
	@rm -rf $(MODELS_DIR)
	@rm -rf $(TMP_DIR)
	@rm -rf $(TEST_RESULTS_DIR)
	@echo "Cleaning completed."
