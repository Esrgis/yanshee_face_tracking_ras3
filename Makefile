# Makefile - Yanshee Face Tracking
# Yeu cau: python trong PATH

PYTHON   = python
SRC      = 0
DURATION = 60
CONFIGS  = ABCD

run:
	$(PYTHON) main_tracker.py

robot:
	$(PYTHON) main_tracker_robot.py

ablation:
	$(PYTHON) scripts/run_ablation.py --source $(SRC) --duration $(DURATION) --configs $(CONFIGS)

analyze:
	$(PYTHON) scripts/analyze_results.py

study: ablation analyze

collect:
	$(PYTHON) scripts/data_collector.py

.PHONY: run robot ablation analyze study collect
