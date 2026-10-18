#!/usr/bin/env bash
# Reproduce everything (about 8 min on 2 CPU cores; the model step dominates)
set -e; cd "$(dirname "$0")"
python3 01_validate.py && python3 02_analysis.py > ../outputs/analysis_log.txt && python3 03_model.py > ../outputs/model_log.txt && python3 04_model_summary_figure.py
