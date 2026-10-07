#!/usr/bin/env bash
# Runs the whole pipeline from the repo root. Takes about a minute on a laptop CPU.
set -e
mkdir -p ml_pipeline/data
for s in 01_eda 02_prep 03_model 04_error_analysis 05_final_exam 06_business_impact; do
  echo "== $s"; python ml_pipeline/$s.py
done
