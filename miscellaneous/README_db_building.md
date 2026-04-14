# EDILIA

## Overview

**EDILIA** is a fully reproducible computational framework designed for next-day minimum temperature forecasting using automated machine learning and spatial predictor selection.

This repository provides:

* Python scripts for data processing and model execution
* Configuration-driven experiment setup
* A fully reproducible execution pipeline via Docker
* Automated environment setup and data acquisition (`install.py`)
* Standardized execution entry point (`run.py`)

The framework is designed to ensure **full reproducibility across platforms** (Linux, Windows) with minimal user intervention.

---

## Associated Publication

This repository accompanies the study:

> *Spatial Predictor Selection for Next-Day Minimum Temperature Forecasting: An Automated Machine Learning Framework Applied Across European Climate Regimes*

---

## Reproducibility

Reproducibility is ensured through:

* Archived dataset on Zenodo (DOI)
* Version-controlled source code
* Docker-based execution environment
* Deterministic numerical configuration (controlled threading)
* Explicit two-step workflow (`install.py` → `run.py`)

---

## Requirements

* Python 3
* Docker (must be installed and accessible from command line)

No additional Python environment setup is required.

---

## Quick Start

Clone the repository:

```bash
git clone https://github.com/grezac/edilia
cd edilia