# EDILIA

## 1. Scope

**EDILIA** is a computational framework for next-day minimum temperature forecasting based on automated machine learning and spatial predictor selection.

The framework is designed to ensure **full computational reproducibility**, enabling independent verification of all numerical experiments reported in the associated publication.

It provides:

- deterministic data acquisition
- configuration-driven experiment definition
- containerized execution environment
- strict separation between installation and execution phases

---

## 2. Associated Publication

*Spatial Predictor Selection for Next-Day Minimum Temperature Forecasting: An Automated Machine Learning Framework Applied Across European Climate Regimes*

---

## 3. Reproducibility Statement

Reproducibility is ensured through:

- **Data provenance**: archived dataset with DOI (Zenodo)
- **Code traceability**: version-controlled repository and tagged release
- **Execution environment**: Docker container
- **Deterministic behavior**: controlled numerical threading
- **Workflow integrity**: explicit separation between installation (`install.py`) and execution (`run.py`)

No undocumented steps or manual preprocessing are required.

---

## 4. System Requirements

- Python 3
- Docker (daemon running)

No additional dependencies are required on the host system.

---

## 5. Workflow

### 5.1 Repository Setup

```bash
git clone https://github.com/grezac/edilia
cd edilia
```

### 5.2 Installation Phase

```bash
python install.py
```

This step performs:

- download of the dataset (~12.5 GB)
- reconstruction of the SQLite database
- creation of required directories

A marker file is created upon completion:

```
data/.setup_complete
```

---

### 5.3 Execution Phase

```bash
python run.py
```

The script:

- verifies the presence of `data/.setup_complete`
- aborts if installation is incomplete
- retrieves the Docker image (if needed)
- launches a container with mounted directories:
  - `/app`
  - `/data`
  - `/output`

Execution is performed inside the container.

---

## 6. Project Structure

```
edilia/
├── configs/           # Experiment configurations
├── data/              # Data and setup marker
├── documentation/     # Technical documentation
├── examples/          # Samples of generated files
├── extras/            # Auxiliary scripts
├── libs/              # Core modules
├── scripts/           # Execution scripts
├── install.py         # Installation phase
├── run.py             # Execution phase
├── requirements.txt   # Informational
└── README.md
```

---

## 7. Configuration

Experiments are defined via JSON files in:

```
configs/
```

These fully specify:

- predictor variables
- model hyperparameters
- temporal windows
- execution settings

No hidden parameters are used.

---

## 8. Outputs

Outputs are written to:

```
data/output/
```

They include:

- predictions
- performance metrics
- model metadata

---

## 9. Data Availability

https://doi.org/10.5281/zenodo.19401698

Includes:

- ERA5 variables
- NOAA NCEI observations
- SQLite database (~12.5 GB)

---

## 10. Code Availability

Development repository:
https://github.com/grezac/edilia

Archived version:
[TO BE FILLED]

---

## 11. Execution Environment

The environment is fully containerized using Docker:

- ensures cross-platform reproducibility
- eliminates dependency conflicts
- guarantees consistent numerical behavior

---

## 12. Practical Considerations

- Initial download size: ~12.5 GB
- Recommended disk space: >15 GB
- Docker must be running before execution

---

## 13. Computational Reproducibility Checklist

### 13.1 Code and Versioning

- [ ] Tagged version of the code is used
- [ ] Archived version with DOI is available
- [ ] No local modifications required

### 13.2 Data

- [ ] Data available via Zenodo
- [ ] No manual preprocessing required
- [ ] Database reconstruction automated

### 13.3 Environment

- [ ] Docker container used
- [ ] No host dependency installation
- [ ] Deterministic numerical configuration

### 13.4 Workflow

- [ ] `install.py` executed successfully
- [ ] `data/.setup_complete` present
- [ ] `run.py` verifies setup before execution

### 13.5 Configuration

- [ ] JSON configuration fully defines experiments
- [ ] No hidden parameters

### 13.6 Randomness Control

The experiments use fixed random seeds:

```
1, 2, 3, 4, 5
```

- [ ] All stochastic components are controlled
- [ ] Results reproducible per seed

### 13.7 Outputs

- [ ] Outputs written to `data/output/`
- [ ] Metrics and predictions available
- [ ] Results traceable to configuration

### 13.8 Hardware

- [ ] CPU execution (no GPU required)
- [ ] Cross-platform reproducibility

### 13.9 Reproducibility Outcome

Successful reproduction implies:

- identical numerical results (within floating-point precision)
- identical selected predictors
- identical performance metrics for each seed

---

## 14. License

CC BY 4.0

---

## 15. Contact

Eric Duhamel  
edilia12380@gmail.com
