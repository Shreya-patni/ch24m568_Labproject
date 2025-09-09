# AILabProject – End-to-End MLOps (Windows, no WSL)

Train a Spark ML pipeline on Titanic data, register/export with MLflow, serve via FastAPI/Uvicorn, smoke-test with a Python client, and check data drift — all from **Windows Command Prompt** using `run_all.bat`.

> ✅ Designed to run in **Command Prompt** (also works in PowerShell with minor syntax tweaks)  
> ✅ No WSL required  
> ✅ Reproducible data snapshot via **DVC + Git tag `v1`**  
> ✅ One-shot pipeline runner: `run_all.bat`

---

## Table of Contents

- [Prerequisites] 
- [Project Layout]
- [Quick Start]
- [Step-by-Step]
- [Data Drift Detection]
- [Configuration]
- [Reproducibility]
- [Port & Process Management]
- [Troubleshooting]
- [Useful Commands]


---

## Prerequisites

- **Conda** (Miniconda/Anaconda)
- **Python 3.10**
- **Java JDK 17** (Temurin/Adoptium recommended)  
  Ensure `JAVA_HOME` is set and `%JAVA_HOME%\bin` is in `PATH`
- **Git**  
  Enable long paths:  
  ```bat
  git config --global core.longpaths true
  ```
- **DVC** (installed via `requirements.txt`)  
- **No WSL required**

> Spark on Windows requires the same Python as your Conda environment.  
> The `.bat` sets:
> ```bat
> set PYSPARK_PYTHON=%CONDA_PREFIX%\python.exe
> set PYSPARK_DRIVER_PYTHON=%CONDA_PREFIX%\python.exe
> ```

---

## Project Layout

```
AILabProject-main/
├─ training/
│  ├─ training.py           # Train, log to MLflow, register + export model
│  ├─ preprocess.py         # Spark feature pipeline builder
│  ├─ spark_session.py      # Spark session factory (local mode)
│  └─ utils.py              # Helper functions, reference profile builder
├─ deployment/
│  ├─ app.py                # FastAPI app (loads exported Spark model)
│  ├─ test.py               # HTTP client: /predict → artifacts_local/test_predictions.csv
│  └─ data_drift_detection.py # Drift checker → artifacts_local/drift_report.json
├─ data/
│  ├─ train.csv.dvc
│  └─ test.csv.dvc
├─ deployment/model/         # Exported MLflow model
├─ mlruns/                   # Local MLflow tracking
├─ artifacts_local/          # Reports, confusion matrices, predictions, drift report
├─ config.yaml               # Paths/configuration
├─ requirements.txt
├─ run_all.bat               # One-shot pipeline runner
└─ README.md
```

---

## Quick Start

Open **Command Prompt**, navigate to project root, and run:

```bat
run_all.bat v1
```

This will:

1. Ensure DVC cache and materialize data if needed  
2. Install dependencies (unless `nodeps` is passed)  
3. Train and register the best model → export to `deployment/model/`  
4. Start API in a new window titled `AILab_API` (default port 8000)  
5. Smoke-test the `/predict` endpoint → `artifacts_local/test_predictions.csv`

---

## Step-by-Step Instructions

### 1) Create and activate Conda environment
```bat
conda create -n id5003w python=3.10 -y
conda activate id5003w
```

### 2) Install dependencies
```bat
python -m pip install -r requirements.txt
```

### 3) Prepare data + DVC snapshot (v1)

Place `train.csv` and `test.csv` in `data/` if missing, then run:

```bat
dvc add data\train.csv
dvc add data\test.csv
git add data\*.dvc .dvc .dvcignore .gitignore .gitattributes
git commit -m "Track raw data with DVC"
git tag -f v1
```

After this, `DATA_REV=v1` represents your reproducible data snapshot.

---

### 4) Train only
```bat
set DATA_REV=v1
python -m training.training
```

This logs to `mlruns/`, registers model `TitanicClassifier`, and exports best Spark model to `deployment/model/`.

---

### 5) Serve API only
```bat
set PYSPARK_PYTHON=%CONDA_PREFIX%\python.exe
set PYSPARK_DRIVER_PYTHON=%CONDA_PREFIX%\python.exe
uvicorn deployment.app:app --host 127.0.0.1 --port 8000
```

---

### 6) Smoke test only

Ensure API is running at `http://127.0.0.1:8000`, then:

```bat
set API_URL=http://127.0.0.1:8000/predict
python deployment\test.py
```

Predictions saved to `artifacts_local/test_predictions.csv`.

---

### 7) Full pipeline (Train → Serve → Test)
```bat
run_all.bat [DATA_REV] [nodeps] [PORT]
```

Defaults:  
- `DATA_REV=v1`  
- `PORT=8000`

Examples:
```bat
run_all.bat
run_all.bat v1
run_all.bat v1 nodeps
run_all.bat v1 nodeps 8080
```

---

## Data Drift Detection

Compare current data to reference profile using KS (numerical) and Chi-square (categorical).

```bat
python deployment\data_drift_detection.py ^
  --input data\test.csv ^
  --out artifacts_local\drift_report.json ^
  --threshold 0.3 ^
  --pval 0.05
```

Optional feature weights:
```json
{ "Age": 0.2, "Fare": 0.2, "Pclass": 0.2, "SibSp": 0.2, "Parch": 0.2 }
```

---

## Configuration

Minimal `config.yaml` example:
```yaml
paths:
  reference_profile: deployment/reference_profile.json
```

---

## Reproducibility

Use DVC + Git tag `v1` to pin data version.

To reproduce same run later:
```bat
git checkout v1
set DATA_REV=v1
python -m training.training
```

---

## Port & Process Management

Check listening process:
```bat
netstat -ano | findstr :8000
```

Kill process by PID:
```bat
taskkill /PID <PID> /F
```

Stop API window started by `run_all.bat`:
```bat
taskkill /FI "WINDOWTITLE eq AILab_API" /F
```

---

## Troubleshooting

- Ensure `JAVA_HOME` points to JDK 17  
- Ensure PYSPARK env vars point to Conda Python  
- Allow localhost ports in firewall/antivirus  
- Restore empty test.csv:
    ```bat
    python -c "import os, shutil, dvc.api; src=dvc.api.get_url('data/test.csv', repo=None, rev=os.environ.get('DATA_REV','v1')); import pathlib; p=pathlib.Path('data/test.csv'); p.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src,p); print('Restored')"
    ```
- Fix missing Git tag `v1`:
    ```bat
    git tag -f v1
    ```

---

## Useful Commands

List MLflow model versions:
```bat
python -c "import mlflow; from mlflow.tracking import MlflowClient as C; c=C(); [print(v) for v in c.get_latest_versions('TitanicClassifier',['None','Staging','Production','Archived'])]"
```

View last predictions:
```bat
python -c "import pandas as pd; print(pd.read_csv('artifacts_local/test_predictions.csv').head())"
```

Start API on different port (foreground):
```bat
uvicorn deployment.app:app --host 127.0.0.1 --port 8080 --log-level debug
```