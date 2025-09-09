import os, shutil
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts

MODEL_NAME = "TitanicClassifier"
STAGE = "Staging"
EXPORT_DIR = os.path.join("deployment", "model")

client = MlflowClient()
vers = client.get_latest_versions(MODEL_NAME, [STAGE])
if not vers:
    raise SystemExit(f"No versions in stage {STAGE}")
mv = vers[0]
print("Using version:", mv.version, "run:", mv.run_id)

# In training we logged the Spark model under artifact_path='spark-model'
src = download_artifacts(run_id=mv.run_id, artifact_path="spark-model")

# Copy so that MLmodel sits directly under deployment/model
if os.path.exists(EXPORT_DIR):
    shutil.rmtree(EXPORT_DIR)
shutil.copytree(src, EXPORT_DIR)
print("Exported files at deployment/model:", os.listdir(EXPORT_DIR))
