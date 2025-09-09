import subprocess
import threading
import uvicorn
import time


def run_mlflow():
    """Start MLflow tracking server."""
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "./artifacts",
        "--host", "0.0.0.0",
        "--port", "5000"
    ]
    subprocess.run(cmd)


def run_fastapi():
    """Start FastAPI model deployment service."""
    uvicorn.run("deployment.app:app", host="0.0.0.0", port=8000, reload=False)


def run_training():
    """Trigger model training pipeline."""
    print("Starting training pipeline...")
    subprocess.run(["python", "-m", "training.training"])
    print("Training finished.")


if __name__ == "__main__":
    # Start MLflow server in background
    threading.Thread(target=run_mlflow, daemon=True).start()
    print("MLflow starting...")
    time.sleep(3)

    # Run training once
    run_training()

    # Launch API service
    run_fastapi()
