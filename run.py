import os
import sys
import subprocess

DATA_DIR = "data"
DB_FILE = os.path.join(DATA_DIR, "METEO_daily.db")
SETUP_FLAG = os.path.join(DATA_DIR, ".setup_complete")
DOCKER_IMAGE = "grezac/edilia:latest"


def check_docker():
    try:
        subprocess.run(["docker", "--version"], check=True, stdout=subprocess.DEVNULL)
    except Exception:
        print("[ERROR] Docker not installed.")
        sys.exit(1)


def check_ready():
    if not os.path.exists(SETUP_FLAG):
        print("[ERROR] Setup not completed. Run install.py first.")
        sys.exit(1)

    if not os.path.exists(DB_FILE):
        print("[ERROR] Database missing. Run install.py.")
        sys.exit(1)


def run_docker():
    print("[DOCKER] Starting container...")

    project_dir = os.path.abspath(".")
    data_dir = os.path.abspath(DATA_DIR)

    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-it",
            "-e",
            "OMP_NUM_THREADS=1",
            "-e",
            "MKL_NUM_THREADS=1",
            "-e",
            "OPENBLAS_NUM_THREADS=1",
            "-v",
            f"{project_dir}:/app",
            "-v",
            f"{data_dir}:/data",
            DOCKER_IMAGE,
            "bash",
        ],
        check=True,
    )


def main():
    check_docker()
    check_ready()
    run_docker()


if __name__ == "__main__":
    main()
