import os
import sys
import urllib.request
import subprocess

# ==============================
# CONFIGURATION
# ==============================

ACCESS_TOKEN = ""  # public upload, no token needed

BASE_URL = "https://zenodo.org/api/records/19401698/files"

FILES = [
    "METEO_daily.db.partaa",
    "METEO_daily.db.partab",
    "METEO_daily.db.partac",
    "METEO_daily.db.partad",
    "METEO_daily.db.partae",
    "METEO_daily.db.partaf",
]

EXTRA_FILES = [
    "database_schema.sql",
    "README.md",
]

DOCKER_IMAGE = "grezac/edilia:latest"

DATA_DIR = "data"
OUTPUT_DIR = "output"
DB_FILE = os.path.join(DATA_DIR, "METEO_daily.db")
SETUP_FLAG = os.path.join(DATA_DIR, ".setup_complete")

# ==============================
# UTILS
# ==============================


def build_url(filename):
    url = f"{BASE_URL}/{filename}/content"
    if ACCESS_TOKEN:
        url += f"?access_token={ACCESS_TOKEN}"
    return url


def download_file(url, dest):
    if os.path.exists(dest):
        print(f"[OK] Already exists: {dest}")
        return

    print(f"[DOWNLOAD] {url}")
    with urllib.request.urlopen(url) as response, open(dest, "wb") as out_file:
        total = response.length
        downloaded = 0

        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)

            if total:
                percent = downloaded * 100 // total
                print(f"\r  -> {percent}%", end="")

    print("\n[DONE]")


def check_docker():
    try:
        subprocess.run(["docker", "--version"], check=True, stdout=subprocess.DEVNULL)
    except Exception:
        print("[ERROR] Docker not found.")
        sys.exit(1)


def reconstruct_db():
    if os.path.exists(DB_FILE):
        print("[OK] Database already exists")
        return

    print("[RECONSTRUCT] Building database...")

    with open(DB_FILE, "wb") as outfile:
        for part in FILES:
            part_path = os.path.join(DATA_DIR, part)
            with open(part_path, "rb") as infile:
                while True:
                    chunk = infile.read(1024 * 1024)
                    if not chunk:
                        break
                    outfile.write(chunk)

    print("[DONE] Database ready")


# ==============================
# MAIN
# ==============================


def main():
    print(
        """
SETUP SCRIPT

This will:
- download ~12.5 GB of data
- reconstruct the database

Run once.

Press Enter to continue...
"""
    )
    input()

    check_docker()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Download
    for f in FILES + EXTRA_FILES:
        url = build_url(f)
        dest = os.path.join(DATA_DIR, f)
        download_file(url, dest)

    reconstruct_db()

    # Optional: pre-pull docker image
    print("[DOCKER] Pulling image...")
    subprocess.run(["docker", "pull", DOCKER_IMAGE], check=True)

    # Flag setup done
    open(SETUP_FLAG, "w").close()

    print("\n[SUCCESS] Setup completed.")
    print(
        "\nYou can now run the development environment using the command `python run.py`."
    )
    print("\nPlease refer to the `documentation/SCRIPTS.md` file for more information.")


if __name__ == "__main__":
    main()
