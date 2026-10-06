"""
Modal training script.
Replace placeholders marked with <...> before running.
"""

import modal
import os
from pathlib import Path


# --- Configuration ---
REPO_NAME = "localization"

LOCAL_REPO_DIR = str(Path(__file__).resolve().parents[1])
LOCAL_DATA_DIR = "/var/tmp/localization/data/pt"
LOCAL_OUTPUT_DIR = "/var/tmp/localization/modal"
 
MODAL_ROOT = "/root/localization_run"
MODAL_DATA_DIR = f"{MODAL_ROOT}/data/pt"
MODAL_OUTPUT_DIR = f"{MODAL_ROOT}/output"
 
# Modal Volume — persists between runs
volume = modal.Volume.from_name("localization-vol", create_if_missing=True)
#modal volume dashboard localization-vol
# --- Image: install deps from repo (cached), then overlay local code fresh on every run ---
image = (
    modal.Image.debian_slim(python_version="3.13.5")
    .env({"PYTHONPATH": f"/root/{REPO_NAME}"})
    #.apt_install("git")
    # Mount your local code fresh on every run (no rebuild needed when you change code)
    .add_local_file(
        f"{LOCAL_REPO_DIR}/requirements.txt",
        remote_path=f"/root/requirements.txt",
        copy=True,  # copy=True allows run_commands to follow
    )
    .run_commands(
        # Clone once just to get requirements.txt — cached after first run
        #f"git clone --branch sharghif {GITHUB_URL} /root/",
        f"pip install -r /root/requirements.txt",
    )
    .add_local_dir(LOCAL_REPO_DIR, remote_path=f"/root/{REPO_NAME}")
    


    
)
 
app = modal.App("localization-training", image=image)
 
 
@app.function(
    gpu="L40S",
    timeout=60 * 60 * 6,  # 6 hours max
    volumes={MODAL_ROOT: volume},
)
def train():
    import sys
    import functools

    # Force unbuffered printing so Modal captures output streaming
    #print = functools.partial(print, flush=True)
    import os
    import sys
 
    # Set ROOT_DIR so the repo resolves all relative paths correctly
    os.environ["ROOT_DIR"] = MODAL_ROOT
    sys.path.insert(0, f"/root/{REPO_NAME}")
 
    # Import your training code
    from src.localization.config.presets import config_dichasus
    from src.localization.training.trainer import run
 
    # Make sure output dir exists
    os.makedirs(MODAL_OUTPUT_DIR, exist_ok=True)
 
    print(f"ROOT_DIR = {MODAL_ROOT}")
    print(f"Data expected at: {MODAL_DATA_DIR}")
    print("Starting training...")
    config_dichasus.mask_ratio_train = 0.9
    config_dichasus.max_epochs = 200
    config_dichasus.train_batch_size = 256
    config_dichasus.learning_rate = 0.001
    config_dichasus.val_batch_size = 256
    config_dichasus.loss_scheduler_start = 0.000001
    config_dichasus.loss_scheduler_steps = 800
    config_dichasus.deterministic = True
    config_dichasus.coeff_time_loss = 0.0
    config_dichasus.coeff_mse_loss = 1.0
    config_dichasus.coeff_mixup_loss = 0.0
    config_dichasus.coeff_boundary_loss = 0.0
    config_dichasus.coeff_dissimilarity_loss = 1.0

    config_dichasus.aggregator_name = "AlignedMTL"
    run(config_dichasus)
 
    print("Training complete. Results are in the Modal Volume.")
    volume.commit()  # flush writes so they persist
 

import subprocess
import os

def download_volume_folder(volume_name: str, remote_path: str, local_path: str):
    os.makedirs(local_path, exist_ok=True)
    print(f"Downloading {remote_path} from volume '{volume_name}'...")
    
    result = subprocess.run([
        "modal", "volume", "get",
        volume_name,
        remote_path,
        local_path
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Download completed successfully!")
    else:
        print("❌ Download failed:")
        print(result.stderr)
    
@app.local_entrypoint()
def main():

    # 1. Upload local data to Modal Volume
    print(f"Uploading data from {LOCAL_DATA_DIR} ...")
    with volume.batch_upload(force=True) as batch:
        batch.put_directory(LOCAL_DATA_DIR, "data/pt")
    print("Upload complete.")
 
    # 2. Run training on Modal
    print("Launching training on Modal (T4 GPU)...")
    train.remote()
    print("Training finished.")
 
    # 3. Download results back to local PC
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
    print(f"Downloading results to {LOCAL_OUTPUT_DIR} ...")
    # Download everything from the volume except the data folder
    for entry in volume.listdir("/"):
        name = entry.path
        if name == "data":
            continue  # skip — we already have this locally
        local_target = os.path.join(LOCAL_OUTPUT_DIR, name)
        download_volume_folder(volume.name, name, local_target)
        print(f"  Downloaded: {name}")

    print(f"Done. Model and logs are in {LOCAL_OUTPUT_DIR}")
