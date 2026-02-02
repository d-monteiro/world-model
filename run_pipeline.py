"""
Master Pipeline Script for World Model Hackathon.

Steps:
1. collect_data.py -> Generates data/transitions.pkl
2. train_vae.py    -> Trains VAE on arm joints, saves to src/models/vae_weights.pth
3. train_rnn.py    -> Trains MDN-RNN on sequences, saves to src/models/rnn_weights.pth
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run_step(script_path, description):
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Running {script_path}...")
    print(f"{'='*60}\n")
    
    try:
        # Check if file exists
        if not (ROOT / script_path).exists():
             print(f"ERROR: Script {script_path} not found!")
             return False

        # Execute
        result = subprocess.run(
            [sys.executable, str(ROOT / script_path)], 
            cwd=str(ROOT),
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FAILED: {description} (Exit code {e.returncode})")
        return False

def main():
    # 1. Collect Data
    # Note: Depending on how many episodes/steps are set in the file, this might take time.
    if not run_step("src/training/collect_data.py", "Data Collection"):
        return

    # 2. Train VAE
    if not run_step("src/training/train_vae.py", "Train VAE (Body Model)"):
        return

    # 3. Train RNN
    if not run_step("src/training/train_rnn.py", "Train MDN-RNN (World Model)"):
        return

    print("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("Models are saved in src/models/")

if __name__ == "__main__":
    main()
