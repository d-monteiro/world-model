"""Step 1: Collect random state data for VAE training."""

import os
import numpy as np
from physical_ai.data.collector import collect_random_data

N_SAMPLES = 1_000_000
SEED = 42
N_WORKERS = 4
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Collecting {N_SAMPLES} samples with {N_WORKERS} workers...")
    data = collect_random_data(n_samples=N_SAMPLES, seed=SEED, n_workers=N_WORKERS)

    output_path = os.path.join(OUTPUT_DIR, "vae_dataset.npy")
    np.save(output_path, data)
    print(f"Saved dataset: {output_path}, shape: {data.shape}")
    print(f"Stats: mean={data.mean(axis=0)}, std={data.std(axis=0)}")


if __name__ == "__main__":
    main()
