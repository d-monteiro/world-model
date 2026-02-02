# Physical AI - World Models for Robotic Arm

3-joint robotic arm manipulation using World Models (VAE + MDN-RNN + CMA-ES) with MuJoCo simulation.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

## Training Pipeline

```bash
python scripts/01_collect_vae_data.py    # Collect 1M state samples
python scripts/02_train_vae.py           # Train VAE
python scripts/03_collect_rnn_data.py    # Encode episodes for RNN
python scripts/04_train_rnn.py           # Train MDN-RNN
python scripts/05_train_controller.py    # Train controller via CMA-ES
python scripts/06_run_agent.py           # Run trained agent
python scripts/07_evaluate.py            # Evaluate over 100 episodes
```
