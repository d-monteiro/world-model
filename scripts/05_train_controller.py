"""Step 5: Train the controller using CMA-ES in the dream world."""

import os
import numpy as np
import torch
import cma

from physical_ai.models.vae import StateVAE, LATENT_DIM
from physical_ai.models.mdnrnn import MDNRNN
from physical_ai.controller.controller import Controller, dream_evaluate

# Hyperparameters
SIGMA0 = 0.5
POPULATION_SIZE = 64
N_ROLLOUTS = 16
HORIZON = 50
MAX_GENERATIONS = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load VAE
    vae = StateVAE(latent_dim=LATENT_DIM).to(DEVICE)
    vae.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "vae.pth"), map_location=DEVICE, weights_only=True))
    vae.eval()

    # Load MDN-RNN
    mdnrnn = MDNRNN().to(DEVICE)
    mdnrnn.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "mdnrnn.pth"), map_location=DEVICE, weights_only=True))
    mdnrnn.eval()

    # Initialize controller
    controller = Controller().to(DEVICE)
    n_params = controller.get_n_params()
    print(f"Controller has {n_params} parameters")

    # CMA-ES
    initial_params = controller.get_params()
    es = cma.CMAEvolutionStrategy(
        initial_params,
        SIGMA0,
        {"popsize": POPULATION_SIZE, "seed": 42},
    )

    best_fitness = float("inf")
    generation = 0

    while not es.stop() and generation < MAX_GENERATIONS:
        solutions = es.ask()

        fitnesses = []
        for params in solutions:
            controller.set_params(params)
            fitness = dream_evaluate(
                controller, vae, mdnrnn,
                n_rollouts=N_ROLLOUTS,
                horizon=HORIZON,
                device=DEVICE,
            )
            fitnesses.append(fitness)

        es.tell(solutions, fitnesses)

        gen_best = min(fitnesses)
        gen_mean = np.mean(fitnesses)
        if gen_best < best_fitness:
            best_fitness = gen_best
            # Save best controller
            best_idx = np.argmin(fitnesses)
            controller.set_params(solutions[best_idx])
            torch.save(
                controller.state_dict(),
                os.path.join(CHECKPOINT_DIR, "controller.pth"),
            )

        generation += 1
        if generation % 10 == 0 or generation == 1:
            print(
                f"Gen {generation} | "
                f"Best: {gen_best:.4f} | "
                f"Mean: {gen_mean:.4f} | "
                f"Overall Best: {best_fitness:.4f} | "
                f"Sigma: {es.sigma:.4f}"
            )

    # Save final best
    print(f"\nCMA-ES finished after {generation} generations.")
    print(f"Best fitness (neg reward): {best_fitness:.4f}")
    print(f"Saved controller to {CHECKPOINT_DIR}/controller.pth")


if __name__ == "__main__":
    main()
