# **Hackathon: 2D Robotic Arm World Model**

## Overview

A 2-day hackathon project focused on **world models for robotics**. The goal is to create a **2D robotic arm environment** where the agent can **pick up objects and plan actions using mental simulation in latent space**. This project demonstrates **how a robot can “imagine” trajectories before acting**, using a Variational Autoencoder (VAE) and a dynamics model.

---

## 🏁 End Goal

By the end of the hackathon, you will have:

1. A **2D robotic arm simulator** (vector-based or small 32×32 image observations).  
2. A **VAE** that compresses environment states into a compact **latent vector**.  
3. A **dynamics model** predicting future latent states given actions.  
4. A **mental rollout demo**: the arm “imagines” pick-and-place trajectories.  
5. (Optional) A **simple planner/controller** using latent rollouts to move objects to goal positions.  

---

## Hackathon Objectives

- Learn to encode high-dimensional states into **latent space**.  
- Build a **predictive model of environment dynamics**.  
- Enable **planning in latent space** without running the environment.  
- Gain hands-on experience with VAE, RNN/MLP dynamics, and simple robotic control.  

---

## 🛠 Architecture

**1. Variational Autoencoder (VAE)**  

- **Encoder:** compresses the environment state → latent vector `z`.  
- **Decoder:** reconstructs the environment state from `z`.  
- **Loss:** reconstruction + KL divergence.  

**2. Dynamics Model**  

- Input: `(z_t, a_t)` → predicts `z_{t+1}`.  
- Can be an MLP or small RNN.  
- Trained to minimize prediction error between predicted `z_{t+1}` and encoded `z_{t+1}`.  

**3. Controller / Planner (Optional)**  

- Uses mental rollouts in latent space to plan actions to move objects to goals.  
- Can be simple MPC or RL agent using latent predictions.

---

## ⚡ Optional Extensions

- **Curiosity-driven exploration:** reward arm for states with high prediction error.  
- **Multiple objects:** more complex latent dynamics.  
- **Visualization of latent space:** compress to 2D and see trajectories.  
- **Image-based VAE:** add visual realism to predictions.  

---

## 🔗 Outcome

By the end of the hackathon, we will have a **functional world model** for a 2D robotic arm — a **proof-of-concept for mental simulation and planning**.
