## Project Structure

```text
Simulacrum-A-Deep-Learning-N-Body-Simulator/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── flybyData.csv       # data generated from montecarlo trials
│   ├── X_scaler.pkl             
│   └── y_scaler.pkl 
│
├── models/
│   ├── solar_system_ai.keras
│
├── src/
│   ├── physicsEngine.py #rk4 + newton's gravitational law
│   ├── datagen.py #montecarlo simulation to generate data set
│   └── nasaData.py # gets data from nasa
│
├── training/
│   └── SimulacrumNeural.ipynb
│
└── visualization/
    └── spaceGraphics.py
```
# Simulacrum: Neural N-Body Simulator


https://github.com/user-attachments/assets/13265ec8-42cb-48f2-ad03-274a7e7a81b4



# Deep-Learning Orbital Mechanics
Simulacrum is a hybrid physics + deep learning system that uses a feed-forward neural network as a surrogate model to approximate chaotic N-body dynamics.

## Overview
Simulacrum is designed to simulate chaotic N-body gravitational interactions. The goal is to approximate the computational cost of traditional N-body simulations (O(N²) complexity per timestep) using a Feed-Forward Neural Network surrogate model 

While the underlying physics engine uses 4th-order Runge-Kutta (RK4) integration to generate high-fidelity training data, the trained neural surrogate model predicts planetary state transitions in constant time per system update (O(1) inference for this fixed 10-body architecture). In general, neural surrogate models scale with input dimensionality and architectural complexity rather than explicit pairwise interaction count.

<img width="1189" height="698" alt="efficiency" src="https://github.com/user-attachments/assets/82c3e9a0-a7bf-4ebf-b968-1e84329e7bf5" />

## Key Features
* **High-Precision Physics:** Implements RK4 integration for planetary dynamics with state vectors sourced from NASA’s NAIF SPICE kernels via `spiceypy`.
* **Parallelized Data Pipeline:** Utilizes Python's concurrent.futures to run a Monte Carlo engine across parallel processes, generating a 32MB dataset of 30,000 unique gravitational flyby scenarios across 17,000+ timesteps per trial
* **Deep Neural Surrogate Model:** A 5-layer feed-forward neural network that maps 17 initial state features to 54 continuous kinematic variables, approximating chaotic perturbations with a global mean relative error of < 1%
* **Real-Time Visualization:** Interactive PyGame environment for rendering planetary trajectories and AI predictions.

## Technical Architecture
### The Physics Engine
The baseline simulation calculates gravitational acceleration using Newton’s Universal Law of Gravitation:

$$F = G \frac{m_1 m_2}{r^2}$$

To maintain numerical stability, the engine uses RK4 integration to evolve positions and velocities over time, ensuring physically consistent trajectories prior to model training.

## Data Pipeline & Ingestion

To train the neural net, a  data generation pipeline was made to simulate chaotic N-body interactions at scale:

- **Ingestion:**  
  Programmatically fetches and parses NASA NAIF SPICE kernels to establish ground-truth initial conditions.

- **Simulation Engine:**  
  Generates randomized "rogue interloper" parameters (mass, velocity, trajectory).

- **Parallel Execution:**  
  Uses Python multiprocessing to compute 30,000 distinct universes simultaneously, utilizing RK4 integration to step forward in time and capture final planetary states.

### The Neural Network
* **Input Layer:** Interloper mass, time horizon, 3D position/velocity vectors, and initial scalar distances to all major celestial bodies.
* **Hidden Layers:** 5-layer Dense architecture with ReLU activation.
* **Output Layer (54 Variables):** Multi-target regression predicting 6 kinematic state vectors (X, Y, Z, VX, VY, VZ) for 9 distinct celestial bodies.
* **Performance:**
    * **Mean Position Error (MAE):** 1,236,958 km (on a solar-system scale).
    * **Global Mean Relative Error:** 0.0097.
    * **Time Complexity:** O(1) inference per system state update, bypassing the O(N²) bottleneck of traditional numerical integrators.

## Model Evaluation & Error Breakdown
<img width="845" height="546" alt="Learning Curve" src="https://github.com/user-attachments/assets/1456f022-e962-49d7-b044-2c09c506cf38" />


The network achieves a Global Relative Error of **0.97%**. However, accuracy varies significantly depending on the chaotic nature of the orbit. Inner planets experience higher variance due to tighter gravitational constraints, while outer planets remain highly predictable.

| Celestial Body | Mean Absolute Error (km) | Relative Error |
| :--- | :--- | :--- |
| **Sun** | 770 | 0.0010 |
| **Mercury** | 2,329,908 | 0.0396 |
| **Venus** | 2,309,841 | 0.0214 | 
| **Earth** | 2,314,337 | 0.0155 |
| **Mars** | 1,850,542 | 0.0081 |
| **Jupiter** | 827,328 | 0.0010 |
| **Saturn** | 660,962 | 0.0005 |
| **Uranus** | 457,943 | 0.0002 |
| **Neptune** | 380,995 | 0.0001 |

<img width="994" height="789" alt="Screenshot 2026-04-14 195012" src="https://github.com/user-attachments/assets/2ae3f70c-14c0-471b-b15c-bc04bead9782" />


### Technical Note: Model Error vs. Visualization Effects
Deviations in predicted trajectories arise from both model and rendering factors:
1. **Chaotic Error Amplification:** The gravitational N-body system is highly sensitive to initial conditions. Small prediction errors compound over time during autoregressive rollout, particularly in the inner planetary orbits (as seen in the Mercury/Venus metrics above).
2. **Model Approximation Error:** The neural network learns a statistical mapping of state transitions rather than exact numerical integration,resulting in bounded but non-zero positional drift
3. **Visualization Transformations:** Nonlinear spatial compression (fractional power scaling) and 2D projection introduce additional perceptual distortion when mapping astronomical coordinates to screen space.

## Data Sources & Dependencies
This project utilizes NASA's **NAIF SPICE** toolkit to ensure high-fidelity planetary states. The following kernels are required for the physics engine to calculate accurate gravitational baselines:

* **LSK (naif0012.tls):** Leapseconds kernel for high-precision time conversion.
* **SPK (de432s.bsp):** Binary planetary ephemeris containing $X, Y, Z$ state vectors for the solar system.
* **PCK (gm_de431.tpc):** Planetary constants kernel containing $GM$ values for mass calculations.

## Future Work
* **Physics-Informed Neural Networks (PINNs):** Explore incorporating physical constraints based on Newtonian gravitational dynamics so the model better follows real orbital dynamics and remains stable over long prediction horizons.
* **Energy-Conservation:** Improve long-term orbital stability by replacing or augmenting the RK4 integrator with energy-conserving symplectic methods (e.g., Leapfrog integration) to reduce drift in simulated orbits
* **Improved Temporal Stability:** Incorporate Keplerian orbital structure or Lagrangian mechanics-inspired constraints to improve the physical realism and long-term stability of predicted orbital trajectories

## Getting Started
### Prerequisites
* Python 3.10+
### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Soldeusss/Simulacrum-A-Deep-Learning-N-Body-Simulator.git
   cd Simulacrum
2. ```bash
   pip install -r requirements.txt
3. ```bash
   python spaceGraphics.py
