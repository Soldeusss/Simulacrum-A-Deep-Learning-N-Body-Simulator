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
Simulacrum is a hybrid physics and deep learning system that uses a neural network to approximate the motion of an N-body gravitational system.

The goal is to compare traditional physics-based simulation with a learned model that predicts how the system evolves over time.

https://github.com/user-attachments/assets/13265ec8-42cb-48f2-ad03-274a7e7a81b4

## Overview
Simulacrum simulates gravitational interactions between multiple bodies using Newtonian physics.

A physics engine generates training data using a 4th-order Runge-Kutta (RK4) integrator. A neural network is then trained to predict how the system evolves from one state to the next.

This allows the neural network to act as a faster approximation of the simulation once trained.

<img width="1189" height="698" alt="efficiency" src="https://github.com/user-attachments/assets/82c3e9a0-a7bf-4ebf-b968-1e84329e7bf5" />

## Key Features
* **High-Precision Physics:** Implements RK4 integration for planetary dynamics with state vectors sourced from NASA’s NAIF SPICE kernels via `spiceypy`.
* **Parallelized Data Pipeline:** Utilizes Python's concurrent.futures to run a Monte Carlo engine across parallel processes, generating a 32MB dataset of 30,000 unique gravitational flyby scenarios across 17,000+ timesteps per trial
* **Deep Neural Surrogate Model:** A 5-layer feed-forward neural network that maps 17 initial state features to 54 continuous kinematic variables, learning complex orbital interactions from simulated data
* **Real-Time Visualization:** Interactive PyGame environment for rendering planetary trajectories and AI predictions.

## Technical Architecture
### The Physics Engine
The baseline simulation calculates gravitational acceleration using Newton’s Universal Law of Gravitation:

$$F = G \frac{m_1 m_2}{r^2}$$

RK4 integration is used to update positions and velocities over time while maintaining numerical stability.

## Data Pipeline & Ingestion

To train the neural net, a data generation pipeline was made to simulate chaotic N-body interactions at scale:

- **Ingestion:**  
  Fetches and parses NASA NAIF SPICE kernels to establish initial conditions.

- **Simulation Engine:**  
  Generates randomized "rogue interloper" parameters (mass, velocity, trajectory).

- **Parallel Execution:**  
  Thousands of independent simulations are run in parallel to generate training data efficiently

### The Neural Network
* **Input Layer:** Interloper mass, time horizon, 3D position/velocity vectors, and initial scalar distances to all major celestial bodies.
* **Hidden Layers:** 5-layer Dense architecture with ReLU activation.
* **Output Layer (54 Variables):** Multi-target regression predicting 6 kinematic state vectors (X, Y, Z, VX, VY, VZ) for 9 distinct celestial bodies.
* **Performance:**
    * **Mean Position Error (MAE):** 1,236,958 km (on a solar-system scale).
    * **Global Mean Relative Error:** 0.0097.
    * **Time Complexity:** Constant-time neural network inference per update, replacing step-by-step physics simulation with direct prediction of the next system state.

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

<br><br>

<img width="1065" height="660" alt="Screenshot 2026-04-23 142809" src="https://github.com/user-attachments/assets/caf24d7f-5bc2-47a3-89a2-80ed739e1c88" />

## Model Limitations

The model uses a simple neural network to approximate orbital motion.
It performs well over short time spans but becomes less accurate over longer simulations. This is mainly due to small errors accumulating over repeated predictions, especially in more sensitive orbital regions.

Some additional differences between predicted and true trajectories also come from the fact that the model is learning an approximation of the system rather than exact physics, and from simplifications made in visualizing 3D motion in 2D.

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
* Python 3.10 to 3.13: TensorFlow is currently most stable within this range.
### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Soldeusss/Simulacrum-A-Deep-Learning-N-Body-Simulator.git
   cd Simulacrum
2. Set up a Virtual Environment:
   ```bash 
   
   windows:
   
   python -m venv venv
   .\venv\Scripts\activate

   mac:
   python3 -m venv venv
   source venv/bin/activate
   
   install dependencies:
   pip install -r requirements.txt
3. Run the simulator
   ```bash
   python nasaData.py #get files from nasa 
   python spaceGraphics.py
   
