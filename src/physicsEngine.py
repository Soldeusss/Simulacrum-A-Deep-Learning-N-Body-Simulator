# The purpose of this module is to compute N-body gravitational motion
# It calculates accelerations using Newtonian gravity and integrates motion using RK4

import numpy as np

# universal gravitational constant used for Newtonian gravity calculations
G = 6.6743e-11

# purpose of this function is to compute derivatives of position and velocity
# returns dx/dt (velocity) and dv/dt (acceleration) for all bodies
def derivatives(bodyStates, mass):

    # determines number of bodies in the system (each body has 6 values)
    num_bodies = len(bodyStates) // 6

    # reshapes flat state vector into (num_bodies, 6)
    bodies = bodyStates.reshape((num_bodies, 6))

    # splits state into position and velocity components
    positions = bodies[:, 0:3]
    velocities = bodies[:, 3:6]

    # initializes acceleration array
    accelerations = np.zeros_like(positions)

    # computes gravitational acceleration for each body
    for i in range(num_bodies):

        acc_x, acc_y, acc_z = 0.0, 0.0, 0.0

        # compares body i against all other bodies
        for j in range(num_bodies):

            if i == j:
                continue  # skip self-interaction

            # calculates distance vector between body i and j
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]

            # computes scalar distance
            dist = np.sqrt(dx**2 + dy**2 + dz**2)

            # avoids division errors at extremely small distances
            if dist > 0:

                # computes gravitational acceleration magnitude
                # a = G * m / r^3 (vector form of Newton's law)
                magnitude = (G * mass[j]) / (dist**3)

                # accumulates acceleration contribution from body j
                acc_x += magnitude * dx
                acc_y += magnitude * dy
                acc_z += magnitude * dz

        # stores final acceleration for body i
        accelerations[i, 0] = acc_x
        accelerations[i, 1] = acc_y
        accelerations[i, 2] = acc_z

    # combines velocity and acceleration into full derivative vector
    stateDerivatives = np.concatenate((velocities, accelerations), axis=1)

    return stateDerivatives.flatten()


# purpose of this function is to integrate motion over time using RK4
# it advances the system state using weighted slope averaging
def rungeKutta(bodyStates, dt, mass):

    # computes slope at start of timestep
    k1 = derivatives(bodyStates, mass)

    # computes slope at midpoint using k1
    k2 = derivatives(bodyStates + 0.5 * dt * k1, mass)

    # computes slope at midpoint using k2
    k3 = derivatives(bodyStates + 0.5 * dt * k2, mass)

    # computes slope at end of timestep using k3
    k4 = derivatives(bodyStates + dt * k3, mass)

    # combines slopes into final state update
    newState = bodyStates + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return newState