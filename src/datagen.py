import numpy as np
import pandas as pd
import concurrent.futures
from physicsEngine import rungeKutta
from nasaData import real_states_array, real_masses_array

dt = 1800           # 1800 seconds (30 minutes)
totalSteps = 17520 # 1 year (365 days * 48 steps per day)
trials = 30000          

def simulate_universe(trial_id):
    # Use a unique seed for every trial to prevent duplicate universes
    rng = np.random.default_rng(trial_id)
    
    #randomizes interloper mass, position and velocity within a range
    interloperMass = rng.uniform(1.0e15, 6.0e24) 
    
    interloperX = rng.choice([-5.0e11, 5.0e11]) 
    interloperY = rng.uniform(-5.0e11, 5.0e11)
    interloperZ = rng.uniform(-5.0e11, 5.0e11) 
    
    interloperVX = rng.uniform(-50000, 50000)
    interloperVY = rng.uniform(-50000, 50000)
    interloperVZ = rng.uniform(-50000, 50000)  
    
    
    # # goal of this block is to check the distance between sun and interloper
    sunPos = real_states_array[0:3] # sun is third position
    dx = interloperX - sunPos[0]
    dy = interloperY - sunPos[1]
    dz = interloperZ - sunPos[2]

    initial_dist_to_sun = np.sqrt(dx**2 + dy**2 + dz**2)
    # spawn's object away from the sun if it was inside of it
    if initial_dist_to_sun < 7.0e8:
        interloperX += 1.0e9 

    # Calculate distances 
    distances = []
    for i in range(9):  # from Sun to Neptune
        start_idx = i * 6
        bodyX, bodyY, bodyZ = real_states_array[start_idx : start_idx + 3]
        d = np.sqrt((interloperX - bodyX)**2 + (interloperY - bodyY)**2 + (interloperZ - bodyZ)**2)
        distances.append(d)
    
    #  combines arrays
    trial_masses = np.concatenate([real_masses_array, [interloperMass]])
    interloperState = np.array([interloperX, interloperY, interloperZ, interloperVX, interloperVY, interloperVZ])
    current_state = np.concatenate([real_states_array, interloperState])
    
    # Randomize the timeline up to 365 days
    trial_steps = rng.integers(1, totalSteps)
    time_in_days = trial_steps / 48.0
    
    for step in range(trial_steps):
        current_state = rungeKutta(current_state, dt, trial_masses)
    
    #loop to recrod the outcomes
    finalPositionsAndVelocities = []
    for i in range(9):
        start_idx = i * 6
        planet_state = current_state[start_idx : start_idx + 6]
        finalPositionsAndVelocities.extend(planet_state)
    
    row_data = [interloperMass, time_in_days, interloperX, interloperY, interloperZ, interloperVX, interloperVY, interloperVZ] + distances + finalPositionsAndVelocities
    
    # print a progress update every 500 trials
    if (trial_id + 1) % 500 == 0:
        print(f"Completed {trial_id + 1} / {trials} universes...")
        
    return row_data

#  use parallel processing to reduce time it takes to go through each trial
if __name__ == '__main__':
    print(f"Starting {trials} trials.")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        csv_data = list(executor.map(simulate_universe, range(trials)))

    inputColumns = [
        'Interloper_Mass', 'Time_Days', 
        'Interloper_StartX', 'Interloper_StartY', 'Interloper_StartZ',
        'Interloper_VelX', 'Interloper_VelY', 'Interloper_VelZ',
        'Dist_to_Sun', 'Dist_to_Mercury', 'Dist_to_Venus', 'Dist_to_Earth', 'Dist_to_Mars',
        'Dist_to_Jupiter', 'Dist_to_Saturn', 'Dist_to_Uranus', 'Dist_to_Neptune'
    ]
                    
    planetNames = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']    
    outputColumns = [f'{name}_{axis}' for name in planetNames for axis in ['X', 'Y', 'Z', 'VX', 'VY', 'VZ']]

    columns = inputColumns + outputColumns
    df = pd.DataFrame(csv_data, columns=columns)
    df.to_csv("flybyData.csv", index=False)
    print("CSV created successfully.")