# The purpose of this script is to load real planetary data from NASA SPICE kernels
# It extracts positions, velocities, and masses for use in the physics simulation

import os
import urllib.request
import numpy as np
import spiceypy as spice

# NASA kernel files required for accurate planetary ephemeris data
kernels = {
    "naif0012.tls": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
    "de432s.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de432s.bsp",
    "gm_de431.tpc": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de431.tpc"
}

# purpose of this block is to ensure all required NASA data files exist locally
print("Checking for NASA data files...")

for filename, url in kernels.items():
    if not os.path.exists(filename):
        print(f"Downloading {filename}")
        urllib.request.urlretrieve(url, filename)

# loads kernel files into SPICE memory for querying planetary data
for filename in kernels.keys():
    spice.furnsh(filename)

# sets simulation start time
# converts human-readable UTC time into ephemeris time (seconds since J2000)
et = spice.str2et("2026-03-19 12:00:00 UTC")

# NAIF planetary IDs used by SPICE system
# Sun=10, Mercury=1, Venus=2, Earth=399, Mars=4, Jupiter=5, Saturn=6, Uranus=7, Neptune=8
planet_ids = [10, 1, 2, 399, 4, 5, 6, 7, 8]
names = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]

# gravitational constant used to convert GM values into mass
G = 6.6743e-11

# stores extracted state vectors and masses
real_states = []
real_masses = []

print("\n--- NASA DATA EXTRACTED ---")

for i, body_id in enumerate(planet_ids):

    # purpose of this call is to get position and velocity from SPICE kernel
    # returns state vector in km and km/s
    state, _ = spice.spkgeo(body_id, et, 'J2000', 0)

    # converts from kilometers to meters for consistency with physics engine
    state_in_meters = np.array(state) * 1000.0
    real_states.extend(state_in_meters)

    # retrieves GM value (gravitational parameter) from SPICE
    # converts GM into mass using G constant
    _, gm_values = spice.bodvrd(str(body_id), 'GM', 1)

    mass_kg = (gm_values[0] * 1e9) / G  # conversion from km^3 to m^3 included

    real_masses.append(mass_kg)

    # prints extracted values for verification
    print(f"{names[i]} Mass: {mass_kg:.2e} kg")
    print(f"{names[i]} X Position: {state_in_meters[0]:.2e} m")


# converts lists into numpy arrays for physics engine usage
real_states_array = np.array(real_states)
real_masses_array = np.array(real_masses)