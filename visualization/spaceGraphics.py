# The purpose of this program is to simulate a neural network based orbital system
# It renders planetary motion using AI-predicted state vectors and visualizes trajectories in real time

import pygame
import numpy as np
import joblib
import os
import sys
import math
from tensorflow.keras.models import load_model
from nasaData import real_states_array  # initial planetary state data used for simulation

# --- PYGAME SETUP ---
pygame.init()

# screen dimensions for simulation display
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("4D Orbital AI - Natural Solar System")

clock = pygame.time.Clock()

# fonts used for UI text rendering
font = pygame.font.SysFont("Arial", 18)
title_font = pygame.font.SysFont("Arial", 28, bold=True)

# colors used for background and UI elements
BLACK, WHITE, GRAY = (0, 0, 0), (255, 255, 255), (150, 150, 150)

# colors assigned to each planet for visualization
PLANET_COLORS = [
    (255, 215, 0), (176, 196, 222), (255, 165, 0), (65, 105, 225),
    (255, 69, 0), (221, 160, 221), (244, 164, 96), (224, 255, 255), (0, 0, 205)
]

# names of celestial bodies in simulation
PLANET_NAMES = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

# visual size of planets (not physically accurate)
PLANET_SIZES = [12, 3, 4, 4, 3, 8, 7, 5, 5]

# scaling factor used to compress astronomical distances into screen space
BASE_SCALE = (HEIGHT / 2.2) / 5e12

camera_x, camera_y = 0.0, 0.0  # used for panning view
zoom = 1.5  # controls zoom level of simulation

# the purpose of this section is to load the trained neural network and normalization tools

print("Loading AI Model and Scalers...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    # loads trained neural network model
    model = load_model(os.path.join(BASE_DIR, "solar_system_ai.keras"))

    # loads feature scaling used during training
    X_scaler = joblib.load(os.path.join(BASE_DIR, "X_scaler.pkl"))
    y_scaler = joblib.load(os.path.join(BASE_DIR, "y_scaler.pkl"))

except Exception as e:
    print(f"Failed to load model/scalers: {e}")
    sys.exit()

# converts initial planetary dataset into usable array format
real_states_array = np.array(real_states_array).reshape(-1, 6)

# number of bodies in the simulation
num_bodies = real_states_array.shape[0]

# queue that stores predicted frames from neural network
frame_queue = []

# keeps track of simulation time in days
global_day = 1

# maximum number of trail points stored per planet
MAX_TRAIL_LENGTH = 300

# stores past positions for each planet to draw orbits
trail_history = {i: [] for i in range(num_bodies)}

# generates background stars for visual effect
num_stars = 200
stars = [
    (np.random.randint(0, WIDTH),
     np.random.randint(0, HEIGHT),
     np.random.randint(1, 3))
    for _ in range(num_stars)
]

# purpose of this function is to convert real space coordinates into screen coordinates
def world_to_screen(x, y):

    # calculates distance from center of system
    distance = math.hypot(x, y)

    if distance == 0:
        scaled_x, scaled_y = 0, 0
    else:
        angle = math.atan2(y, x)

        # compresses large distances so planets fit on screen
        scaled_distance = (distance ** 0.35) * 0.02 * zoom

        # converts polar coordinates back to cartesian screen space
        scaled_x = scaled_distance * math.cos(angle)
        scaled_y = scaled_distance * math.sin(angle)

    # applies camera offset and centers simulation on screen
    screen_x = int(WIDTH / 2 + scaled_x + camera_x)
    screen_y = int(HEIGHT / 2 - scaled_y + camera_y)

    return screen_x, screen_y


# Draws planet glow effect
# purpose of this function is to visually render planets with a glow effect
def draw_glowing_planet(surface, color, pos, radius):

    # creates temporary surface for glow rendering
    glow_radius = radius * 3
    glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)

    # outer glow layer
    pygame.draw.circle(glow_surf, (*color, 30), (glow_radius, glow_radius), glow_radius)

    # inner glow layer
    pygame.draw.circle(glow_surf, (*color, 100), (glow_radius, glow_radius), int(radius * 1.5))

    # solid planet body
    pygame.draw.circle(glow_surf, color, (glow_radius, glow_radius), radius)

    # draw glow onto main screen
    surface.blit(glow_surf, (pos[0] - glow_radius, pos[1] - glow_radius))


# purpose of this function is to render text onto the screen
def draw_text(text, x, y, color=WHITE, is_title=False):

    # selects font type based on title flag
    f = title_font if is_title else font

    # renders text surface
    surface = f.render(text, True, color)

    # draws text to screen
    screen.blit(surface, (x, y))


# --- ROUGH ASTEROID OBJECT ---
# purpose of this section is to simulate a moving external gravitational object

Interloper_Mass = 6.4e23

# starting position of rogue object
Interloper_Pos = [4e12, 4e12, 0]

# constant velocity of object (linear motion approximation)
Interloper_Vel = [-100000, -110000, 0]


def generate_batch(start_day, batch_size=50):

    # extracts initial position and velocity of interloper
    start_x, start_y, start_z = Interloper_Pos
    vel_x, vel_y, vel_z = Interloper_Vel

    # calculates distance from interloper to each planet
    initial_distances = []
    for i in range(num_bodies):
        bx, by, bz = real_states_array[i][:3]
        initial_distances.append(np.linalg.norm([start_x - bx, start_y - by, start_z - bz]))

    # builds input vector for neural network
    raw_inputs = [Interloper_Mass, 1, start_x, start_y, start_z,
                  vel_x, vel_y, vel_z] + initial_distances

    X_base = np.array(raw_inputs, dtype=float)

    # applies log scaling to mass for stability
    X_base[0] = np.log10(Interloper_Mass)

    batch_inputs = []
    ghost_positions = []

    # generates prediction frames over time window
    for day in range(start_day, start_day + batch_size):

        temp = X_base.copy()
        temp[1] = day  # updates time step

        # computes deterministic motion of interloper
        gx = start_x + vel_x * day * 86400
        gy = start_y + vel_y * day * 86400
        gz = start_z + vel_z * day * 86400

        temp[2:5] = [gx, gy, gz]

        batch_inputs.append(temp)
        ghost_positions.append((gx, gy))

    # normalizes input data before prediction
    X_scaled = X_scaler.transform(np.array(batch_inputs))

    # runs neural network prediction
    y_scaled = model.predict(X_scaled, verbose=0)

    # converts prediction back to real scale
    predicted_states = y_scaler.inverse_transform(y_scaled)

    # packages results into frame format for rendering
    frames = [
        {
            "day": start_day + i,
            "y_real": predicted_states[i],
            "ghost_pos": ghost_positions[i]
        }
        for i in range(batch_size)
    ]

    return frames


# main loop
running = True
print("Starting simulation...")

while running:

    # clears screen for new frame
    screen.fill((5, 5, 15))

    # draws star background
    for star_x, star_y, star_size in stars:
        pygame.draw.circle(screen, (200, 200, 200), (star_x, star_y), star_size)

    # handles user input events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # zoom in/out using mouse wheel
        if event.type == pygame.MOUSEWHEEL:
            zoom *= 1.1 if event.y > 0 else 0.9

        # camera movement using mouse drag
        elif event.type == pygame.MOUSEMOTION:
            if pygame.mouse.get_pressed()[0]:
                dx, dy = event.rel
                camera_x += dx
                camera_y += dy

    # generates new frames when queue is low
    if len(frame_queue) < 10:
        frame_queue.extend(generate_batch(global_day, batch_size=50))
        global_day += 50

    # gets next frame for rendering
    frame = frame_queue.pop(0)

    # draws UI text
    draw_text(f"Day: {frame['day']}", 20, 20, is_title=True)
    draw_text("Scroll: Zoom | Drag: Pan", 20, 60, GRAY)
    draw_text(f"Zoom Level: {zoom:.2f}x", 20, 90, GRAY)

    # draws planets and orbital trails
    for i in range(num_bodies):

        px, py = frame['y_real'][i*6], frame['y_real'][i*6 + 1]

        trail_history[i].append((px, py))

        if len(trail_history[i]) > MAX_TRAIL_LENGTH:
            trail_history[i].pop(0)

        # draws orbital path
        screen_trail = [world_to_screen(tx, ty) for tx, ty in trail_history[i]]
        pygame.draw.lines(screen, PLANET_COLORS[i], False, screen_trail, 1)

        # draws planet
        sx, sy = world_to_screen(px, py)
        size = max(2, int(PLANET_SIZES[i] * math.sqrt(zoom)))

        draw_glowing_planet(screen, PLANET_COLORS[i], (sx, sy), size)

        # draws labels for selected planets
        if zoom > 2.0 or i in [0, 3, 5]:
            draw_text(PLANET_NAMES[i], sx + 15, sy - 15, WHITE)

    # draws rogue object
    gx, gy = frame['ghost_pos']
    screen_gx, screen_gy = world_to_screen(gx, gy)

    pygame.draw.circle(screen, (255, 50, 50), (screen_gx, screen_gy), 6)
    draw_text("Rogue Object", screen_gx + 15, screen_gy - 15, (255, 50, 50))

    pygame.display.flip()

    clock.tick(60)

pygame.quit()