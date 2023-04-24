'''
Here's an overview of the main components in the code:

- Import necessary libraries (numpy, torch, and matplotlib.pyplot) and set the device to either GPU or CPU, depending on availability.
- Define the grid size (nx and ny), time steps (nt), speed of light (c), time step size (dt), grid spacing (dx), permittivity (eps0), and permeability (mu0) constants.
- Initialize Ex, Ey, and Hz field tensors on the specified device (GPU or CPU).
- Define the gaussian_pulse function, which calculates the Gaussian pulse value at a given time (t). The pulse is later injected into the Ey field at the center of the grid as a source.
- Define the run_simulation function, which performs the FDTD update equations for the Ex, Ey, and Hz fields for the specified number of time steps (nt). At each time step, it updates the Hz field, followed by the Ex and Ey fields. Finally, it injects the Gaussian pulse into the Ey field at the center of the grid.
- Run the simulation by calling the run_simulation function with the specified parameters and tensors.
- Visualize the final Ex field using matplotlib.pyplot. The color map (RdBu) represents the values of the electric field (V/m) at the end of the simulation.

'''

import numpy as np
import torch
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Grid size
nx, ny = 200, 200

# Time steps
nt = 500

# Speed of light
c = 299792458

# Time step size (Courant condition)
dx = 1e-2
dt = dx / (2 * c)

# Permittivity and permeability
eps0 = 8.854e-12
mu0 = 4 * np.pi * 1e-7

# Set up tensors for Ex, Ey, and Hz fields
Ex = torch.zeros((ny, nx), device=device)
Ey = torch.zeros((ny, nx), device=device)
Hz = torch.zeros((ny, nx), device=device)

# def gaussian_pulse(t, t0, spread):
#     return torch.exp(-((t - t0) ** 2) / (2 * spread ** 2))

def gaussian_pulse(t, t0, spread, device):
    t = torch.tensor(t, device=device)  # Convert t to a tensor on the specified device
    return torch.exp(-((t - t0) ** 2) / (2 * spread ** 2))

def run_simulation(nt, Ex, Ey, Hz, dt, dx, eps0, mu0):
    # Time loop
    for t in range(nt):
        # Update magnetic field
        Hz[:-1, :-1] += (dt / mu0 / dx) * (Ex[:-1, 1:] - Ex[1:, :-1] - Ey[1:, :-1] + Ey[:-1, 1:])

        # Update electric fields
        Ex[1:, 1:] -= (dt / eps0 / dx) * (Hz[1:, 1:] - Hz[:-1, 1:])
        Ey[1:, 1:] += (dt / eps0 / dx) * (Hz[1:, 1:] - Hz[1:, :-1])

        # Inject Gaussian source
        t0 = 30 * dt
        spread = 10 * dt
        # pulse = gaussian_pulse(t * dt, t0, spread)
        pulse = gaussian_pulse(t * dt, t0, spread, device)
        Ey[ny // 2, nx // 2] += pulse

    return Ex, Ey, Hz

Ex, Ey, Hz = run_simulation(nt, Ex, Ey, Hz, dt, dx, eps0, mu0)

# Visualize the final Ex field
Ex_np = Ex.cpu().numpy()
plt.imshow(Ex_np, cmap='RdBu', extent=[0, nx * dx, 0, ny * dx], origin='lower')
plt.colorbar(label='Electric field (V/m)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Ex field at t = {} T'.format(nt))
plt.show()
