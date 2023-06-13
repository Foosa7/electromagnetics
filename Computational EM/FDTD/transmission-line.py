import numpy as np
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
c = 3e8
fr4_er = 4.4
imp0 = 377.0

# Grid size and time steps
nx, ny = 200, 200
nt = 500

# Time step size (Courant condition)
dx = 1e-2
dt = dx / (2 * c)

# Initialize fields
Ex = torch.zeros((ny, nx), device=device)
Ey = torch.zeros((ny, nx), device=device)
Hz = torch.zeros((ny, nx), device=device)

# Permittivity and permeability
eps0 = 8.854e-12
mu0 = 4 * np.pi * 1e-7
eps_r = torch.ones((ny, nx), device=device) * fr4_er
eps_r[:, : nx // 2] = 1
eps = eps0 * eps_r

# Gaussian pulse parameters
t0 = 30 * dt
spread = 10 * dt

# TF/SF boundaries
tfsf_min_x = nx // 4
tfsf_max_x = 3 * nx // 4
tfsf_min_y = ny // 4
tfsf_max_y = 3 * ny // 4

# Run simulation
for t in range(nt):
    # Update magnetic field
    Hz[:-1, :-1] += (dt / mu0 / dx) * (Ex[:-1, 1:] - Ex[1:, :-1] - Ey[1:, :-1] + Ey[:-1, 1:])

    # Update electric fields
    Ex[1:, 1:] -= (dt / eps[1:, 1:] / dx) * (Hz[1:, 1:] - Hz[:-1, 1:])
    Ey[1:, 1:] += (dt / eps[1:, 1:] / dx) * (Hz[1:, 1:] - Hz[1:, :-1])

    # Inject Gaussian source
    # pulse = torch.exp(-((t - t0) ** 2) / (2 * spread ** 2)).to(device)
    pulse = torch.exp(-((torch.tensor(t, device=device) - t0) ** 2) / (2 * spread ** 2))
    Ey[ny // 2, nx // 2] += pulse

    # TF/SF corrections for Ex field
    Ex[tfsf_min_y - 1, tfsf_min_x : tfsf_max_x] -= dt / (dx * eps[tfsf_min_y - 1, tfsf_min_x : tfsf_max_x]) * pulse
    Ex[tfsf_max_y, tfsf_min_x : tfsf_max_x] += dt / (dx * eps[tfsf_max_y, tfsf_min_x : tfsf_max_x]) * pulse

    # TF/SF corrections for Ey field
    Ey[tfsf_min_y : tfsf_max_y, tfsf_min_x - 1] -= dt / (dx * eps[tfsf_min_y : tfsf_max_y, tfsf_min_x - 1]) * pulse

# Visualize the final Ex field
Ex_np = Ex.cpu().numpy()
plt.imshow(Ex_np, cmap='RdBu', extent=[0, nx * dx, 0, ny * dx], origin='lower')
plt.colorbar(label='Electric field (V/m)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Electric field Ex')
plt.show()

# Frequency range for S-parameters calculation
freqs = np.linspace(1e9, 10e9, 100)
s11 = []
s21 = []

# Calculate S-parameters
for freq in freqs:
    E_inc = np.sin(2 * np.pi * freq * np.arange(nt) * dt)
    Ex_fft = np.fft.fft(Ex_np[ny // 2, :])
    E_inc_fft = np.fft.fft(E_inc)
    freqs_fft = np.fft.fftfreq(nt, dt)

    # Find the nearest frequency index in the FFT array
    idx = np.argmin(np.abs(freqs_fft - freq))

    # Measure the reflected and transmitted fields
    refl_field = Ex_fft[idx] / E_inc_fft[idx]
    trans_field = Ex_fft[idx] / E_inc_fft[idx]

    # Calculate S11 and S21
    s11.append(-refl_field)
    s21.append(trans_field)

# Plot the S-parameters
plt.figure()
plt.plot(freqs / 1e9, 20 * np.log10(np.abs(s11)), label="S11")
plt.plot(freqs / 1e9, 20 * np.log10(np.abs(s21)), label="S21")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Magnitude (dB)")
plt.legend()
plt.grid()
plt.title("S-parameters of a Transmission Line on FR4 using FDTD")
plt.show()
