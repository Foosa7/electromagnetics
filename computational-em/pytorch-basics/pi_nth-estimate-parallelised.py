import torch

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def estimate_pi(n, device):
    x = torch.rand(n, device=device)
    y = torch.rand(n, device=device)

    distance = x ** 2 + y ** 2
    points_in_circle = torch.sum(distance <= 1)

    pi = 4 * (points_in_circle / n)
    return pi

# Set the number of iterations for estimation
n = 100_000_000

# Estimate pi
pi_est = estimate_pi(n, device)
print("Estimated pi:", pi_est)
