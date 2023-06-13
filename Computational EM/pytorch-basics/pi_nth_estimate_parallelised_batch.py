import torch

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def estimate_pi(n, batch_size, device):
    total_points_in_circle = 0
    total_points = 0

    for i in range(0, n, batch_size):
        batch_points = min(batch_size, n - i)

        x = torch.rand(batch_points, device=device)
        y = torch.rand(batch_points, device=device)

        distance = x ** 2 + y ** 2
        points_in_circle = torch.sum(distance <= 1)

        total_points_in_circle += points_in_circle.item()
        total_points += batch_points

    pi = 4 * (total_points_in_circle / total_points)
    return pi

# Set the number of iterations for estimation and batch size
n = 100_000_000
batch_size = 99_000_000

# Estimate pi
pi_est = estimate_pi(n, batch_size, device)
print("Estimated pi:", pi_est)
